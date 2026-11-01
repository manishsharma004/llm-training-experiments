"""Power, memory, and temperature from the AMDGPU sysfs the kernel already exposes.

amdgpu reports power in microwatts and temperature in millidegrees. There is no
energy counter on this card, so watt-hours are the trapezoid of the power
samples taken between training steps. VRAM figures are the whole card, which
includes the display, plus the bytes this process has allocated when PyTorch
is asked.
"""

from __future__ import annotations

import time
from pathlib import Path

from tinyllm.device import RX5500_DEVICE_IDS


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def find_gpu_device(drm_root: Path | None = None) -> Path | None:
    """PCI device directory for the RX 5500, or the first amdgpu card."""

    root = drm_root or Path("/sys/class/drm")
    if not root.exists():
        return None
    fallback: Path | None = None
    seen: set[str] = set()
    for card in sorted(root.glob("card[0-9]")):
        device = card / "device"
        vendor = _read_text(device / "vendor").lower()
        if vendor != "0x1002":
            continue
        resolved = str(device.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        device_id = _read_text(device / "device").lower()
        hwmon = next(device.glob("hwmon/hwmon*"), None)
        if hwmon is None or _read_text(hwmon / "name") != "amdgpu":
            continue
        if device_id in RX5500_DEVICE_IDS:
            return device
        if fallback is None:
            fallback = device
    return fallback


def read_gpu(device: Path) -> dict:
    """One sample. Missing files become null rather than a failed run."""

    hwmon = next(device.glob("hwmon/hwmon*"), None)
    power_uw = _read_int(hwmon / "power1_average") if hwmon else None
    cap_uw = _read_int(hwmon / "power1_cap") if hwmon else None
    temperatures: dict[str, float] = {}
    if hwmon is not None:
        for index in range(1, 6):
            label = _read_text(hwmon / f"temp{index}_label")
            raw = _read_int(hwmon / f"temp{index}_input")
            if not label or raw is None or raw == 0:
                continue
            temperatures[label] = raw / 1000.0
    return {
        "pci": device.resolve().name,
        "power_w": None if power_uw is None else power_uw / 1_000_000.0,
        "power_cap_w": None if cap_uw is None else cap_uw / 1_000_000.0,
        "vram_used_bytes": _read_int(device / "mem_info_vram_used"),
        "vram_total_bytes": _read_int(device / "mem_info_vram_total"),
        "temperature_c": temperatures,
    }


class _Stat:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.last: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.last = value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def raise_peak(self, value: float) -> None:
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def export(self, digits: int | None) -> dict | None:
        if self.count == 0 or self.minimum is None or self.maximum is None or self.last is None:
            return None
        average = self.total / self.count

        def num(value: float) -> float | int:
            if digits is None:
                return int(round(value))
            return round(value, digits)

        return {"min": num(self.minimum), "avg": num(average), "max": num(self.maximum), "last": num(self.last)}


class GpuMeter:
    """Accumulate power, watt-hours, VRAM, and temperature across a run."""

    def __init__(self, device: Path, clock=time.perf_counter) -> None:
        self.device = device
        self.clock = clock
        self.power = _Stat()
        self.vram = _Stat()
        self.process = _Stat()
        self.temperatures: dict[str, _Stat] = {}
        self.power_cap_w: float | None = None
        self.vram_total_bytes: int | None = None
        self.energy_wh = 0.0
        self._previous_power: float | None = None
        self._previous_time: float | None = None
        self.started = clock()

    @classmethod
    def open(cls, drm_root: Path | None = None) -> GpuMeter | None:
        device = find_gpu_device(drm_root)
        if device is None:
            return None
        return cls(device)

    def sample(self, process_vram_bytes: int | None = None, process_vram_peak_bytes: int | None = None) -> dict:
        reading = read_gpu(self.device)
        now = self.clock()
        power = reading["power_w"]
        if power is not None:
            if self._previous_power is not None and self._previous_time is not None:
                hours = max(0.0, now - self._previous_time) / 3600.0
                self.energy_wh += (self._previous_power + power) / 2.0 * hours
            self._previous_power = power
            self._previous_time = now
            self.power.add(power)
        if reading["power_cap_w"] is not None:
            self.power_cap_w = reading["power_cap_w"]
        if reading["vram_used_bytes"] is not None:
            self.vram.add(reading["vram_used_bytes"])
        if reading["vram_total_bytes"] is not None:
            self.vram_total_bytes = reading["vram_total_bytes"]
        for label, celsius in reading["temperature_c"].items():
            self.temperatures.setdefault(label, _Stat()).add(celsius)
        if process_vram_bytes is not None:
            self.process.add(process_vram_bytes)
        if process_vram_peak_bytes is not None:
            self.process.raise_peak(process_vram_peak_bytes)
        return reading

    def summary(self) -> dict:
        temperatures = {}
        for label, stat in self.temperatures.items():
            exported = stat.export(1)
            if exported is not None:
                temperatures[label] = exported
        return {
            "pci": self.device.resolve().name,
            "samples": self.power.count or self.vram.count,
            "elapsed_s": round(max(0.0, self.clock() - self.started), 3),
            "power_w": self.power.export(2),
            "power_cap_w": None if self.power_cap_w is None else round(self.power_cap_w, 2),
            "energy_wh": round(self.energy_wh, 4),
            "vram_bytes": self.vram.export(None),
            "vram_total_bytes": self.vram_total_bytes,
            "process_vram_bytes": self.process.export(None),
            "temperature_c": temperatures,
        }
