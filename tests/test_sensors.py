from pathlib import Path

from tinyllm.sensors import GpuMeter


def _card(root: Path, power_uw: int, vram: int, edge_mc: int, junction_mc: int) -> Path:
    device = root / "card1" / "device"
    hwmon = device / "hwmon" / "hwmon3"
    hwmon.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "device").write_text("0x7340\n", encoding="utf-8")
    (hwmon / "name").write_text("amdgpu\n", encoding="utf-8")
    (hwmon / "power1_average").write_text(f"{power_uw}\n", encoding="utf-8")
    (hwmon / "power1_cap").write_text("65000000\n", encoding="utf-8")
    (hwmon / "temp1_label").write_text("edge\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text(f"{edge_mc}\n", encoding="utf-8")
    (hwmon / "temp2_label").write_text("junction\n", encoding="utf-8")
    (hwmon / "temp2_input").write_text(f"{junction_mc}\n", encoding="utf-8")
    (hwmon / "temp3_label").write_text("mem\n", encoding="utf-8")
    (hwmon / "temp3_input").write_text("0\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text(f"{vram}\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("4278190080\n", encoding="utf-8")
    return device


def test_watt_hours_use_the_trapezoid_and_vram_tracks_extrema(tmp_path: Path):
    device = _card(tmp_path, power_uw=10_000_000, vram=1000, edge_mc=40_000, junction_mc=45_000)
    clock = {"now": 0.0}
    meter = GpuMeter(device, clock=lambda: clock["now"])
    meter.sample(process_vram_bytes=100)
    (device / "hwmon" / "hwmon3" / "power1_average").write_text("30000000\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("5000\n", encoding="utf-8")
    (device / "hwmon" / "hwmon3" / "temp1_input").write_text("60000\n", encoding="utf-8")
    clock["now"] = 3600.0
    meter.sample(process_vram_bytes=400, process_vram_peak_bytes=900)
    report = meter.summary()
    assert report["energy_wh"] == 20.0
    assert report["power_w"]["min"] == 10
    assert report["power_w"]["max"] == 30
    assert report["power_w"]["avg"] == 20
    assert report["power_cap_w"] == 65
    assert report["vram_bytes"]["min"] == 1000
    assert report["vram_bytes"]["max"] == 5000
    assert report["process_vram_bytes"]["max"] == 900
    assert report["process_vram_bytes"]["avg"] == 250
    assert report["temperature_c"]["edge"]["min"] == 40
    assert report["temperature_c"]["edge"]["max"] == 60
    assert "mem" not in report["temperature_c"]


def test_open_finds_the_rx5500_ahead_of_another_amd_card(tmp_path: Path):
    other = tmp_path / "card2" / "device" / "hwmon" / "hwmon0"
    other.mkdir(parents=True)
    (tmp_path / "card2" / "device" / "vendor").write_text("0x1002\n", encoding="utf-8")
    (tmp_path / "card2" / "device" / "device").write_text("0x1638\n", encoding="utf-8")
    (other / "name").write_text("amdgpu\n", encoding="utf-8")
    _card(tmp_path, power_uw=4_000_000, vram=10, edge_mc=50_000, junction_mc=51_000)
    meter = GpuMeter.open(tmp_path)
    assert meter is not None
    assert (meter.device / "device").read_text(encoding="utf-8").strip() == "0x7340"
