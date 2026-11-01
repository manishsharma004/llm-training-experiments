"""Where a training step will run, and whether an RX 5500 can host it.

PyTorch on AMD uses the CUDA API on top of ROCm (``torch.cuda``, ``torch.version.hip``).
The RX 5500 is Navi 14, architecture gfx1012. Current generic ROCm wheels often
omit that target. A gfx1012 build sees the card directly. A gfx1030 build can
be pointed at it with ``HSA_OVERRIDE_GFX_VERSION=10.3.0``, which this process
sets only when the lab is asked for compatibility mode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Navi 14 PCI ids used by the RX 5500 family.
RX5500_DEVICE_IDS = {
    "0x7340",
    "0x7341",
    "0x7343",
    "0x7347",
    "0x734f",
}
RX5500_VRAM_BYTES = 4 * 1024**3
GFX1012_TORCH = "torch[device-gfx1012]==2.14.0+rocm10.1.0"
GFX1012_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"


def scan_pci() -> list[dict]:
    """AMD GPUs visible to the kernel, without importing PyTorch."""

    found = []
    root = Path("/sys/class/drm")
    if not root.exists():
        return found
    seen: set[str] = set()
    for card in sorted(root.glob("card[0-9]")):
        device = card / "device"
        vendor_path = device / "vendor"
        if not vendor_path.exists():
            continue
        vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        if vendor != "0x1002":
            continue
        pci = device.resolve().name
        if pci in seen:
            continue
        seen.add(pci)
        device_id = (device / "device").read_text(encoding="utf-8").strip().lower() if (device / "device").exists() else ""
        found.append(
            {
                "pci": pci,
                "vendor": vendor,
                "device": device_id,
                "rx5500": device_id in RX5500_DEVICE_IDS,
            }
        )
    return found


def memory_budget(
    *,
    parameters: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    vram_bytes: int = RX5500_VRAM_BYTES,
) -> dict:
    """A conservative fp32 estimate the lab can show next to a 4GB card.

    Weights, Adam moments, and gradients are exact multiples of the parameter
    count. Activations are an upper bound: residual stream copies plus the
    attention score matrix, which is what grows when context gets long.
    """

    weights = parameters * 4
    adam = parameters * 8
    gradients = parameters * 4
    residual = batch_size * block_size * n_embd * 4 * n_layer * 8
    scores = batch_size * n_head * block_size * block_size * 4 * n_layer
    activations = residual + scores
    total = weights + adam + gradients + activations
    return {
        "precision": "fp32",
        "weights_bytes": weights,
        "optimizer_bytes": adam,
        "gradient_bytes": gradients,
        "activation_bytes": activations,
        "total_bytes": total,
        "vram_bytes": vram_bytes,
        "fits": total < int(vram_bytes * 0.75),
        "headroom_bytes": vram_bytes - total,
    }


def _preferred_cuda_index(torch_module) -> int:
    """Prefer the RX 5500 when an iGPU is also enumerated."""

    chosen = 0
    for index in range(torch_module.cuda.device_count()):
        if "5500" in torch_module.cuda.get_device_name(index).lower():
            return index
    return chosen


def probe(parameters: int = 1_035_008, batch_size: int = 16, block_size: int = 128) -> dict:
    """Describe the GPU this process can actually train on."""

    pci = scan_pci()
    rx5500 = any(row["rx5500"] for row in pci)
    info: dict = {
        "pci": pci,
        "rx5500_present": rx5500,
        "torch_gpu": False,
        "backend": "cpu",
        "name": "CPU",
        "hip": None,
        "total_vram_bytes": RX5500_VRAM_BYTES if rx5500 else None,
        "override": os.environ.get("HSA_OVERRIDE_GFX_VERSION") or None,
        "gpu_blocker": None,
        "arch": None,
        "index": None,
        "install": _install_hint(),
        "guidance": _guidance(rx5500),
    }
    try:
        import torch
    except ImportError as exc:
        info["gpu_blocker"] = f"PyTorch is not installed ({exc})."
        info["budget"] = memory_budget(
            parameters=parameters,
            batch_size=batch_size,
            block_size=block_size,
            n_layer=5,
            n_embd=128,
            n_head=4,
            vram_bytes=info["total_vram_bytes"] or RX5500_VRAM_BYTES,
        )
        return info

    hip = getattr(torch.version, "hip", None)
    info["hip"] = hip
    if torch.cuda.is_available():
        index = _preferred_cuda_index(torch)
        properties = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        info.update(
            {
                "torch_gpu": True,
                "backend": "rocm" if hip else "cuda",
                "name": name,
                "index": index,
                "arch": getattr(properties, "gcnArchName", None),
                "total_vram_bytes": int(properties.total_memory),
                "gpu_blocker": None,
            }
        )
        if "5500" in name.lower():
            info["rx5500_present"] = True
    else:
        info["gpu_blocker"] = _blocker(rx5500, hip, torch_installed=True)
    vram = info["total_vram_bytes"] or RX5500_VRAM_BYTES
    info["budget"] = memory_budget(
        parameters=parameters,
        batch_size=batch_size,
        block_size=block_size,
        n_layer=5,
        n_embd=128,
        n_head=4,
        vram_bytes=vram,
    )
    from tinyllm.sensors import find_gpu_device, read_gpu

    gpu_device = find_gpu_device()
    info["sensors"] = None if gpu_device is None else read_gpu(gpu_device)
    info["precision"] = "fp32"
    info["precision_reason"] = (
        "The million-parameter preset plus AdamW is a few tens of megabytes in fp32. "
        "fp32 also avoids half-precision kernels, which are unreliable on gfx1012."
    )
    return info


def resolve(preference: str):
    """Return a torch device. ``gpu`` fails with a readable error when none is visible."""

    import torch

    choice = preference if preference in {"auto", "cpu", "gpu"} else "auto"
    if choice == "cpu":
        return torch.device("cpu"), probe()
    info = probe()
    if info["torch_gpu"]:
        index = info.get("index") or 0
        return torch.device(f"cuda:{index}"), info
    if choice == "gpu":
        raise RuntimeError(info["gpu_blocker"] or "No GPU is visible to PyTorch.")
    return torch.device("cpu"), info


def _blocker(rx5500: bool, hip: str | None, torch_installed: bool) -> str:
    if not torch_installed:
        return "PyTorch is not installed."
    if rx5500 and not hip:
        return (
            "The kernel sees an RX 5500, but this PyTorch build has no ROCm. "
            "Install a gfx1012 wheel, or a ROCm build and enable RX 5500 compatibility mode."
        )
    if rx5500 and hip:
        return (
            "ROCm is installed but it did not expose the RX 5500. "
            "Use a gfx1012 PyTorch build, or retry with compatibility mode "
            "(HSA_OVERRIDE_GFX_VERSION=10.3.0) if this wheel contains gfx1030 kernels."
        )
    if hip:
        return "ROCm is installed, but torch.cuda.is_available() is false."
    return "PyTorch has no visible GPU. Training will use the CPU until a ROCm or CUDA build sees one."


def _install_hint() -> str:
    return (
        f".venv/bin/python -m pip install --index-url {GFX1012_INDEX} "
        f"--extra-index-url https://pypi.org/simple \"{GFX1012_TORCH}\""
    )


def _guidance(rx5500: bool) -> list[str]:
    lines = [
        "The RX 5500 / 5500 XT is Navi 14 (gfx1012) with 4GB of memory.",
        "Preferred: install AMD's gfx1012 wheel (torch 2.14.0+rocm10.1.0 with the device-gfx1012 extra). The lab's device probe prints the exact pip command when the card is present and this process is still on a CPU build.",
        "Fallback: on a ROCm wheel that ships gfx1030 kernels, compatibility mode sets HSA_OVERRIDE_GFX_VERSION=10.3.0 for the training process only. Those kernels were compiled for a different chip and can be unstable.",
        "The 1,035,008-parameter model stays in fp32. Weights, gradients, and Adam states are about 16MB, so the 4GB limit is not what binds this preset.",
    ]
    if not rx5500:
        lines.append("This machine's PCI scan did not find an RX 5500. Auto mode uses a GPU when PyTorch can see one, and the CPU otherwise.")
    return lines


def main() -> int:
    print(json.dumps(probe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
