"""Greedy or sampled completion from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from tinyllm.checkpoint import load_model
from tinyllm.tokenizer import decode, encode


@torch.no_grad()
def complete(
    checkpoint: Path,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> str:
    torch_device = torch.device(device)
    model, _blob = load_model(checkpoint, torch_device)
    ids = encode(prompt) or [10]
    tokens = torch.tensor([ids], dtype=torch.long, device=torch_device)
    # Keep the prompt inside the context window.
    tokens = tokens[:, -model.config.block_size :]
    out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    text = decode(out[0].tolist())
    return text
