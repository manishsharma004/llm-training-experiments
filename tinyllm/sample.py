"""Greedy or sampled completion from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from tinyllm.checkpoint import load_model
from tinyllm.tokenizer import decode_for_vocab, encode_for_vocab


@torch.no_grad()
def complete(
    checkpoint: Path,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> dict:
    torch_device = torch.device(device)
    model, _blob = load_model(checkpoint, torch_device)
    vocab_size = model.config.vocab_size
    ids = encode_for_vocab(prompt, vocab_size) or [min(10, vocab_size - 1)]
    tokens = torch.tensor([ids], dtype=torch.long, device=torch_device)
    # Keep the prompt inside the context window.
    tokens = tokens[:, -model.config.block_size :]
    out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    text = decode_for_vocab(out[0].tolist(), vocab_size)
    return {"prompt": prompt, "text": text, "vocabSize": vocab_size}
