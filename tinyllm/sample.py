"""Greedy or sampled completion from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from tinyllm.checkpoint import load_model
from tinyllm.tokenizer import decode_for_vocab, encode_for_vocab


@torch.no_grad()
def next_logits(checkpoint: Path, prompt: str, device: str = "cpu") -> dict:
    """One forward pass. The returned vector is the next-token logit at the last prompt position."""

    torch_device = torch.device(device)
    model, _blob = load_model(checkpoint, torch_device)
    model.eval()
    vocab_size = model.config.vocab_size
    ids = encode_for_vocab(prompt, vocab_size) or [min(10, vocab_size - 1)]
    tokens = torch.tensor([ids], dtype=torch.long, device=torch_device)
    tokens = tokens[:, -model.config.block_size :]
    logits, _loss = model(tokens)
    last = logits[0, -1].detach().float().cpu()
    return {
        "prompt": prompt,
        "vocabSize": vocab_size,
        "tokenIds": tokens[0].tolist(),
        "logits": [round(float(value), 4) for value in last],
    }


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
