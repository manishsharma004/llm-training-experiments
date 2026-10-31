"""Batches for next-token pretraining and prompt-masked finetuning."""

from __future__ import annotations

import torch

from tinyllm.tokenizer import encode


def encode_corpus(text: str) -> torch.Tensor:
    ids = encode(text)
    if len(ids) < 2:
        raise ValueError("corpus must contain at least 2 bytes")
    return torch.tensor(ids, dtype=torch.long)


def pretrain_batch(
    stream: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows. The target is the input shifted one token to the left."""

    if stream.numel() <= block_size:
        # Repeat a short corpus so every window is a full context.
        repeats = block_size * 2 // max(stream.numel(), 1) + 2
        stream = stream.repeat(repeats)
    high = stream.numel() - block_size - 1
    starts = torch.randint(0, high + 1, (batch_size,))
    inputs = torch.stack([stream[start : start + block_size] for start in starts])
    labels = torch.stack([stream[start + 1 : start + 1 + block_size] for start in starts])
    return inputs.to(device), labels.to(device)


def _fit_pair(prompt: str, completion: str, block_size: int) -> tuple[list[int], int]:
    """Return token ids and the number of leading prompt tokens.

    The sequence is truncated from the left of the prompt so the completion
    still fits inside `block_size + 1` tokens (input plus the last label).
    """

    prompt_ids = encode(prompt)
    completion_ids = encode(completion)
    if not completion_ids:
        completion_ids = [10]
    # Need room for the completion and at least one prompt byte when present.
    budget = block_size + 1
    overflow = len(prompt_ids) + len(completion_ids) - budget
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]
    ids = prompt_ids + completion_ids
    if len(ids) < 2:
        ids = ids + [10]
    return ids, len(prompt_ids)


def finetune_batch(
    examples: list[dict[str, str]],
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad examples in a batch and ignore prompt tokens in the loss.

    Label position i predicts token i+1. Prompt tokens are context, so every
    label that still points at a prompt token is set to -100. Padding is also
    ignored, and `key_padding` keeps attention off those positions.
    """

    if not examples:
        raise ValueError("finetune set is empty")
    chosen = [examples[int(i) % len(examples)] for i in torch.randint(0, len(examples), (batch_size,))]
    encoded = [_fit_pair(row["prompt"], row["completion"], block_size) for row in chosen]
    width = max(len(ids) - 1 for ids, _prompt_len in encoded)
    width = max(1, min(block_size, width))

    inputs = torch.full((batch_size, width), 0, dtype=torch.long)
    labels = torch.full((batch_size, width), -100, dtype=torch.long)
    key_padding = torch.ones((batch_size, width), dtype=torch.bool)

    for row, (ids, prompt_len) in enumerate(encoded):
        tokens = ids[: width + 1]
        length = len(tokens) - 1
        inputs[row, :length] = torch.tensor(tokens[:-1], dtype=torch.long)
        target = torch.tensor(tokens[1:], dtype=torch.long)
        # Supervised positions predict a completion token.
        # token index j+1 is in the completion when j+1 >= prompt_len.
        for index in range(length):
            predicts = index + 1
            if predicts >= prompt_len:
                labels[row, index] = target[index]
        key_padding[row, :length] = False

    return inputs.to(device), labels.to(device), key_padding.to(device)
