"""Checkpoint load and save.

A checkpoint is the model weights plus the configs needed to rebuild it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM


def save_checkpoint(
    path: Path,
    model: TinyLM,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    loss: float,
    phase: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": train_config.__dict__.copy(),
            "step": step,
            "loss": loss,
            "phase": phase,
        },
        path,
    )


def summarize(path: Path) -> dict:
    """Read the configs stored beside the weights. The tensors stay on CPU."""

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    config = blob.get("model_config") or {}
    train = blob.get("train_config") or {}
    loss = blob.get("loss")
    return {
        "phase": blob.get("phase"),
        "step": blob.get("step"),
        "loss": None if loss is None else float(loss),
        "vocabSize": config.get("vocab_size"),
        "blockSize": config.get("block_size"),
        "batchSize": train.get("batch_size"),
    }


def load_model(path: Path, device: torch.device) -> tuple[TinyLM, dict]:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(blob["model_config"])
    model = TinyLM(config).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob
