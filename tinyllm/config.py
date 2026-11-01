"""Model and run settings.

The default preset is a decoder-only Transformer with an 8,192-piece
vocabulary trained on the book corpus. See `tinyllm.params.breakdown`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    vocab_size matches the byte-level BPE table in ``tinyllm/bpe.json``.
    """

    vocab_size: int = 8192
    block_size: int = 128
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.n_layer < 1:
            raise ValueError("n_layer must be at least 1")
        if self.n_head < 1:
            raise ValueError("n_head must be at least 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        cfg = cls(**known)
        cfg.validate()
        return cfg


@dataclass
class TrainConfig:
    """Optimization settings for one pretrain or finetune run."""

    steps: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-3
    min_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    grad_clip: float = 1.0
    seed: int = 7
    log_every: int = 1
    sample_every: int = 20
    corpus_repeats: int = 80

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.corpus_repeats < 1:
            raise ValueError("corpus_repeats must be at least 1")


def dump_config(model: ModelConfig, train: TrainConfig) -> str:
    payload = {"model": model.to_dict(), "train": asdict(train)}
    return json.dumps(payload, indent=2)
