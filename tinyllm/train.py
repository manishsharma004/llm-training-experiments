"""Pretraining and supervised finetuning loops.

Pretraining predicts every next byte of the harbor prose.
Finetuning uses the same weights but the loss is computed only where the
label is a completion token. Prompt tokens stay in the context so the model
can condition on the question.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from tinyllm.checkpoint import load_model, save_checkpoint
from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.corpus import PROBES, finetune_examples, prose_corpus
from tinyllm.data import encode_corpus, finetune_batch, pretrain_batch
from tinyllm.model import TinyLM
from tinyllm.schedule import learning_rate
from tinyllm.status import read_status, write_status
from tinyllm.tokenizer import decode, encode


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _device() -> torch.device:
    return torch.device("cpu")


def _combined_metrics(status_path: Path | None, metrics: list[dict], phase: str) -> list[dict]:
    """Keep metrics from earlier phases when a pipeline reuses one status file."""

    prior: list[dict] = []
    if status_path is not None:
        old = read_status(status_path) or {}
        prior = [row for row in old.get("metrics", []) if row.get("phase") != phase]
    return prior + metrics


def _probe(model: TinyLM, prompts: list[str]) -> list[dict[str, str]]:
    model.eval()
    rows = []
    for prompt in prompts:
        ids = encode(prompt) or [10]
        tokens = torch.tensor([ids], dtype=torch.long)
        tokens = tokens[:, -model.config.block_size :]
        out = model.generate(tokens, max_new_tokens=24, temperature=0.0)
        rows.append({"prompt": prompt, "completion": decode(out[0].tolist())})
    model.train()
    return rows


def _run(
    *,
    phase: str,
    model: TinyLM,
    model_config: ModelConfig,
    train_config: TrainConfig,
    out: Path,
    status_path: Path | None,
    init_loss: float | None = None,
) -> dict:
    device = _device()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )
    stream = encode_corpus(prose_corpus(train_config.corpus_repeats))
    examples = finetune_examples()
    metrics: list[dict] = []
    last_loss = float(init_loss) if init_loss is not None else math.nan
    probes: list[dict[str, str]] = []

    for step in range(1, train_config.steps + 1):
        lr = learning_rate(
            step,
            train_config.steps,
            train_config.learning_rate,
            train_config.min_learning_rate,
            train_config.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if phase == "pretrain":
            inputs, labels = pretrain_batch(stream, model_config.block_size, train_config.batch_size, device)
            _logits, loss = model(inputs, labels=labels)
        elif phase == "finetune":
            inputs, labels, key_padding = finetune_batch(
                examples,
                model_config.block_size,
                train_config.batch_size,
                device,
            )
            _logits, loss = model(inputs, labels=labels, key_padding=key_padding)
        else:
            raise ValueError(f"unknown phase {phase}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        last_loss = float(loss.item())
        record = {"step": step, "loss": round(last_loss, 4), "lr": lr, "phase": phase}
        if step == 1 or step == train_config.steps or step % train_config.log_every == 0:
            metrics.append(record)
        if step == 1 or step == train_config.steps or step % train_config.sample_every == 0:
            probes = _probe(model, PROBES)

        if status_path is not None:
            write_status(
                status_path,
                {
                    "state": "running",
                    "phase": phase,
                    "step": step,
                    "totalSteps": train_config.steps,
                    "loss": last_loss,
                    "lr": lr,
                    "metrics": _combined_metrics(status_path, metrics, phase),
                    "probes": probes,
                    "checkpoint": str(out),
                },
            )

    save_checkpoint(out, model, model_config, train_config, train_config.steps, last_loss, phase)
    summary = {
        "state": "done",
        "phase": phase,
        "step": train_config.steps,
        "totalSteps": train_config.steps,
        "loss": last_loss,
        "metrics": _combined_metrics(status_path, metrics, phase),
        "probes": probes,
        "checkpoint": str(out),
    }
    if status_path is not None:
        write_status(status_path, summary)
    return summary


def pretrain(
    model_config: ModelConfig,
    train_config: TrainConfig,
    out: Path,
    status_path: Path | None = None,
) -> dict:
    _set_seed(train_config.seed)
    model = TinyLM(model_config)
    return _run(
        phase="pretrain",
        model=model,
        model_config=model_config,
        train_config=train_config,
        out=out,
        status_path=status_path,
    )


def finetune(
    init: Path,
    train_config: TrainConfig,
    out: Path,
    status_path: Path | None = None,
) -> dict:
    _set_seed(train_config.seed + 1)
    device = _device()
    model, blob = load_model(init, device)
    model_config = ModelConfig.from_dict(blob["model_config"])
    return _run(
        phase="finetune",
        model=model,
        model_config=model_config,
        train_config=train_config,
        out=out,
        status_path=status_path,
        init_loss=blob.get("loss"),
    )
