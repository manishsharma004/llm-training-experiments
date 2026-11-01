"""Pretraining and supervised finetuning loops.

Pretraining predicts every next byte of the harbor prose.
Finetuning uses the same weights but the loss is computed only where the
label is a completion token. Prompt tokens stay in the context so the model
can condition on the question.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from tinyllm.checkpoint import load_model, save_checkpoint
from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.corpus import PROBES, finetune_examples, prose_corpus
from tinyllm.data import encode_corpus, finetune_batch, pretrain_batch
from tinyllm.device import resolve
from tinyllm.model import TinyLM
from tinyllm.schedule import learning_rate
from tinyllm.sensors import GpuMeter
from tinyllm.status import read_status, write_status
from tinyllm.tokenizer import decode, encode


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _load_bin(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) < 2:
        raise ValueError(f"{path} does not contain a training stream")
    return torch.tensor(list(raw), dtype=torch.long)


def _load_sft(path: Path) -> list[dict[str, str]]:
    import json

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({"prompt": str(record["prompt"]), "completion": str(record["completion"])})
    if not rows:
        raise ValueError(f"{path} has no finetune pairs")
    return rows


def _combined_metrics(status_path: Path | None, metrics: list[dict], phase: str) -> list[dict]:
    """Keep metrics from earlier phases when a pipeline reuses one status file."""

    prior: list[dict] = []
    if status_path is not None:
        old = read_status(status_path) or {}
        prior = [row for row in old.get("metrics", []) if row.get("phase") != phase]
    return prior + metrics


def _probe(model: TinyLM, prompts: list[str], device: torch.device) -> list[dict[str, str]]:
    model.eval()
    rows = []
    for prompt in prompts:
        ids = encode(prompt) or [10]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
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
    device_preference: str = "cpu",
    stream: torch.Tensor | None = None,
    val_stream: torch.Tensor | None = None,
    examples: list[dict[str, str]] | None = None,
    data_source: str = "harbor corpus",
) -> dict:
    device, device_info = resolve(device_preference)
    model.to(device)
    model.train()
    device_label = device_info.get("name") or device.type
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )
    if stream is None:
        stream = encode_corpus(prose_corpus(train_config.corpus_repeats))
    if examples is None:
        examples = finetune_examples()
    metrics: list[dict] = []
    last_loss = float(init_loss) if init_loss is not None else math.nan
    probes: list[dict[str, str]] = []
    tokens_trained = 0
    tokens_seen = 0
    run_started = time.perf_counter()
    meter = GpuMeter.open()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def compute_report() -> dict | None:
        allocated = peak = None
        if device.type == "cuda":
            allocated = int(torch.cuda.memory_allocated(device))
            peak = int(torch.cuda.max_memory_allocated(device))
        if meter is None:
            return None
        meter.sample(process_vram_bytes=allocated, process_vram_peak_bytes=peak)
        return meter.summary()

    compute = compute_report()

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

        started = time.perf_counter()
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

        seen = int(inputs.shape[0] * inputs.shape[1])
        trained = int((labels != -100).sum().item())
        tokens_seen += seen
        tokens_trained += trained

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        elapsed = max(time.perf_counter() - started, 1e-6)
        tokens_per_sec = (train_config.batch_size * model_config.block_size) / elapsed

        last_loss = float(loss.item())
        val_loss = None
        if val_stream is not None and (step == 1 or step == train_config.steps or step % train_config.sample_every == 0):
            val_loss = _validation_loss(model, val_stream, model_config.block_size, train_config.batch_size, device)
        record = {
            "step": step,
            "loss": round(last_loss, 4),
            "lr": lr,
            "phase": phase,
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        if val_loss is not None:
            record["val_loss"] = round(val_loss, 4)
        if step == 1 or step == train_config.steps or step % train_config.log_every == 0:
            metrics.append(record)
        if step == 1 or step == train_config.steps or step % train_config.sample_every == 0:
            probes = _probe(model, PROBES, device)
        compute = compute_report()

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
                    "device": device_label,
                    "arch": device_info.get("arch"),
                    "parameters": parameter_count,
                    "backend": device_info.get("backend"),
                    "precision": "fp32",
                    "tokensPerSec": round(tokens_per_sec, 1),
                    "tokensTrained": tokens_trained,
                    "tokensSeen": tokens_seen,
                    "elapsedS": round(time.perf_counter() - run_started, 3),
                    "dataSource": data_source,
                    "valLoss": val_loss,
                    "compute": compute,
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
        "device": device_label,
        "arch": device_info.get("arch"),
        "parameters": parameter_count,
        "backend": device_info.get("backend"),
        "precision": "fp32",
        "tokensPerSec": round(tokens_per_sec, 1),
        "tokensTrained": tokens_trained,
        "tokensSeen": tokens_seen,
        "elapsedS": round(time.perf_counter() - run_started, 3),
        "dataSource": data_source,
        "valLoss": val_loss,
        "compute": compute,
    }
    if status_path is not None:
        write_status(status_path, summary)
    return summary


def _validation_loss(
    model: TinyLM,
    stream: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        inputs, labels = pretrain_batch(stream, block_size, batch_size, device)
        _logits, loss = model(inputs, labels=labels)
    model.train()
    return float(loss.item())


def pretrain(
    model_config: ModelConfig,
    train_config: TrainConfig,
    out: Path,
    status_path: Path | None = None,
    device_preference: str = "cpu",
    data_dir: Path | None = None,
) -> dict:
    _set_seed(train_config.seed)
    model = TinyLM(model_config)
    stream = None
    val_stream = None
    source = "harbor corpus"
    if data_dir is not None:
        stream = _load_bin(data_dir / "train.bin")
        val_path = data_dir / "val.bin"
        if val_path.exists():
            val_stream = _load_bin(val_path)
        source = str(data_dir / "train.bin")
    return _run(
        phase="pretrain",
        model=model,
        model_config=model_config,
        train_config=train_config,
        out=out,
        status_path=status_path,
        device_preference=device_preference,
        stream=stream,
        val_stream=val_stream,
        data_source=source,
    )


def finetune(
    init: Path,
    train_config: TrainConfig,
    out: Path,
    status_path: Path | None = None,
    device_preference: str = "cpu",
    sft_path: Path | None = None,
) -> dict:
    _set_seed(train_config.seed + 1)
    device, _info = resolve(device_preference)
    model, blob = load_model(init, device)
    model_config = ModelConfig.from_dict(blob["model_config"])
    examples = _load_sft(sft_path) if sft_path is not None else None
    source = str(sft_path) if sft_path is not None else "harbor questions"
    return _run(
        phase="finetune",
        model=model,
        model_config=model_config,
        train_config=train_config,
        out=out,
        status_path=status_path,
        init_loss=blob.get("loss"),
        device_preference=device_preference,
        examples=examples,
        data_source=source,
    )
