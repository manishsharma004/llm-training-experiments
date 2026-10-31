"""A few optimizer steps on a smaller model.

The full ~1M preset is covered by the parameter test. This file checks that
pretraining loss moves down and that finetuning builds a masked batch that
can backpropagate.
"""

from pathlib import Path

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.train import finetune, pretrain


def test_pretrain_loss_drops_and_finetune_runs(tmp_path: Path):
    model = ModelConfig(n_layer=2, n_embd=64, n_head=4, block_size=48, vocab_size=256, mlp_ratio=4)
    train = TrainConfig(
        steps=8,
        batch_size=8,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=1,
        sample_every=4,
        log_every=1,
        corpus_repeats=20,
        seed=3,
    )
    pre_path = tmp_path / "pretrain.pt"
    status = tmp_path / "status.json"
    summary = pretrain(model, train, pre_path, status)
    assert pre_path.exists()
    losses = [row["loss"] for row in summary["metrics"]]
    assert losses[-1] < losses[0]
    assert status.read_text().find('"state": "done"') != -1

    ft = TrainConfig(
        steps=3,
        batch_size=4,
        learning_rate=3e-3,
        min_learning_rate=1e-3,
        warmup_steps=0,
        sample_every=3,
        log_every=1,
        seed=3,
    )
    ft_summary = finetune(pre_path, ft, tmp_path / "sft.pt", tmp_path / "ft.json")
    assert ft_summary["state"] == "done"
    assert ft_summary["probes"]
