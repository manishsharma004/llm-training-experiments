"""Pause between finished steps and continue from that step."""

import json
import time
from pathlib import Path
from threading import Thread

import torch

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.train import continue_run, pretrain


def _tiny() -> tuple[ModelConfig, TrainConfig]:
    model = ModelConfig(vocab_size=8192, n_layer=1, n_embd=32, n_head=4, block_size=16, mlp_ratio=2)
    train = TrainConfig(
        steps=3,
        batch_size=2,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=0,
        sample_every=100,
        log_every=1,
        corpus_repeats=1,
        seed=3,
    )
    return model, train


def _wait_for(path: Path, state: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if path.exists():
            last = path.read_text(encoding="utf-8")
            payload = json.loads(last)
            if payload.get("state") == state:
                return payload
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {state}: {last}")


def test_pause_holds_the_step_and_disk_resume_matches(tmp_path: Path):
    model, train = _tiny()
    status_path = tmp_path / "status.json"
    control = tmp_path / "control.json"
    out = tmp_path / "model.pt"
    control.write_text(json.dumps({"command": "pause"}), encoding="utf-8")
    errors: list[BaseException] = []

    def run() -> None:
        try:
            pretrain(model, train, out, status_path, device_preference="cpu")
        except BaseException as exc:  # noqa: BLE001 — the assertion reports the thread error
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    try:
        paused = _wait_for(status_path, "paused", 90)
        assert paused["step"] == 1
        assert "Resume continues at step 2" in paused["message"]
        time.sleep(0.6)
        held = json.loads(status_path.read_text(encoding="utf-8"))
        assert held["state"] == "paused"
        assert held["step"] == 1
        blob = tmp_path / "resume.pt"
        assert blob.exists()
        saved = torch.load(blob, map_location="cpu", weights_only=False)
        assert saved["step"] == 1
        assert saved["optimizer"]["state"]
        copy = tmp_path / "resume-copy.pt"
        copy.write_bytes(blob.read_bytes())
        control.write_text(json.dumps({"command": "run"}), encoding="utf-8")
        thread.join(120)
        assert errors == []
        done = json.loads(status_path.read_text(encoding="utf-8"))
        assert done["state"] == "done"
        assert done["step"] == 3
        assert [row["step"] for row in done["metrics"]] == [1, 2, 3]
        assert not blob.exists()
        finished = torch.load(out, map_location="cpu", weights_only=False)
        summary = continue_run(copy, tmp_path / "resume-status.json", device_preference="cpu")
        assert summary["state"] == "done"
        assert summary["step"] == 3
        assert [row["step"] for row in summary["metrics"]] == [1, 2, 3]
        assert summary["tokensTrained"] == train.steps * train.batch_size * model.block_size
        restored = torch.load(out, map_location="cpu", weights_only=False)
        for key, value in finished["model"].items():
            assert torch.equal(value, restored["model"][key]), key
    finally:
        if control.exists():
            control.write_text(json.dumps({"command": "run"}), encoding="utf-8")
        thread.join(120)
