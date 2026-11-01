from pathlib import Path

import pytest

from tinyllm.checkpoint import save_checkpoint, summarize
from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.sample import complete, next_logits
from tinyllm.tokenizer import decode_for_vocab, encode, encode_for_vocab


def _save(path: Path, vocab_size: int) -> None:
    config = ModelConfig(vocab_size=vocab_size, n_layer=1, n_embd=32, n_head=4, block_size=32, dropout=0.0)
    model = TinyLM(config)
    save_checkpoint(path, model, config, TrainConfig(steps=1), step=1, loss=1.0, phase="pretrain")


def test_byte_checkpoint_encodes_the_prompt_as_bytes(tmp_path: Path):
    path = tmp_path / "byte.pt"
    _save(path, 256)
    prompt = "The lighthouse lamp is"
    assert any(token_id >= 256 for token_id in encode(prompt))
    result = complete(path, prompt, max_new_tokens=4, temperature=0.0)
    assert result["vocabSize"] == 256
    assert result["text"].startswith(prompt)


def test_bpe_checkpoint_round_trips_the_prompt(tmp_path: Path):
    path = tmp_path / "bpe.pt"
    _save(path, 8192)
    prompt = "The lighthouse lamp is"
    result = complete(path, prompt, max_new_tokens=2, temperature=0.0)
    assert result["vocabSize"] == 8192
    assert result["text"].startswith(prompt)


def test_summary_reads_the_saved_vocabulary(tmp_path: Path):
    path = tmp_path / "byte.pt"
    _save(path, 256)
    row = summarize(path)
    assert row["vocabSize"] == 256
    assert row["phase"] == "pretrain"
    assert row["step"] == 1
    assert row["blockSize"] == 32
    assert row["batchSize"] == 16


def test_next_logits_match_the_greedy_token(tmp_path: Path):
    path = tmp_path / "bpe.pt"
    _save(path, 8192)
    prompt = "The lighthouse lamp is"
    dist = next_logits(path, prompt)
    assert dist["vocabSize"] == 8192
    assert len(dist["logits"]) == 8192
    assert dist["tokenIds"] == encode_for_vocab(prompt, 8192)
    best = max(range(len(dist["logits"])), key=lambda index: dist["logits"][index])
    completed = complete(path, prompt, max_new_tokens=1, temperature=0.0)
    assert completed["text"] == decode_for_vocab(dist["tokenIds"] + [best], 8192)


def test_byte_logits_stay_inside_the_byte_table(tmp_path: Path):
    path = tmp_path / "byte.pt"
    _save(path, 256)
    dist = next_logits(path, "The lighthouse lamp is")
    assert dist["vocabSize"] == 256
    assert len(dist["logits"]) == 256
    assert max(dist["tokenIds"]) < 256


def test_small_vocab_reports_an_id_it_cannot_embed(tmp_path: Path):
    path = tmp_path / "small.pt"
    _save(path, 64)
    with pytest.raises(ValueError, match="vocabulary of 64"):
        complete(path, "The lighthouse lamp is", max_new_tokens=1)
