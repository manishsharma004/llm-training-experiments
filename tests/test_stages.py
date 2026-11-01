from pathlib import Path

from tinyllm.config import ModelConfig
from tinyllm.device import memory_budget
from tinyllm.params import breakdown
from tinyllm.stages import clean, clean_text, gather, prepare
from tinyllm.tokenizer import STREAM_MAGIC, decode, unpack_ids


def test_clean_text_strips_html_and_rejects_junk():
    cleaned, reasons = clean_text("<p>The lighthouse lamp is copper. " + ("The bell is bronze. " * 8) + "</p>")
    assert reasons == []
    assert cleaned is not None
    assert "<" not in cleaned
    assert "lighthouse" in cleaned
    empty, why = clean_text("@@@ ###")
    assert empty is None
    assert "too_short" in why or "low_letter_ratio" in why


def test_pipeline_drops_the_duplicate_and_packs_bins(tmp_path: Path):
    raw = Path("data/raw")
    work = tmp_path / "work"
    gathered = gather(raw, work)
    assert gathered["documents"] >= 5
    report = clean(work)
    assert report["reasons"].get("duplicate", 0) >= 1
    assert report["reasons"].get("too_short", 0) + report["reasons"].get("low_letter_ratio", 0) >= 1
    assert report["kept"] >= 2
    prepared = prepare(work, block_size=64)
    train = (work / "train.bin").read_bytes()
    val = (work / "val.bin").read_bytes()
    assert train.startswith(STREAM_MAGIC)
    assert len(unpack_ids(train)) == prepared["train_tokens"]
    assert prepared["val_tokens"] >= 2
    assert prepared["train_bytes"] > prepared["train_tokens"]
    assert prepared["tokenizer"]["vocab_size"] == 8192
    assert prepared["sft_pairs"] >= 10
    assert train not in val
    assert "<html>" not in decode(unpack_ids(train))


def test_book_vocabulary_preset_fits_in_four_gigabytes():
    parameters = breakdown(ModelConfig())["total"]
    assert parameters == 2_050_816
    budget = memory_budget(
        parameters=parameters,
        batch_size=16,
        block_size=128,
        n_layer=5,
        n_embd=128,
        n_head=4,
    )
    assert budget["fits"]
    assert budget["total_bytes"] < 512 * 1024**2
    assert budget["weights_bytes"] + budget["optimizer_bytes"] + budget["gradient_bytes"] == parameters * 16
