"""Train the 8,192-piece byte-level BPE table on the cleaned book corpus.

Run with the repo virtualenv after ``pip install tokenizers``. Do not use
``uv sync`` or ``uv run`` for this: those commands replace the AMD torch wheel.

    .venv/bin/python scripts/train_bpe.py

The script writes ``tinyllm/bpe.json``, which the lab and the trainer load.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "corpus" / "work" / "cleaned.jsonl"
OUT = ROOT / "tinyllm" / "bpe.json"
VOCAB_SIZE = 8192


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit(f"Missing {CLEANED}. Gather and clean the book corpus first.")
    rows = []
    for line in CLEANED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["text"])
    tokenizer = Tokenizer(BPE(unk_token=None, fuse_unk=False))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(rows, trainer=trainer)
    saved = json.loads(tokenizer.to_str())
    model = saved["model"]
    pieces = [None] * len(model["vocab"])
    for token, index in model["vocab"].items():
        pieces[index] = token
    if any(piece is None for piece in pieces) or len(pieces) != VOCAB_SIZE:
        raise SystemExit(f"Expected {VOCAB_SIZE} pieces, got {len(pieces)}")
    payload = {
        "version": 1,
        "kind": "byte_level_bpe",
        "vocab_size": VOCAB_SIZE,
        "add_prefix_space": False,
        "trained_on": "data/corpus/work/cleaned.jsonl",
        "documents": len(rows),
        "pieces": pieces,
        "merges": model["merges"],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "vocab_size": VOCAB_SIZE, "merges": len(model["merges"]), "documents": len(rows)}))


if __name__ == "__main__":
    main()
