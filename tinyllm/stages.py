"""Gather, clean, and prepare text before any weights are updated.

Each stage writes a report under ``data/work`` so the lab can show what was
kept, what was rejected, and how the bytes were packed into training windows.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

RAW_SUFFIXES = {".txt", ".md"}
MIN_CHARS = 80
MAX_CHARS = 200_000
HTML_TAGS = re.compile(r"<[^>]+>")
MULTISPACE = re.compile(r"[ \t]+")
MULTILINE = re.compile(r"\n{3,}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def gather(raw_dir: Path, work_dir: Path) -> dict:
    """Read every text file under ``raw_dir`` into ``gathered.jsonl``."""

    work_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    if raw_dir.exists():
        files = [path for path in sorted(raw_dir.rglob("*")) if path.is_file() and path.suffix.lower() in RAW_SUFFIXES]
    else:
        files = []
    for path in files:
        if path.name.startswith("."):
            continue
        text = _read_text(path)
        documents.append(
            {
                "id": path.stem,
                "source": path.as_posix(),
                "bytes": len(text.encode("utf-8")),
                "chars": len(text),
                "text": text,
            }
        )
    _write_jsonl(work_dir / "gathered.jsonl", documents)
    report = {
        "stage": "gather",
        "raw_dir": raw_dir.as_posix(),
        "documents": len(documents),
        "bytes": sum(row["bytes"] for row in documents),
        "sources": [
            {"source": row["source"], "bytes": row["bytes"], "chars": row["chars"]} for row in documents
        ],
    }
    _write_json(work_dir / "gather_report.json", report)
    return report


def clean_text(text: str) -> tuple[str | None, list[str]]:
    """Return cleaned text, or None when the document should be dropped.

    The reason list is empty when the document is kept. Callers add
    ``duplicate`` after this returns.
    """

    reasons: list[str] = []
    without_tags = HTML_TAGS.sub(" ", text)
    normalized = unicodedata.normalize("NFKC", without_tags)
    kept_chars = []
    for char in normalized:
        code = ord(char)
        if char in "\n\t" or code >= 32:
            kept_chars.append(char)
    collapsed = MULTILINE.sub("\n\n", MULTISPACE.sub(" ", "".join(kept_chars))).strip()
    if len(collapsed) < MIN_CHARS:
        reasons.append("too_short")
    if len(collapsed) > MAX_CHARS:
        reasons.append("too_long")
    letters = sum(char.isalnum() for char in collapsed)
    if collapsed and letters / len(collapsed) < 0.5:
        reasons.append("low_letter_ratio")
    if reasons:
        return None, reasons
    return collapsed, []


def clean(work_dir: Path) -> dict:
    """Clean ``gathered.jsonl`` and drop exact duplicates."""

    rows = _read_jsonl(work_dir / "gathered.jsonl")
    kept = []
    dropped = []
    seen: set[str] = set()
    reason_counts: dict[str, int] = {}
    for row in rows:
        original = row.get("text", "")
        cleaned, reasons = clean_text(original)
        if cleaned is not None:
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if digest in seen:
                reasons = ["duplicate"]
                cleaned = None
            else:
                seen.add(digest)
        if cleaned is None:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            dropped.append(
                {
                    "source": row.get("source", ""),
                    "reasons": reasons,
                    "excerpt": original[:240],
                }
            )
            continue
        kept.append(
            {
                "id": row.get("id", ""),
                "source": row.get("source", ""),
                "text": cleaned,
                "chars": len(cleaned),
            }
        )
    _write_jsonl(work_dir / "cleaned.jsonl", kept)
    report = {
        "stage": "clean",
        "input_documents": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "reasons": reason_counts,
        "rules": [
            "Strip HTML tags.",
            "Normalize to Unicode NFKC and drop control characters.",
            "Collapse repeated spaces and extra blank lines.",
            f"Drop documents shorter than {MIN_CHARS} characters or longer than {MAX_CHARS}.",
            "Drop documents whose letters and digits are under half the characters.",
            "Drop exact duplicates of an already kept document.",
        ],
        "dropped_examples": dropped[:12],
        "kept_preview": [
            {
                "source": row["source"],
                "chars": row["chars"],
                "after": row["text"][:360],
            }
            for row in kept[:6]
        ],
    }
    _write_json(work_dir / "clean_report.json", report)
    return report


def _pack(documents: list[dict]) -> bytes:
    chunks = [row["text"].strip().encode("utf-8") for row in documents if row.get("text", "").strip()]
    if not chunks:
        return b""
    return b"\n".join(chunks) + b"\n"


def prepare(work_dir: Path, block_size: int = 128, val_ratio: float = 0.15, seed: int = 7) -> dict:
    """Split cleaned documents and pack them into byte streams.

    ``seed`` is recorded for the report. The split itself is by sorted source
    name, not by a random shuffle, so the same files always produce the same
    train and validation streams.
    """

    del seed  # reserved so the lab can show that the split is deterministic
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    documents = _read_jsonl(work_dir / "cleaned.jsonl")
    if not documents:
        raise ValueError("No cleaned documents. Run the clean stage first.")
    ordered = sorted(documents, key=lambda row: row.get("source", ""))
    if len(ordered) >= 2 and val_ratio > 0:
        val_count = max(1, int(round(len(ordered) * val_ratio)))
        val_count = min(val_count, len(ordered) - 1)
    else:
        val_count = 0
    val_docs = ordered[-val_count:] if val_count else []
    train_docs = ordered[:-val_count] if val_count else ordered
    val_sources = {row.get("source", "") for row in val_docs}
    train_bytes = _pack(train_docs)
    val_bytes = _pack(val_docs)
    if len(train_bytes) < 2:
        raise ValueError("Training stream is empty after packing.")
    if len(val_bytes) < 2:
        # A single-document corpus still gets a held-out tail.
        split_at = max(2, int(len(train_bytes) * 0.9))
        val_bytes = train_bytes[split_at:]
        train_bytes = train_bytes[:split_at]
        if len(val_bytes) < 2:
            val_bytes = train_bytes[-min(len(train_bytes), 64) :]
    (work_dir / "train.bin").write_bytes(train_bytes)
    (work_dir / "val.bin").write_bytes(val_bytes)
    sft_source = _copy_sft(work_dir)
    preview = train_bytes[:220].decode("utf-8", errors="replace")
    report = {
        "stage": "prepare",
        "block_size": block_size,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens": len(train_bytes),
        "val_tokens": len(val_bytes),
        "train_windows": max(0, len(train_bytes) - block_size),
        "separator": "newline byte between documents",
        "preview": preview,
        "sft_pairs": sft_source,
        "assignment": [
            {"source": row.get("source", ""), "split": "validation" if row.get("source", "") in val_sources else "train"}
            for row in ordered
        ],
    }
    _write_json(work_dir / "prepare_report.json", report)
    return report


def _copy_sft(work_dir: Path) -> int:
    """Keep question pairs that survived a light clean, if a raw file exists."""

    candidates = [
        work_dir.parent / "raw" / "sft.jsonl",
        Path("data/raw/sft.jsonl"),
        work_dir / "sft.source.jsonl",
    ]
    rows = []
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record.get("prompt", "")).strip()
            completion = str(record.get("completion", "")).strip()
            if len(prompt) < 8 or len(completion) < 1:
                continue
            if not completion.endswith("\n"):
                completion = completion + "\n"
            if not completion.startswith(" "):
                completion = " " + completion
            rows.append({"prompt": prompt if prompt.endswith(":") or prompt.endswith("\n") else prompt, "completion": completion})
        break
    if rows:
        _write_jsonl(work_dir / "sft.jsonl", rows)
    return len(rows)


def snapshot(work_dir: Path, raw_dir: Path) -> dict:
    """Read the reports the lab renders. Missing stages are null."""

    raw_files = []
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in RAW_SUFFIXES:
                raw_files.append(
                    {
                        "source": path.as_posix(),
                        "bytes": path.stat().st_size,
                    }
                )
    return {
        "raw": raw_files,
        "gather": _read_json(work_dir / "gather_report.json"),
        "clean": _read_json(work_dir / "clean_report.json"),
        "prepare": _read_json(work_dir / "prepare_report.json"),
        "ready": (work_dir / "train.bin").exists(),
    }


def save_raw_document(raw_dir: Path, name: str, text: str) -> dict:
    """Write one user document into the raw folder."""

    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError("Name the document with letters or digits.")
    if len(text.strip()) < 1:
        raise ValueError("The document is empty.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Documents are limited to {MAX_CHARS} characters.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{slug}.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return {"source": path.as_posix(), "bytes": path.stat().st_size}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="tinyllm.stages")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--raw", default="data/raw")
    common.add_argument("--work", default="data/work")

    gather_cmd = sub.add_parser("gather", parents=[common])
    gather_cmd.set_defaults(func=lambda args: gather(Path(args.raw), Path(args.work)))
    clean_cmd = sub.add_parser("clean", parents=[common])
    clean_cmd.set_defaults(func=lambda args: clean(Path(args.work)))
    prepare_cmd = sub.add_parser("prepare", parents=[common])
    prepare_cmd.add_argument("--block-size", type=int, default=128)
    prepare_cmd.set_defaults(
        func=lambda args: prepare(Path(args.work), block_size=args.block_size)
    )
    report_cmd = sub.add_parser("report", parents=[common])
    report_cmd.set_defaults(func=lambda args: snapshot(Path(args.work), Path(args.raw)))
    ingest_cmd = sub.add_parser("ingest", parents=[common])
    ingest_cmd.add_argument("--name", required=True)
    ingest_cmd.set_defaults(
        func=lambda args: save_raw_document(Path(args.raw), args.name, sys.stdin.read())
    )

    args = parser.parse_args(argv)
    try:
        print(json.dumps(args.func(args), indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
