"""Byte tokenizer.

Each UTF-8 byte is one token. Vocabulary size is fixed at 256, so the
embedding table stays small enough for a million-parameter model and the lab
can show every id without a separate merge table.
"""

from __future__ import annotations

VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    data = text.encode("utf-8")
    return list(data)


def decode(ids: list[int]) -> str:
    raw = bytes(max(0, min(255, int(i))) for i in ids)
    return raw.decode("utf-8", errors="replace")


def describe(text: str) -> list[dict]:
    """Token records for the lab: id, byte, and a display glyph."""

    rows = []
    for index, byte in enumerate(text.encode("utf-8")):
        if byte == 10:
            glyph = "↵"
        elif byte == 32:
            glyph = "␣"
        elif 32 <= byte < 127:
            glyph = chr(byte)
        else:
            glyph = f"\\x{byte:02x}"
        rows.append({"index": index, "id": byte, "glyph": glyph})
    return rows
