"""Byte-level BPE tokenizer trained on the book corpus.

The vocabulary is 8,192 pieces: the 256 UTF-8 bytes, plus 7,936 merges learned
from the cleaned Project Gutenberg texts. Any string still round-trips, because
a byte that never merged stays a token of its own. The merge table lives in
``tinyllm/bpe.json``. Training streams store these ids as little-endian
uint16 with a ``TKv1`` header, not as raw bytes.
"""

from __future__ import annotations

import array
import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

VOCAB_SIZE = 8192
STREAM_MAGIC = b"TKv1"
_TABLE_PATH = Path(__file__).with_name("bpe.json")


def _bytes_to_unicode() -> dict[int, str]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapping.append(256 + extra)
            extra += 1
    return {byte: chr(point) for byte, point in zip(visible, mapping)}


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {char: byte for byte, char in _BYTE_TO_UNICODE.items()}


def _load_table() -> tuple[list[str], dict[tuple[str, str], int]]:
    payload = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    pieces = list(payload["pieces"])
    if len(pieces) != VOCAB_SIZE:
        raise ValueError(f"bpe.json has {len(pieces)} pieces, expected {VOCAB_SIZE}")
    ranks: dict[tuple[str, str], int] = {}
    for rank, pair in enumerate(payload["merges"]):
        ranks[(pair[0], pair[1])] = rank
    return pieces, ranks


_PIECES, _RANKS = _load_table()
_VOCAB = {piece: index for index, piece in enumerate(_PIECES)}


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _is_space(char: str) -> bool:
    return char.isspace()


def _take_spaced(text: str, index: int, kind) -> int | None:
    """Match `` ?`` plus a run of letters, numbers, or punctuation."""

    start = index
    if text[index] == " " and index + 1 < len(text) and not _is_space(text[index + 1]):
        index += 1
    if index >= len(text) or not kind(text[index]):
        return None
    index += 1
    while index < len(text) and kind(text[index]):
        index += 1
    return index if index > start else None


def _split_words(text: str) -> list[str]:
    """GPT-2 byte-level split. One leading space stays attached to the next word."""

    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "'":
            rest = text[index + 1 :]
            suffix = ""
            if rest.startswith(("ll", "ve", "re")):
                suffix = rest[:2]
            elif rest[:1] in {"s", "d", "m", "t"}:
                suffix = rest[:1]
            if suffix:
                parts.append("'" + suffix)
                index += 1 + len(suffix)
                continue
        for kind in (_is_letter, _is_number, _is_other):
            end = _take_spaced(text, index, kind)
            if end is not None:
                parts.append(text[index:end])
                index = end
                break
        else:
            if not _is_space(text[index]):
                parts.append(text[index])
                index += 1
                continue
            end = index + 1
            while end < length and _is_space(text[end]):
                end += 1
            # ``\s+(?!\S)`` keeps a final space for the next word when the run
            # is followed by text. A trailing run is one piece.
            if end < length and end - index >= 2:
                end -= 1
            parts.append(text[index:end])
            index = end
    return parts


def _is_other(char: str) -> bool:
    return not _is_space(char) and not _is_letter(char) and not _is_number(char)


def _pairs(symbols: list[str]) -> set[tuple[str, str]]:
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


@lru_cache(maxsize=100_000)
def _bpe_word(word: str) -> tuple[int, ...]:
    symbols = list(word)
    pairs = _pairs(symbols)
    if not pairs:
        return (_VOCAB[symbols[0]],)
    while True:
        pair = min(pairs, key=lambda item: _RANKS.get(item, 10**12))
        if pair not in _RANKS:
            break
        first, second = pair
        merged: list[str] = []
        index = 0
        while index < len(symbols):
            try:
                found = symbols.index(first, index)
            except ValueError:
                merged.extend(symbols[index:])
                break
            merged.extend(symbols[index:found])
            index = found
            if index < len(symbols) - 1 and symbols[index] == first and symbols[index + 1] == second:
                merged.append(first + second)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        symbols = merged
        if len(symbols) == 1:
            break
        pairs = _pairs(symbols)
    return tuple(_VOCAB[symbol] for symbol in symbols)


def encode(text: str) -> list[int]:
    ids: list[int] = []
    for part in _split_words(text):
        mapped = "".join(_BYTE_TO_UNICODE[byte] for byte in part.encode("utf-8"))
        ids.extend(_bpe_word(mapped))
    return ids


def encode_bytes(text: str) -> list[int]:
    """The earlier checkpoints stored one UTF-8 byte per token."""

    return list(text.encode("utf-8"))


def decode_bytes(ids: list[int]) -> str:
    raw = bytes(index for index in (int(token_id) for token_id in ids) if 0 <= index < 256)
    return raw.decode("utf-8", errors="replace")


def encode_for_vocab(text: str, vocab_size: int) -> list[int]:
    """Encode with the table the checkpoint was trained on."""

    ids = encode_bytes(text) if vocab_size <= 256 else encode(text)
    overflow = next((token_id for token_id in ids if token_id >= vocab_size), None)
    if overflow is not None:
        raise ValueError(
            f"This checkpoint has a vocabulary of {vocab_size}. "
            f"The prompt needs id {overflow}, which that table does not contain."
        )
    return ids


def decode_for_vocab(ids: list[int], vocab_size: int) -> str:
    if vocab_size <= 256:
        return decode_bytes(ids)
    return decode(ids)


def decode(ids: list[int]) -> str:
    chars: list[str] = []
    for token_id in ids:
        index = int(token_id)
        if index < 0 or index >= VOCAB_SIZE:
            continue
        chars.append(_PIECES[index])
    raw = bytes(_UNICODE_TO_BYTE[char] for char in "".join(chars))
    return raw.decode("utf-8", errors="replace")


def piece_text(token_id: int) -> str:
    index = int(token_id)
    if index < 0 or index >= VOCAB_SIZE:
        return ""
    return decode([index])


def piece_glyph(token_id: int) -> str:
    text = piece_text(token_id)
    if text == "\n":
        return "↵"
    if text == " ":
        return "␣"
    glyph = text.replace("\n", "↵").replace(" ", "␣")
    return glyph or "·"


def describe(text: str) -> list[dict]:
    rows = []
    for index, token_id in enumerate(encode(text)):
        piece = piece_text(token_id)
        rows.append({"index": index, "id": token_id, "glyph": piece_glyph(token_id), "piece": piece})
    return rows


def profile() -> dict:
    return {
        "kind": "byte_level_bpe",
        "vocab_size": VOCAB_SIZE,
        "merges": len(_RANKS),
        "base_bytes": 256,
        "table": "tinyllm/bpe.json",
        "trained_on": "cleaned Project Gutenberg books in data/corpus",
    }


def pack_ids(ids: list[int]) -> bytes:
    values = array.array("H", ids)
    if sys.byteorder != "little":
        values.byteswap()
    return STREAM_MAGIC + values.tobytes()


def unpack_ids(raw: bytes) -> list[int]:
    """Read a TKv1 stream. A file without that header is the older raw-byte format."""

    if raw.startswith(STREAM_MAGIC):
        body = raw[4:]
        if len(body) % 2:
            raise ValueError("token stream is truncated")
        values = array.array("H")
        values.frombytes(body)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()
    return list(raw)
