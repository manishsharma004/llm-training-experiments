#!/usr/bin/env python3
"""Download public-domain books and split them into documents the cleaner accepts.

The texts are Project Gutenberg editions. This script removes the Gutenberg
header and footer, then cuts the book at paragraph boundaries so each file
stays under the cleaner's length cap. It does not write new sentences.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

# Pre-1929 works in the US public domain. Sizes are approximate plain-text bytes.
BOOKS: list[tuple[int, str]] = [
    (10, "king-james-bible"),
    (2600, "war-and-peace"),
    (1184, "count-of-monte-cristo"),
    (1399, "anna-karenina"),
    (145, "middlemarch"),
    (2701, "moby-dick"),
    (1400, "great-expectations"),
    (1260, "jane-eyre"),
    (768, "wuthering-heights"),
    (1342, "pride-and-prejudice"),
    (158, "emma"),
    (76, "huckleberry-finn"),
    (345, "dracula"),
    (84, "frankenstein"),
    (1661, "sherlock-holmes"),
    (98, "tale-of-two-cities"),
    (2554, "crime-and-punishment"),
    (1228, "origin-of-species"),
    (1727, "odyssey"),
    (1497, "republic"),
    (205, "walden"),
    (36, "war-of-the-worlds"),
]

START = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
CHUNK_CHARS = 120_000


def strip_gutenberg(text: str) -> str:
    """Drop the license banner. The book between the markers is unchanged."""

    start = START.search(text)
    end = END.search(text)
    body = text
    if start:
        body = text[start.end() :]
    if end and (not start or end.start() > start.end()):
        body = body[: end.start() - (start.end() if start else 0)]
    return body.strip()


def chunk_paragraphs(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into pieces the clean stage will keep."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        pieces = [paragraph[index : index + limit] for index in range(0, len(paragraph), limit)]
        for piece in pieces:
            extra = len(piece) + (2 if current else 0)
            if current and size + extra > limit:
                chunks.append("\n\n".join(current))
                current = [piece]
                size = len(piece)
            else:
                current.append(piece)
                size += extra
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def download(book_id: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    request = urllib.request.Request(url, headers={"User-Agent": "tinyllm-lab/0.1 (public-domain corpus; one-shot)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def fetch_all(raw_dir: Path, pause: float = 0.4) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    books = []
    files = 0
    characters = 0
    for book_id, slug in BOOKS:
        text = strip_gutenberg(download(book_id))
        parts = chunk_paragraphs(text)
        for index, part in enumerate(parts, start=1):
            path = raw_dir / f"{book_id:05d}-{slug}-{index:03d}.txt"
            path.write_text(part + "\n", encoding="utf-8")
            files += 1
            characters += len(part)
        books.append({"id": book_id, "slug": slug, "chars": len(text), "files": len(parts)})
        time.sleep(pause)
    return {"books": books, "files": files, "chars": characters}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/corpus/raw"))
    args = parser.parse_args()
    report = fetch_all(args.raw)
    print(f"books {len(report['books'])} files {report['files']} chars {report['chars']}")
    for book in report["books"]:
        print(f"{book['id']:>6} {book['slug']} chars {book['chars']} files {book['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
