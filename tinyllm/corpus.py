"""A tiny invented harbor, repeated often enough to memorize.

The sentences are original and internally consistent so a short CPU run can
drive the loss down. The finetune set teaches a question format the prose
never uses. Facts live in `harbor.json`, which the lab UI reads as well.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_harbor() -> dict:
    path = Path(__file__).with_name("harbor.json")
    return json.loads(path.read_text(encoding="utf-8"))


def prose_sentence(subject: str, value: str) -> str:
    return f"The {subject} is {value}."


def prose_corpus(repeats: int = 80) -> str:
    facts = load_harbor()["facts"]
    sentences = [prose_sentence(row["subject"], row["value"]) for row in facts]
    paragraph = " ".join(sentences)
    return ("\n".join([paragraph] * repeats)).strip() + "\n"


def finetune_examples() -> list[dict[str, str]]:
    """Supervised pairs. The question format is absent from the prose corpus."""

    rows = []
    harbor = load_harbor()
    for row in harbor["facts"]:
        prompt = f"Question: What is the {row['subject']}?\nAnswer:"
        completion = f" {row['value']}\n"
        rows.append({"prompt": prompt, "completion": completion})
    for row in harbor["paraphrases"]:
        rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    return rows


PROBES = [
    "The lighthouse lamp is",
    "The captain is",
    "Question: What is the lighthouse lamp?\nAnswer:",
    "Question: Who keeps the harbor watch?\nAnswer:",
]
