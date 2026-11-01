"""Project the real embedding and one attention pass into something a screen can show.

The token table is 256 vectors of width 128. The lab draws the first three
principal components, and keeps the nearest neighbors from the full width so
a click is not only a guess about the flattened picture. Attention weights
come from the same forward the trainer uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tinyllm.checkpoint import load_model
from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.tokenizer import encode


def glyph_for(byte: int) -> str:
    if byte == 10:
        return "↵"
    if byte == 32:
        return "␣"
    if 32 <= byte < 127:
        return chr(byte)
    return f"\\x{byte:02x}"


def load_for_inspect(checkpoint: Path | None) -> tuple[TinyLM, str]:
    if checkpoint is not None:
        model, _blob = load_model(checkpoint, torch.device("cpu"))
        return model, checkpoint.as_posix()
    model = TinyLM(ModelConfig())
    model.eval()
    return model, "untrained initialization"


def _neighbors(weight: torch.Tensor, count: int = 4) -> list[list[int]]:
    normal = F.normalize(weight, dim=1)
    scores = normal @ normal.T
    scores.fill_diagonal_(-2.0)
    top = torch.topk(scores, k=min(count, weight.shape[0] - 1), dim=1).indices
    return top.tolist()


def build_geometry(model: TinyLM, source: str, prompt: str = "The lighthouse lamp") -> dict:
    weight = model.token_embedding.weight.detach().float().cpu()
    centered = weight - weight.mean(dim=0, keepdim=True)
    _u, _s, basis = torch.pca_lowrank(centered, q=3)
    coords = centered @ basis[:, :3]
    # Scale each axis on its own so a long thin cloud still fills the view.
    # Neighbors below stay in the original width, not in this picture.
    for axis in range(3):
        span = coords[:, axis].abs().amax().clamp_min(1e-6)
        coords[:, axis] = coords[:, axis] / span
    neighbors = _neighbors(weight)
    points = []
    for index in range(weight.shape[0]):
        points.append(
            {
                "id": index,
                "glyph": glyph_for(index),
                "x": round(float(coords[index, 0]), 4),
                "y": round(float(coords[index, 1]), 4),
                "z": round(float(coords[index, 2]), 4),
                "neighbors": neighbors[index],
                "preview": [round(float(value), 3) for value in weight[index, :8]],
            }
        )

    ids = encode(prompt)[: model.config.block_size]
    if len(ids) < 2:
        ids = encode("The lamp")
    tokens = torch.tensor([ids], dtype=torch.long)
    positions = torch.arange(len(ids))
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        for block in model.blocks:
            hidden = block(hidden, record=captured)
    layers = []
    for weights in captured:
        heads = weights[0].float().cpu()
        layers.append([[[round(float(value), 4) for value in row] for row in head] for head in heads])

    return {
        "source": source,
        "width": model.config.n_embd,
        "layers": model.config.n_layer,
        "heads": model.config.n_head,
        "points": points,
        "attention": {
            "text": prompt,
            "ids": ids,
            "glyphs": [glyph_for(byte) for byte in ids],
            "layers": layers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a 3D projection of the token embedding and one attention pass.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt", default="The lighthouse lamp")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    model, source = load_for_inspect(checkpoint)
    print(json.dumps(build_geometry(model, source, args.prompt)))


if __name__ == "__main__":
    main()
