"""Project the real embedding and one attention pass into something a screen can show.

The token table is one vector per vocabulary piece. The lab draws the first
three principal components, and keeps the nearest neighbors from the full
width so a click is not only a guess about the flattened picture. Attention
weights come from the same forward the trainer uses.
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
from tinyllm.tokenizer import encode_for_vocab, piece_glyph


def glyph_for(token_id: int) -> str:
    return piece_glyph(token_id)


def load_for_inspect(checkpoint: Path | None) -> tuple[TinyLM, str, torch.Tensor | None, int | None]:
    if checkpoint is not None:
        model, blob = load_model(checkpoint, torch.device("cpu"))
        seed = int((blob.get("train_config") or {}).get("seed", 0))
        start = initial_token_rows(model.config, seed)
        return model, checkpoint.as_posix(), start, seed
    model = TinyLM(ModelConfig())
    model.eval()
    return model, "untrained initialization", None, None


def initial_token_rows(config: ModelConfig, seed: int) -> torch.Tensor:
    """The token table `TinyLM` builds immediately after `torch.manual_seed(seed)`.

    Training calls that seed and then constructs the module, so these rows are
    the start state of a run that saved the same seed.
    """

    torch.manual_seed(seed)
    fresh = TinyLM(config)
    return fresh.token_embedding.weight.detach().float().cpu().clone()


def _neighbors(weight: torch.Tensor, count: int = 4) -> tuple[list[list[int]], list[list[float]]]:
    normal = F.normalize(weight, dim=1)
    scores = normal @ normal.T
    scores.fill_diagonal_(-2.0)
    top = torch.topk(scores, k=min(count, weight.shape[0] - 1), dim=1).indices
    cosines: list[list[float]] = []
    indexes: list[list[int]] = []
    for row, chosen in enumerate(top.tolist()):
        indexes.append(chosen)
        cosines.append([round(float(torch.dot(normal[row], normal[other])), 4) for other in chosen])
    return indexes, cosines


def _shared_axes(start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project both tables with one basis, so a line from start to end is meaningful."""

    start_centered = start - start.mean(dim=0, keepdim=True)
    end_centered = end - end.mean(dim=0, keepdim=True)
    both = torch.cat([start_centered, end_centered], dim=0)
    _u, _s, basis = torch.pca_lowrank(both, q=min(3, both.shape[1]))
    start_xyz = start_centered @ basis[:, :3]
    end_xyz = end_centered @ basis[:, :3]
    span = torch.cat([start_xyz, end_xyz], dim=0).abs().amax(dim=0).clamp_min(1e-6)
    return start_xyz / span, end_xyz / span


def build_geometry(
    model: TinyLM,
    source: str,
    prompt: str = "The lighthouse lamp",
    start: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict:
    end = model.token_embedding.weight.detach().float().cpu()
    if start is None:
        start = end
        start_kind = "same"
    else:
        start = start.detach().float().cpu()
        start_kind = "seed"
    start_xyz, end_xyz = _shared_axes(start, end)
    travel = torch.linalg.vector_norm(end - start, dim=1)
    cosine_to_start = (F.normalize(end, dim=1) * F.normalize(start, dim=1)).sum(dim=1)
    neighbors, neighbor_cos = _neighbors(end)
    points = []
    for index in range(end.shape[0]):
        points.append(
            {
                "id": index,
                "glyph": glyph_for(index),
                "x": round(float(end_xyz[index, 0]), 4),
                "y": round(float(end_xyz[index, 1]), 4),
                "z": round(float(end_xyz[index, 2]), 4),
                "sx": round(float(start_xyz[index, 0]), 4),
                "sy": round(float(start_xyz[index, 1]), 4),
                "sz": round(float(start_xyz[index, 2]), 4),
                "travel": round(float(travel[index]), 4),
                "cosineToStart": round(float(cosine_to_start[index]), 4),
                "neighbors": neighbors[index],
                "neighborCos": neighbor_cos[index],
                "preview": [round(float(value), 3) for value in end[index, :16]],
                "startHead": [round(float(value), 3) for value in start[index, :16]],
            }
        )

    ids = encode_for_vocab(prompt, model.config.vocab_size)[: model.config.block_size]
    if len(ids) < 2:
        ids = encode_for_vocab("The lamp", model.config.vocab_size)
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
        "startKind": start_kind,
        "seed": seed,
        "meanTravel": round(float(travel.mean()), 4),
        "meanCosineToStart": round(float(cosine_to_start.mean()), 4),
        "points": points,
        "attention": {
            "text": prompt,
            "ids": ids,
            "glyphs": [glyph_for(token_id) for token_id in ids],
            "layers": layers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a 3D projection of the token embedding and one attention pass.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt", default="The lighthouse lamp")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    model, source, start, seed = load_for_inspect(checkpoint)
    print(json.dumps(build_geometry(model, source, args.prompt, start=start, seed=seed)))


if __name__ == "__main__":
    main()
