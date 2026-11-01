"""The 3D lab view is a projection of the real embedding, not a drawing of one."""

import torch

from tinyllm.config import ModelConfig
from tinyllm.inspect import build_geometry, initial_token_rows
from tinyllm.model import TinyLM


def test_geometry_keeps_full_width_neighbors_and_causal_attention():
    model = TinyLM(ModelConfig(n_layer=1, n_embd=32, n_head=4, block_size=32))
    model.eval()
    payload = build_geometry(model, "test init", prompt="Hi")
    assert payload["width"] == 32
    assert len(payload["points"]) == model.config.vocab_size
    assert len(payload["points"][0]["preview"]) == 16
    assert len(payload["points"][0]["startHead"]) == 16
    assert len(payload["points"][0]["neighbors"]) == 4
    assert 0 not in payload["points"][0]["neighbors"]

    layers = payload["attention"]["layers"]
    assert len(layers) == 1
    head = layers[0][0]
    assert len(head) == len(payload["attention"]["ids"])
    for row, weights in enumerate(head):
        assert abs(sum(weights) - 1) < 1e-3
        if row + 1 < len(weights):
            assert max(weights[row + 1 :]) == 0


def test_seed_rebuilds_the_token_rows_and_the_projection_keeps_both():
    config = ModelConfig(vocab_size=128, n_layer=1, n_embd=32, n_head=4, block_size=16)
    torch.manual_seed(11)
    model = TinyLM(config)
    start = model.token_embedding.weight.detach().clone()
    assert torch.allclose(initial_token_rows(config, 11), start)
    with torch.no_grad():
        model.token_embedding.weight[:10].add_(0.5)
    payload = build_geometry(model, "moved", prompt="Hi", start=start, seed=11)
    assert payload["startKind"] == "seed"
    assert payload["meanTravel"] > 0
    row = payload["points"][0]
    assert row["travel"] > 0
    assert (row["x"], row["y"], row["z"]) != (row["sx"], row["sy"], row["sz"])
    assert len(row["neighborCos"]) == 4
    assert all(-1.0 <= value <= 1.0 for value in row["neighborCos"])
    untouched = build_geometry(model, "same", prompt="Hi")
    assert untouched["startKind"] == "same"
    assert untouched["meanTravel"] == 0.0


def test_recorded_attention_matches_a_manual_softmax():
    model = TinyLM(ModelConfig(n_layer=1, n_embd=32, n_head=4, block_size=16))
    model.eval()
    tokens = torch.tensor([[10, 20, 30, 40]])
    record: list = []
    with torch.no_grad():
        model.blocks[0](model.token_embedding(tokens), record=record)
    weights = record[0][0, 0]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert float(weights[0, 1]) == 0.0
    assert float(weights[0, 0]) == 1.0
