"""The 3D lab view is a projection of the real embedding, not a drawing of one."""

import torch

from tinyllm.config import ModelConfig
from tinyllm.inspect import build_geometry
from tinyllm.model import TinyLM


def test_geometry_keeps_full_width_neighbors_and_causal_attention():
    model = TinyLM(ModelConfig(n_layer=1, n_embd=32, n_head=4, block_size=32))
    model.eval()
    payload = build_geometry(model, "test init", prompt="Hi")
    assert payload["width"] == 32
    assert len(payload["points"]) == model.config.vocab_size
    assert len(payload["points"][0]["preview"]) == 8
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
