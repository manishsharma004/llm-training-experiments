import torch

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.params import about_one_million, breakdown


def test_default_preset_is_about_one_million_and_matches_the_module():
    config = ModelConfig()
    report = breakdown(config)
    model = TinyLM(config)
    counted = sum(parameter.numel() for parameter in model.parameters())
    assert report["total"] == counted
    assert report["lm_head"] == 0
    assert about_one_million(config)
    # The lab copy quotes this exact default so the UI and the model agree.
    assert counted == 1_035_008


def test_weight_tying_shares_the_embedding_storage():
    model = TinyLM(ModelConfig())
    assert model.lm_head.weight is model.token_embedding.weight
    names = [name for name, _parameter in model.named_parameters()]
    assert names.count("token_embedding.weight") == 1
    assert "lm_head.weight" not in names


def test_rejects_heads_that_do_not_divide_width():
    try:
        ModelConfig(n_embd=128, n_head=3).validate()
    except ValueError:
        return
    raise AssertionError("expected validate() to reject an uneven head split")


def test_forward_shapes_and_a_single_backward():
    config = ModelConfig(n_layer=1, n_embd=32, n_head=4, block_size=16, vocab_size=64)
    model = TinyLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(tokens, labels=labels)
    assert logits.shape == (2, 8, config.vocab_size)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
