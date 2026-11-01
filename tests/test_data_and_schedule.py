import torch

from tinyllm.corpus import finetune_examples, prose_corpus
from tinyllm.data import finetune_batch
from tinyllm.schedule import learning_rate
from tinyllm.tokenizer import decode, describe, encode


def test_questions_are_held_out_of_the_prose():
    prose = prose_corpus(1)
    assert "Question:" not in prose
    assert "The lighthouse lamp is copper." in prose
    prompts = [row["prompt"] for row in finetune_examples()]
    assert any(prompt.startswith("Question: What is the lighthouse lamp?") for prompt in prompts)
    assert any("harbor watch" in prompt for prompt in prompts)


def test_bpe_roundtrip_and_description():
    text = "The lamp is copper.\n"
    assert encode(text) == [627, 5507, 351, 1001, 7670, 13, 198]
    assert decode(encode(text)) == text
    rows = describe(text)
    assert rows[0]["glyph"] == "The"
    assert rows[-1]["glyph"] == "↵"
    assert all(0 <= row["id"] < 8192 for row in rows)
    assert "".join(row["piece"] for row in rows) == text


def test_finetune_labels_ignore_the_prompt():
    examples = [
        {"prompt": "Question: color?\nAnswer:", "completion": " copper\n"},
    ]
    inputs, labels, key_padding = finetune_batch(examples, block_size=64, batch_size=2, device=torch.device("cpu"))
    prompt_len = len(encode(examples[0]["prompt"]))
    assert prompt_len > 1
    # The first supervised label predicts the first completion byte.
    supervised = (labels[0] != -100).nonzero().flatten()
    assert supervised.numel() > 0
    assert int(supervised[0]) == prompt_len - 1
    assert bool(key_padding[0, 0]) is False
    assert inputs.shape[0] == 2


def test_learning_rate_warms_up_then_decays():
    peak = 1.0
    floor = 0.1
    assert learning_rate(1, 100, peak, floor, 10) == 0.1
    assert abs(learning_rate(10, 100, peak, floor, 10) - 1.0) < 1e-9
    mid = learning_rate(55, 100, peak, floor, 10)
    end = learning_rate(100, 100, peak, floor, 10)
    assert peak > mid > end
    assert abs(end - floor) < 1e-9
