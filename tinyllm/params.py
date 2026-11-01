"""Parameter accounting shared with the lab UI.

The SvelteKit page `web/src/lib/params.ts` implements this same formula.
`tests/test_params.py` checks it against `model.parameters()` so the two
copies cannot drift silently.
"""

from __future__ import annotations

from tinyllm.config import ModelConfig


def breakdown(config: ModelConfig) -> dict[str, int]:
    """Return named parameter groups and their sizes.

    Biases are omitted on the linear maps (a common small-model choice) and
    kept on LayerNorm, matching `nn.LayerNorm`. The token embedding and the
    output head share one matrix, so the head adds no extra parameters.
    """

    config.validate()
    emb = config.n_embd
    hidden = config.n_embd * config.mlp_ratio
    token_embedding = config.vocab_size * emb
    position_embedding = config.block_size * emb
    attention = 4 * emb * emb
    mlp = 2 * emb * hidden
    # Two LayerNorms per block, each with a weight and a bias.
    block_norm = 4 * emb
    per_block = attention + mlp + block_norm
    blocks = config.n_layer * per_block
    final_norm = 2 * emb
    total = token_embedding + position_embedding + blocks + final_norm
    return {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "attention": config.n_layer * attention,
        "mlp": config.n_layer * mlp,
        "layer_norm": config.n_layer * block_norm + final_norm,
        "per_block": per_block,
        "blocks": blocks,
        "final_norm": final_norm,
        "lm_head": 0,
        "total": total,
    }


def about_one_million(config: ModelConfig | None = None) -> bool:
    """True for a small lab model, including the 8,192-piece book preset.

    The original byte vocabulary was 1,035,008 parameters. The book vocabulary
    keeps the same depth and width and lands at 2,050,816.
    """

    cfg = config or ModelConfig()
    total = breakdown(cfg)["total"]
    return 800_000 <= total <= 2_500_000
