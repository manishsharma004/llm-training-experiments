"""Decoder-only Transformer, written to be read next to a textbook diagram.

One block is pre-norm attention, a residual add, pre-norm MLP, and another
residual add. The input embedding and the final logits share weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyllm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # One packed projection for query, key, and value, then an output map.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding: torch.Tensor | None = None,
        record: list | None = None,
    ) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(self.n_embd, dim=2)
        query = query.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal = self.mask[:, :, :steps, :steps] == 0
        scores = scores.masked_fill(causal, float("-inf"))
        if key_padding is not None:
            # key_padding: True where the key position is padding.
            scores = scores.masked_fill(key_padding[:, None, None, :], float("-inf"))
            # A fully masked row (a pad query) would make softmax emit NaN.
            fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(fully_masked, 0.0)

        weights = F.softmax(scores, dim=-1)
        if record is not None:
            record.append(weights.detach())
        weights = self.attn_drop(weights)
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_drop(self.proj(mixed))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.n_embd * config.mlp_ratio
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        key_padding: torch.Tensor | None = None,
        record: list | None = None,
    ) -> torch.Tensor:
        attn_record: list | None = [] if record is not None else None
        x = x + self.attn(self.norm1(x), key_padding=key_padding, record=attn_record)
        if record is not None and attn_record:
            record.append(attn_record[0])
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the head reuses the token table. The parameter count
        # in `params.breakdown` depends on this.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        key_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _batch, steps = token_ids.shape
        if steps > self.config.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.config.block_size}")
        positions = torch.arange(steps, device=token_ids.device)
        hidden = self.drop(self.token_embedding(token_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden, key_padding=key_padding)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            window = token_ids[:, -self.config.block_size :]
            logits, _ = self(window)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
                    cutoff = values[:, -1].unsqueeze(-1)
                    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids
