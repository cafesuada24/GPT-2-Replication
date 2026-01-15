"""This module contains implementation of a Transformer block."""

import torch
from omegaconf import DictConfig
from torch import nn

from .attention import MultiHeadAttention, MultiHeadAttentionConfig
from .feedforward import FeedForward
from .layernorm import LayerNorm


class Transformer(nn.Module):
    """"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        mha_config = MultiHeadAttentionConfig(
            d_in=config.d_emb,
            d_out=config.d_emb,
            n_heads=config.n_heads,
            context_length=config.context_length,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
        )
        self.layernorm1 = LayerNorm(config.d_emb)
        self.layernorm2 = LayerNorm(config.d_emb)
        self.mha = MultiHeadAttention(mha_config)
        self.feedforward = FeedForward(config.d_emb)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""

        shortcut = x
        x = self.layernorm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        return x + shortcut

