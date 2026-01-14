"""This module contains implementation of GPT2 model"""

from omegaconf import DictConfig
import torch
from torch import nn

from .layernorm import LayerNorm
from .transformer import Transformer


class GPT2(nn.Module):
    """"""

    def __init__(self, config: DictConfig) -> None:
        """"""

        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        self.pos_embedding = nn.Embedding(config.context_length, config.d_emb)
        self.dropout = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(
            *(Transformer(config) for _ in range(config.n_layers))
        )
        self.final_norm = LayerNorm(config.d_emb)
        self.linear_output = nn.Linear(config.d_emb, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""

        _, n_tokens = x.shape

        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(
            torch.arange(
                n_tokens,
                device=x.device,
            )
        )

        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.linear_output(x)

        return logits
