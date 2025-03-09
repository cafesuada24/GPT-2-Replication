"""This module implements Normalization Layer"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """"""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""

        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)

        return self.scale * x_norm + self.shift
