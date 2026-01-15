"""This module contains implementation of GPT2 model."""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from .layernorm import LayerNorm
from .transformer import Transformer


def _assign(left: torch.Tensor, right: np.ndarray) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(
            f'Shape mismatch. Left: {left.shape}, ',
            f'Right: {right.shape}',
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: 'GPT2', params: dict[str, Any]) -> None:
    gpt.pos_embedding.weight = _assign(gpt.pos_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = _assign(gpt.token_embedding.weight, params['wte'])

    for b in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split(
            (params['blocks'][b]['attn']['c_attn'])['w'],
            3,
            axis=-1,
        )
        gpt.trf_blocks[b].mha.w_query.weight = _assign(
            gpt.trf_blocks[b].mha.w_query.weight,
            q_w.T,
        )
        gpt.trf_blocks[b].mha.w_key.weight = _assign(
            gpt.trf_blocks[b].mha.w_key.weight,
            k_w.T,
        )
        gpt.trf_blocks[b].mha.w_value.weight = _assign(
            gpt.trf_blocks[b].mha.w_value.weight,
            v_w.T,
        )
        q_b, k_b, v_b = np.split(
            (params['blocks'][b]['attn']['c_attn'])['b'],
            3,
            axis=-1,
        )
        gpt.trf_blocks[b].mha.w_query.bias = _assign(
            gpt.trf_blocks[b].mha.w_query.bias,
            q_b,
        )
        gpt.trf_blocks[b].mha.w_key.bias = _assign(
            gpt.trf_blocks[b].mha.w_key.bias,
            k_b,
        )
        gpt.trf_blocks[b].mha.w_value.bias = _assign(
            gpt.trf_blocks[b].mha.w_value.bias,
            v_b,
        )
        gpt.trf_blocks[b].mha.out_proj.weight = _assign(
            gpt.trf_blocks[b].mha.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T,
        )
        gpt.trf_blocks[b].mha.out_proj.bias = _assign(
            gpt.trf_blocks[b].mha.out_proj.bias,
            params['blocks'][b]['attn']['c_proj']['b'],
        )

        gpt.trf_blocks[b].feedforward.layers[0].weight = _assign(
            gpt.trf_blocks[b].feedforward.layers[0].weight,
            params['blocks'][b]['mlp']['c_fc']['w'].T,
        )
        gpt.trf_blocks[b].feedforward.layers[0].bias = _assign(
            gpt.trf_blocks[b].feedforward.layers[0].bias,
            params['blocks'][b]['mlp']['c_fc']['b'],
        )
        gpt.trf_blocks[b].feedforward.layers[2].weight = _assign(
            gpt.trf_blocks[b].feedforward.layers[2].weight,
            params['blocks'][b]['mlp']['c_proj']['w'].T,
        )
        gpt.trf_blocks[b].feedforward.layers[2].bias = _assign(
            gpt.trf_blocks[b].feedforward.layers[2].bias,
            params['blocks'][b]['mlp']['c_proj']['b'],
        )

        gpt.trf_blocks[b].layernorm1.scale = _assign(
            gpt.trf_blocks[b].layernorm1.scale,
            params['blocks'][b]['ln_1']['g'],
        )
        gpt.trf_blocks[b].layernorm1.shift = _assign(
            gpt.trf_blocks[b].layernorm1.shift,
            params['blocks'][b]['ln_1']['b'],
        )
        gpt.trf_blocks[b].layernorm2.scale = _assign(
            gpt.trf_blocks[b].layernorm2.scale,
            params['blocks'][b]['ln_2']['g'],
        )
        gpt.trf_blocks[b].layernorm2.shift = _assign(
            gpt.trf_blocks[b].layernorm2.shift,
            params['blocks'][b]['ln_2']['b'],
        )

        gpt.final_norm.scale = _assign(gpt.final_norm.scale, params['g'])
        gpt.final_norm.shift = _assign(gpt.final_norm.shift, params['b'])
        gpt.linear_output.weight = _assign(gpt.linear_output.weight, params['wte'])


class GPT2(nn.Module):
    """"""

    def __init__(self, config: DictConfig) -> None:
        """"""

        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        self.pos_embedding = nn.Embedding(config.context_length, config.d_emb)
        self.dropout = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(
            *(Transformer(config) for _ in range(config.n_layers)),
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
            ),
        )

        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.linear_output(x)
