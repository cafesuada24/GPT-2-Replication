import torch
from torch import nn


class CausualAttention(nn.Module):
    '''
    Causual Attention, or Masked Attention, restricts a model
    to consider only previous and current inputs in a sequence.
    '''

    def __init__(self,
                 d_in: int, d_out: int,
                 context_length: int,
                 dropout: float,
                 qkv_bias: bool = False) -> None:
        super().__init__()

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n_batches, n_tokens, d_in = inputs.shape

        queries = self.w_query(inputs)
        keys = self.w_query(inputs)
        values = self.w_query(inputs)
        attn_scores = queries @ keys.T
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens],
            -torch.inf,
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
