"""
attention.py - Multi-Head Attention Module

This module implements the Multi-Head Attention mechanism as described in
"Attention Is All You Need" (Vaswani et al., 2017). It includes:

- `MultiHeadAttentionConfig`: A dataclass for configuring multi-head attention parameters.
- `MultiHeadAttention`: A PyTorch module that computes self-attention over input sequences.

Multi-Head Attention allows a model to jointly attend to different positions in a sequence
by using multiple attention heads, improving its ability to capture complex dependencies.

Classes:
    - MultiHeadAttentionConfig: Stores attention hyperparameters.
    - MultiHeadAttention: Implements the multi-head attention mechanism.

Example Usage:
    >>> config = MultiHeadAttentionConfig(
        d_in=512, d_out=512, n_heads=8,
        context_length=128, dropout=0.1)
    >>> attn = MultiHeadAttention(config)
    >>> output = attn(inputs)

References:
    - Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultiHeadAttentionConfig:
    """
    Configuration for the Multi-Head Attention mechanism.

    This class encapsulates the hyperparameters required to initialize a multi-head
    attention module, improving code readability and maintainability.

    Attributes:
        d_in (int): Input embedding dimension.
        d_out (int): Output embedding dimension.
        n_heads (int): Number of attention heads.
        context_length (int): Maximum sequence length the model can handle.
        d_head (int, optional): Attention head dimension, must be equal to n_out // n_heads.
                                Default is -1 (will be inferred).
        dropout (float, optional): Dropout rate applied to attention weights. Default is 0.0.
        qkv_bias (bool, optional): Whether to include bias in the query, key, and value projections.
                                   Default is False.

    Example:
        config = MultiHeadAttentionConfig(
            d_in=512, d_out=512, n_heads=8, context_length=128, dropout=0.1, qkv_bias=True
        )
    """

    d_in: int
    d_out: int
    n_heads: int
    context_length: int
    d_head: int = -1
    dropout: float = 0.0
    qkv_bias: bool = False

    def __post__init__(self) -> None:
        if self.d_in <= 0:
            raise ValueError("d_in must be a positive integer.")
        if self.d_out <= 0:
            raise ValueError("d_out must be a positive integer.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be a positive integer.")
        if self.d_head < -1 or self.d_head == 0:
            raise ValueError("d_head must be a positive integer (except -1).")
        if self.d_head == -1:
            if self.d_out % self.n_heads > 0:
                raise ValueError("d_out must be divisible by n_heads.")
            self.d_head = self.d_out // self.n_heads
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be a positive integer.")


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention as described in "Attention Is All You Need"
    (Vaswani et al., 2017).

    This module allows the model to jointly attend to information from different
    representation subspaces at different positions. It projects the input queries,
    keys, and values into multiple heads, applies scaled dot-product attention,
    and concatenates the results.

    Args:
        config (MultiHeadAttentionConfig): configuration of the model

    Example:
        attn = MultiHeadAttention(config)
    """

    def __init__(self, config: MultiHeadAttentionConfig) -> None:
        super().__init__()

        self.config = config

        self.w_query = nn.Linear(
            self.config.d_in, self.config.d_out, bias=self.config.qkv_bias
        )
        self.w_key = nn.Linear(
            self.config.d_in, self.config.d_out, bias=self.config.qkv_bias
        )
        self.w_value = nn.Linear(
            self.config.d_in, self.config.d_out, bias=self.config.qkv_bias
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.out_proj = nn.Linear(self.config.d_out, self.config.d_out)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(self.config.context_length, self.config.context_length),
                diagonal=1,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-head attention output.

        This method applies scaled dot-product attention across multiple heads,
        allowing the model to focus on different parts of the input sequence.

        Inputs:
            inputs: Shape (batch_size, seq_len, d_in)

        Outputs:
            Tensor: Shape (batch_size, seq_len, d_out), attended values.

        Example:
            output = attn(inputs)
        """

        n_batches, n_tokens, _ = inputs.shape

        queries = self.w_query(inputs)
        keys = self.w_key(inputs)
        values = self.w_value(inputs)

        queries, keys, values = (
            tensor.view(
                n_batches, n_tokens, self.config.n_heads, self.config.d_head
            ).transpose(1, 2)
            for tensor in (queries, keys, values)
        )
        # Shape (num batches, num heads, num tokens, dim heads)

        # queries = queries.view(
        #     n_batches,
        #     n_tokens,
        #     self.config.n_heads,
        #     self.config.d_head).transpose(1, 2)
        # keys = keys.view(
        #     n_batches,
        #     n_tokens,
        #     self.config.n_heads,
        #     self.config.d_head).transpose(1, 2)
        # values = values.view(
        #     n_batches,
        #     n_tokens,
        #     self.config.n_heads,
        #     self.config.d_head).transpose(1, 2)

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # Shape (_, _, num tokens, num tokens)
        attn_scores.masked_fill_(
            getattr(self, "mask").bool()[:n_tokens, :n_tokens],
            -torch.inf,
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # Shape (_, num tokens, num heads, dim heads)
        context_vec = context_vec.contiguous().view(
            n_batches, n_tokens, self.config.d_out
        )
        context_vec = self.out_proj(context_vec)

        return context_vec
