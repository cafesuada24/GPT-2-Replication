"""This module implements Feedforward layer in Transformer blocks."""

import torch
from torch import nn

from .gelu import GELU


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used in Transformer models.

    This module consists of two linear layers with an activation function in between. It is 
    applied independently to each position in the sequence. The standard formulation is:

    Args:
        d_emb (int): Dimensionality of input and output embeddings.
        activation (nn.Module, optional): Activation function (default: GELU).

    Example:
        >>> ff = FeedForward(d_emb=512)
        >>> x = torch.randn(32, 128, 512)  # (batch_size, seq_len, d_emb)
        >>> y = ff(x)  # Output shape: (32, 128, 512)
    """

    def __init__(self, d_emb: int, activation: nn.Module = GELU()) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),
            activation,
            nn.Linear(4 * d_emb, d_emb),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward transformation to the input tensor.

        The input is projected to a higher-dimensional space (`d_hidden`), passed through an 
        activation function, and projected back to the original dimension (`d_model`).

        Args:
            x (Tensor): Input tensor of shape (batch size, num tokens, d_emb).

        Returns:
            Tensor: Output tensor of the same shape (batch size, num tokens, d_emb).
        """

        return self.layers(x)
