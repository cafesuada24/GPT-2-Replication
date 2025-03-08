"""This module contains GELU activation function."""

import torch
from torch import nn


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is commonly used in Transformer-based model like GPT and BERT.
    It applies element-wise transformation.

    Compared to ReLU, GELU retains small negative values, making it useful for models that
    benefit from smoother gradients.

    Reference:
        - Hendrycks & Gimpel, 2016: https://arxiv.org/abs/1606.08415

    Example:
        >>> gelu = GELU()
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> y = gelu(x)  # Applies GELU activation
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the GELU activation function element-wise.

        Args:
            x (Tensor): Input tensor of any shape.

        Returns:
            Tensor: Tensor of the same shape as input with GELU applied.
        """

        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.44715 * x ^ 3)
                )
            )
        )
