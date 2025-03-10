"""This module contains model evaluators"""

import torch

from .model.gpt2 import GPT2


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPT2,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Calculate the model loss on a data batch.

    Args:
        input_batch (Tensor): an input tensor of shape (batch size, num tokens).
        target_batch (Tensor): an target tensor of the same shape (batch size, num tokens).
    Returns:
        Tensor: a scalar tensor contains calculated model loss.
    """

    input_batch = input_batch.to(device)
    target_batch = input_batch.to(device)

    logits: torch.Tensor = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss
