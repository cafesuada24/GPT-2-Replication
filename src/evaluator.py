"""This module contains model evaluators"""

import torch
from torch.utils.data import DataLoader

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
        model (GPT2): a GPT2 model
        device (str or torch.device): the device that the model is sent to
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


def calc_loss_loader(
    dataloader: DataLoader,
    model: GPT2,
    device: str | torch.device,
    n_batches: int | None = None,
) -> float:
    """
    Calculate the model loss on a data loader.

    Args:
        dataloader (Dataloader): dataset
        model (GPT2): a GPT2 model
        device (str or torch.device): the device that the model is sent to
        n_batches (int or None): first n_batches to be included in calculation.
                                 Default is None, or equally the whole dataloader.
    Returns:
        Tensor: a scalar tensor contains calculated model loss.
    """

    assert n_batches is None or n_batches > 0, "n_batches must be a positive integer."

    if len(dataloader) == 0:
        return float("nan")
    if n_batches is None:
        n_batches = len(dataloader)
    else:
        n_batches = min(n_batches, len(dataloader))

    total_loss = 0.0

    for index, (input_batch, target_batch) in enumerate(dataloader):
        if index >= n_batches:
            break
        loss = calc_loss_batch(
            input_batch=input_batch,
            target_batch=target_batch,
            model=model,
            device=device,
        )
        total_loss += loss.item()

    return total_loss


def evaluate_model(
    model: GPT2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str | torch.device,
) -> tuple[float, float]:
    """
    Calculates model loss on two separated datasets - train and validation.

    Args:

    Returns:
    """

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss
