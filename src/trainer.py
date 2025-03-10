"""This module contains model trainers"""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .evaluator import calc_loss_batch
from .model.gpt2 import GPT2


def train_model(
    model: GPT2,
    dataloader: DataLoader,
    optimizer: Optimizer,
    n_epochs: int,
    device: str | torch.device = 'cpu'
):
    """"""

    for _ in range(n_epochs):
        model.train()

        for input_batch, target_batch in dataloader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
