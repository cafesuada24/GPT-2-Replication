"""This module contains model trainers."""

from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dataset.dataset import DataSample
from src.tokenizer import Tokenizer, text_to_token_ids, token_ids_to_text

from .evaluator import calc_loss_batch, evaluate_model
from .model.gpt2 import GPT2


def generate_text_simple(
    model: GPT2,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(
    model: GPT2,
    tokenizer: Tokenizer,
    device: str | torch.device,
    start_context: str,
):
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()


def train_model(
    model: GPT2,
    train_loader: DataLoader[DataSample],
    val_loader: DataLoader[DataSample],
    optimizer: Optimizer,
    n_epochs: int,
    eval_freq: int,
    eval_iter: int,
    logger: Logger,
    tokenizer: Tokenizer,
    start_context: str,
    device: str | torch.device = 'cpu',
):
    """"""

    train_losses: list[float] = []
    val_losses: list[float] = []
    global_step = -1

    for epoch in range(n_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq != 0:
                continue

            train_loss, val_loss = evaluate_model(
                model,
                train_loader,
                val_loader,
                device,
                eval_iter=eval_iter,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logger.info(
                f'Epoch {epoch + 1} (Step {global_step:06d}): '
                f'Train loss {train_loss:.3f}, '
                f'Val loss {val_loss:.3f}'
            )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses
