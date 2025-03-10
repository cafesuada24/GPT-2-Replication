"""This module contains model text generator."""

import torch
from .model.gpt2 import GPT2


def generate(
    model: GPT2,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Generates text using GPT2 model.
    This generator uses top-k sampling along with temperature scaling,
    returning diverse results while still maintaining grammatical coherence and correctness.

    Args:
        model (GPT2): a trained GPT2 model.
        max_new_tokens (int): maximum number of new tokens to generate.
        token_ids (Tensor): a Tensor contains token ids of input text.
        context_size (int): number of tokens to be included in a row.
        temperature (float): temperature to scale. Default is 0.0.
        top_k (int): top first k highest probability to select. Default is None.
        eos_id (int): end of sequence id. Default is None.
    Returns:
        Tensor: generated token ids
    """

    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        if next_token_id == eos_id:
            break
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
    return token_ids
