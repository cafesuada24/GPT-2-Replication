import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.transformer import Transformer


@pytest.fixture
def config() -> DictConfig:
    return OmegaConf.create({
        'vocab_size': 50257,
        'context_length': 1024,
        'drop_rate': 0.1,
        'qkv_bias': False,
        'd_emb': 768,
        'n_layers': 12,
        'n_heads': 12,
    })

@pytest.fixture
def n_batches() -> int:
    return 2

@pytest.fixture
def n_tokens() -> int:
    return 5

def test_transformer_forward(config: DictConfig, n_batches: int, n_tokens: int):
    trf = Transformer(config)

    x = torch.randn(n_batches, n_tokens, config.d_emb)

    output = trf(x)

    assert output.shape == (n_batches, n_tokens, config.d_emb)

# def test_transformer_backprop(config, n_batches, n_tokens):
#     """Ensure TransformerBlock supports backpropagation."""
#
#     trf = Transformer(config)
#
#     x = torch.randn(n_batches, n_tokens, config.d_emb)
#
#     output = trf(x)
#     loss = output.sum()
#     loss.backward()
#
#     assert x.grad is not None, "Gradients are not flowing properly."
