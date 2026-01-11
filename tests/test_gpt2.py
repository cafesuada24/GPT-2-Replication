import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.gpt2 import GPT2


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

def test_gpt2_shape(config: DictConfig, n_batches: int, n_tokens: int):
    gpt2 = GPT2(config)

    x = torch.randint(0, config.vocab_size, (n_batches, n_tokens))

    output = gpt2(x)

    assert output.shape == (n_batches, n_tokens, config.vocab_size)
