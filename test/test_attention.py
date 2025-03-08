import torch
import pytest

from src.model.attention import MultiHeadAttention, MultiHeadAttentionConfig


class TestAttention:
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],
        ]  # step
    )

    @pytest.mark.parametrize(
        "batch_size, d_in, d_out, n_heads, context_length, batch",
        [
            (2, 3, 4, 2, 6, torch.stack((inputs, inputs))),  # Small test case
            # (4, 32, 128, 128, 8, 32), # Medium case
            # (8, 64, 256, 256, 16, 64) # Larger case
        ],
    )
    def test_multihead_attention_forward(
        self, batch_size, d_in, d_out, n_heads, context_length, batch
    ):
        config = MultiHeadAttentionConfig(
            d_in=d_in, d_out=d_out, n_heads=n_heads, context_length=context_length
        )
        attn = MultiHeadAttention(config)
        output = attn(batch)

        assert output.shape == (batch_size, context_length, d_out)
