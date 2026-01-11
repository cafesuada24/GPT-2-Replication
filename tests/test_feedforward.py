import pytest
import torch

from src.model.feedforward import FeedForward


class TestFeedForward:
    @pytest.mark.parametrize(
        'x, expected_dim',
        [(torch.rand(2, 3, 768), (2, 3, 768))]
    )
    def test_output_dimension(self, x: torch.Tensor, expected_dim: tuple[int, int]):
        ff = FeedForward(x.shape[-1])
        output = ff.forward(x)

        assert output.shape == expected_dim
