import pytest
import torch

from src.tokenizer import (
    TiktokenTokenizer,
    Tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


class TestTokenizer:
    tokenizer: Tokenizer = TiktokenTokenizer('gpt2')

    @pytest.mark.parametrize('text, result', [('Hello, I am', [[15496, 11, 314, 716]])])
    def test_text_to_token_ids(self, text: str, result: list[list[int]]) -> None:
        token_ids = text_to_token_ids(
            text,
            self.tokenizer,
            allowed_special={'<|endoftext|>'},
        )

        assert token_ids.tolist() == result

    @pytest.mark.parametrize(
        'token_ids, result', [(torch.tensor([[15496, 11, 314, 716]]), 'Hello, I am')]
    )
    def test_token_ids_to_text(self, token_ids: torch.Tensor, result: str) -> None:
        raw_text = token_ids_to_text(token_ids, self.tokenizer)

        assert raw_text == result
