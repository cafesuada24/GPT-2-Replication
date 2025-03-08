"""Contains text encoder/decoder"""

import abc

import torch
import tiktoken


class Tokenizer(metaclass=abc.ABCMeta):
    """Encodes/decodes string into tensor of token ids and vice versa"""

    @classmethod
    def __subclasshook__(cls, subclass: type, /) -> bool:
        return (
            hasattr(subclass, "encode")
            and callable(subclass.encode)
            and hasattr(subclass, "decode")
            and callable(subclass.decode)
        )

    @abc.abstractmethod
    def encode(self, text: str, *args, **kwargs) -> list[int]:
        """Encodes text into a tensor of integers"""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, token_ids: list[int], *args, **kwargs) -> str:
        """Decodes an encoded text back to string"""
        raise NotImplementedError


class TiktokenTokenizer(Tokenizer):
    """Uses tiktoken from GPT"""

    def __init__(self, encoding_name: str) -> None:
        super().__init__()

        self.__tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        return self.__tokenizer.encode(text, *args, **kwargs)

    def decode(self, token_ids: list[int], *args, **kwargs) -> str:
        return self.__tokenizer.decode(token_ids, *args, **kwargs)


def get_tokenizer(tokenizer_name: str, init_args: dict):
    """Get corresponding tokenizer"""

    match tokenizer_name:
        case "tiktoken":
            return TiktokenTokenizer(**init_args)
        case _:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

def text_to_token_ids(
        text: str,
        tokenizer: Tokenizer, *,
        allowed_special: set | None = None) -> torch.Tensor:
    """
    Converts text into a tensor of token ids

    Inputs:
        text (str): raw text to convert
        tokenizer (Tokenizer): converter
        allowed_special (set): special characters to be included in dictionary. Default is None
    Outputs:
        Tensor: Shape (batch(1), num tokens)
    """

    encoded = tokenizer.encode(text, allowed_special=allowed_special)
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Tokenizer) -> str:
    """
    Converts token ids back to raw text

    Inputs:
        token_ids (Tensor): token ids of shape (batch size (1), num tokens)
        tokenizer (Tokenizer): converter
    Outputs:
        str: raw text
    """

    token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())
