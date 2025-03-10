""""""

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer


class GPTDataset(Dataset):
    def __init__(
        self,
        raw_text: str,
        tokenizer: Tokenizer,
        max_sequence_len: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        tokens = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

        self.inputs = []
        self.targets = []

        for index in range(len(tokens) - max_sequence_len, stride):
            self.inputs.append(torch.tensor(tokens[index : index + max_sequence_len]))
            self.targets.append(
                torch.tensor(tokens[index + 1 : index + max_sequence_len + 1])
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def create_dataloader(
    raw_text: str,
    tokenizer: Tokenizer,
    max_sequence_len: int,
    batch_size: int,
    stride: int,
    shuffle: bool,
) -> DataLoader:

    dataset = GPTDataset(
        raw_text=raw_text,
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
        stride=stride,
    )
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return dataloader
