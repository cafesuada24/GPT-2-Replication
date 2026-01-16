"""Contains the Dataset and DataLoader implementation for the GPT model."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import Tokenizer


@dataclass
class DataConfig:
    """"""

    max_sequence_len: int = 256
    batch_size: int = 4
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0


type DataSample = tuple[torch.Tensor, torch.Tensor]


class GPTDataset(Dataset[DataSample]):
    """"""

    def __init__(
        self,
        raw_text: str,
        tokenizer: Tokenizer,
        data_config: DataConfig,
    ) -> None:
        super().__init__()

        tokens = tokenizer.encode(raw_text)

        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for index in range(
            0,
            len(tokens) - data_config.max_sequence_len,
            data_config.stride,
        ):
            self.inputs.append(
                torch.tensor(tokens[index : index + data_config.max_sequence_len]),
            )
            self.targets.append(
                torch.tensor(
                    tokens[index + 1 : index + data_config.max_sequence_len + 1],
                ),
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> DataSample:
        return self.inputs[index], self.targets[index]


def create_dataloader(
    raw_text: str,
    tokenizer: Tokenizer,
    data_config: DataConfig | None = None,
) -> DataLoader[DataSample]:
    """"""

    if data_config is None:
        data_config = DataConfig()

    dataset = GPTDataset(
        raw_text=raw_text,
        tokenizer=tokenizer,
        data_config=data_config,
    )

    return DataLoader(
        dataset,
        shuffle=data_config.shuffle,
        batch_size=data_config.batch_size,
        drop_last=data_config.drop_last,
        num_workers=data_config.num_workers,
    )



def train_test_split(
    raw_text: str,
    train_ratio: float = 0.9,
) -> tuple[str, str]:
    split_index = int(train_ratio * len(raw_text))
    train_text = raw_text[:split_index]
    test_text = raw_text[split_index:]

    return train_text, test_text
