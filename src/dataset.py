""""""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer


@dataclass
class DataConfig:
    """"""

    max_sequence_len: int = 256
    batch_size: int = 4
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0


class GPTDataset(Dataset):
    """"""

    def __init__(
        self,
        raw_text: str,
        tokenizer: Tokenizer,
        data_config: DataConfig,
    ) -> None:
        super().__init__()

        tokens = tokenizer.encode(raw_text)

        self.inputs = []
        self.targets = []

        for index in range(
            0, len(tokens) - data_config.max_sequence_len, data_config.stride
        ):
            self.inputs.append(
                torch.tensor(tokens[index : index + data_config.max_sequence_len])
            )
            self.targets.append(
                torch.tensor(
                    tokens[index + 1 : index + data_config.max_sequence_len + 1]
                )
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def create_dataloader(
    raw_text: str, tokenizer: Tokenizer, data_config: DataConfig | None = None,
) -> DataLoader:
    """"""

    if data_config is None:
        data_config = DataConfig()

    dataset = GPTDataset(
        raw_text=raw_text,
        tokenizer=tokenizer,
        data_config=data_config,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=data_config.shuffle,
        batch_size=data_config.batch_size,
        drop_last=data_config.drop_last,
        num_workers=data_config.num_workers,
    )

    return dataloader
