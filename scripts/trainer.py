#!/usr/bin/env python3

"""GPT2 Trainer."""

import argparse
import time
from pathlib import Path

import torch

from src.dataset.dataset import DataConfig, create_dataloader, train_test_split
from src.model.gpt2 import GPT2
from src.tokenizer import TiktokenTokenizer
from src.trainer import train_model
from src.utils.config import config
from src.utils.logger import get_logger


def _pretrain(args: argparse.Namespace) -> None:
    logger = get_logger('Pretrainer')
    with open(args.file, encoding='utf-8') as f:
        raw_text = f.read()

    logger.info(f'Loaded "{args.file}" with {len(raw_text)} characters.')

    config.model.config.context_length = args.context_length

    tokenizer = TiktokenTokenizer()

    train_text, test_text = train_test_split(raw_text, train_ratio=args.train_ratio)

    train_loader = create_dataloader(
        train_text,
        tokenizer,
        DataConfig(
            batch_size=args.batch_size,
            max_sequence_len=config.model.config.context_length,
            stride=config.model.config.context_length,
        ),
    )
    test_loader = create_dataloader(
        test_text,
        tokenizer,
        DataConfig(
            batch_size=args.batch_size,
            max_sequence_len=config.model.config.context_length,
            stride=config.model.config.context_length,
            drop_last=False,
            shuffle=False,
        ),
    )

    logger.info(f'Data splitted with train ratio of {args.train_ratio}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Selected device {device}.')

    model = GPT2(config.model.config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model initialized with {total_params:,} parameters.')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    logger.info('Optimizer loaded.')
    logger.info('Training hyper-parameters:')
    logger.info(f'  Number of epochs: {args.num_epochs}')
    logger.info(f'  Batch size: {args.batch_size}')
    logger.info(f'  Learning rate: {args.learning_rate}')

    logger.info('Pretraining process started.')

    start_time = time.perf_counter()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        n_epochs=args.num_epochs,
        eval_freq=5,
        eval_iter=5,
        logger=logger,
        tokenizer=tokenizer,
        start_context='Every effort moves you',
    )
    end_time = time.perf_counter()

    logger.info(
        f'Pretraining process finished in {end_time - start_time:0.4f} seconds.',
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimier_state_dict': optimizer.state_dict(),
            'config': {
                'context_length': config.model.config.context_length,
            },
        },
        args.output,
    )

    logger.info(f'Model saved to {args.output}.')


def _finetune(args: argparse.Namespace) -> None: ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GPT2 Trainer')

    subparser = parser.add_subparsers(dest='command', help='Available subcommands.')

    #####################
    # Pretrainer parser #
    #####################

    pretrain_parser = subparser.add_parser('pretrain', help='Pretrain a new model')

    pretrain_parser.add_argument(
        '-f',
        '--file',
        help='Training data',
        default=config.paths.data_dir + 'the-verdict.txt',
    )
    pretrain_parser.add_argument(
        '--model_size',
        help='Model size (124M, 355M, 774M, 1558M)',
        default='124M',
        choices=['124M', '355M', '774M', '1558M'],
    )
    pretrain_parser.add_argument(
        '-o',
        '--output',
        help='Model saving path',
        default=config.paths.model_dir + 'gpt2.pth',
    )
    pretrain_parser.add_argument(
        '--num_epochs',
        help='Number of training epochs',
        default=10,
        type=int,
    )
    pretrain_parser.add_argument(
        '--context_length',
        help="Model's context length",
        default=1024,
        type=int,
    )
    pretrain_parser.add_argument(
        '--train_ratio',
        help='Ratio of training data',
        default=0.9,
        type=float,
    )
    pretrain_parser.add_argument(
        '--batch_size',
        help='Size of a batch',
        default=2,
        type=float,
    )
    pretrain_parser.add_argument(
        '-l',
        '--learning_rate',
        help='Learning rate',
        default=0.0004,
        type=float,
    )
    pretrain_parser.set_defaults(func=_pretrain)

    ###################
    # Finetune parser #
    ###################

    finetune_parser = subparser.add_parser(
        'finetune',
        help='Finetune an existing model',
    )

    finetune_parser.set_defaults(func=_finetune)
    # ----------------

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
