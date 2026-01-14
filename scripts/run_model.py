#!/usr/bin/env python3

"""Run llm."""

import argparse

import torch

from src.inference import generate
from src.model.gpt2 import GPT2
from src.tokenizer import TiktokenTokenizer, text_to_token_ids, token_ids_to_text
from src.utils.config import config
from src.utils.logger import get_logger

_logger = get_logger('Model')


def _load_model(file: str, device: str) -> GPT2:
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Cuda is not available.')
    _logger.info(f'Use device: {device}')

    checkpoint = torch.load(file, map_location=device)
    model = GPT2(config.model.config)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def main(args: argparse.Namespace) -> None:
    device = 'cuda' if args.cuda else 'cpu'
    model = _load_model(args.file, device)
    tokenizer = TiktokenTokenizer()
    print('Model > Hi, how can I help you')

    try:
        while True:
            prompt = input('Prompt > ')
            response = generate(
                model=model,
                token_ids=text_to_token_ids(prompt, tokenizer).to(device),
                max_new_tokens=args.max_new_token,
                context_size=args.context_length,
                temperature=args.temperature,
            )
            print('GPT >', token_ids_to_text(response, tokenizer))
    finally:
        print('Goodbye!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GPT2 Launch')

    parser.add_argument(
        '-f',
        '--file',
        help='Model pkl file path',
        default=config.paths.model_dir + 'gpt2.pkl',
    )

    parser.add_argument(
        '--max-new-token',
        help='Max new token to generate',
        default=1024,
    )

    parser.add_argument(
        '--context-length',
        help='Context size',
        default=config.model.config.context_length,
    )

    parser.add_argument(
        '-t',
        '--temperature',
        help='Model temperature',
        default=0.7,
    )

    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use CUDA',
    )

    main(parser.parse_args())
