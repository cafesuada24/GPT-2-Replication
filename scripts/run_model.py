#!/usr/bin/env python3

"""Run llm."""

import argparse
from typing import Any

import torch

from scripts.gpt_download import download_and_load_gpt2
from src.inference import generate
from src.model.gpt2 import GPT2, load_weights_into_gpt
from src.tokenizer import TiktokenTokenizer, text_to_token_ids, token_ids_to_text
from src.utils.config import config
from src.utils.logger import get_logger

_logger = get_logger('Model')


def _load_model(args: argparse.Namespace, device: str) -> tuple[dict[str, Any], GPT2]:
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Cuda is not available.')
    _logger.info(f'Use device: {device}')

    settings = None
    params = None
    checkpoint = None

    if args.custom_model:
        checkpoint = torch.load(args.model_path, map_location=device)
        settings = checkpoint['config']
        if 'context_length' not in settings:
            settings['context_length'] = config.model.config.context_length
        config.model.config.context_length = settings['context_length']
    else:
        settings, params = download_and_load_gpt2(
            model_size=args.gpt_model,
            models_dir=args.model_path,
        )
        config.model.config.context_length = settings['n_ctx']  # GPT use context length of 1024
        config.model.config.qkv_bias = True
        settings['context_length'] = settings['n_ctx']

    model = GPT2(config.model.config)

    if args.custom_model:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        load_weights_into_gpt(model, params)

    model.to(device)
    model.eval()

    return settings, model


def main(args: argparse.Namespace) -> None:
    device = 'cuda' if args.cuda else 'cpu'
    settings, model = _load_model(args, device)
    tokenizer = TiktokenTokenizer()

    print('Model > Hi, how can I help you')

    try:
        while True:
            prompt = input('Prompt > ')
            response = generate(
                model=model,
                token_ids=text_to_token_ids(prompt, tokenizer).to(device),
                max_new_tokens=args.max_new_tokens,
                context_size=settings['context_length'],
                temperature=args.temperature,
            )
            print('GPT >', token_ids_to_text(response, tokenizer))
    finally:
        print('Goodbye!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GPT2 Launch')

    model_mode_group = parser.add_mutually_exclusive_group(required=True)

    model_mode_group.add_argument(
        '--gpt_model',
        help='Use OpenAI GPT2 weight',
        default='124M',
        choices=['124M', '355M', '774M', '1558M'],
    )

    model_mode_group.add_argument(
        '--custom_model',
        action='store_true',
        help='Use custom model',
    )

    parser.add_argument(
        '--model_path',
        help='Model path',
        required=True,
    )

    parser.add_argument(
        '--max_new_tokens',
        help='Max new tokens to generate',
        default=200,
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
