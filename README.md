# GPT-2 Replication
A comprehensive PyTorch implementation of OpenAI's GPT-2. This repository supports replicating the original architecture, loading pre-trained weights, and fine-tuning for specific downstream tasks like classification and instruction following.

## Features
- Multi-Scale Support: Fully supports all standard GPT-2 sizes: `124M`, `355M`, `774M`, and `1558M`.
- Pre-trained Weights: Scripts to automatically download and load official OpenAI GPT-2 weights.
- Versatile Training Modes:
    - Pre-training: Train from scratch or continue training on raw text documents.
    - Fine-Tuning (Classification): Add a classification head for tasks like sentiment analysis.
    - Fine-Tuning (Instruct): Optimize the model to follow user instructions (RLHF or SFT styles).Efficient Data Pipeline: Optimized for processing large document corpora.
 
## Installation
```bash
git clone https://github.com/cafesuada24/GPT-2-Replication.git

cd GPT-2-Replication

# use pip
python3 -m pip install -r requirements.txt
# or use uv
uv sync && source .venv/bin/activate # Linux only

```

## Model Weights
You can download the official pre-trained weights for initialization.
```bash
# Options: 124M, 355M, 774M, 1558M
python scripts/download_weights.py --model_size 124M
```

## Usage
### 1. Pre-training on Documents
Train the model on a corpus of text documents (unsupervised learning).
```bash
# Example
python3 train.py \
    --mode pretrain \
    --model_size 124M \
    --data_dir data/the-verdict.txt \
    --train_ratio 0.9 \ # Ratio of training data
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.0004 \
    --output checkpoints/gpt2.pkl

```
### 2. Fine-Tuning: Classification
Fine-tune the model for a specific label (e.g., spam detection, sentiment). This freezes the base layers (optional) and trains a classification head.
### 3. Fine-Tuning: Instruction Following
Fine-tune the model to respond to prompts/instructions (Supervised Fine-Tuning).
*Input format expected: JSONL with `{"prompt": "...", "completion": "..."}`*
### 4. Inference
```bash
# Example
python3 run_model.py \
    --model checkpoints/gpt2_model.pkl \
    --max_new_tokens 200 \
    --context_length 512 \
    --temperature 0.7 \
    --cuda
```
## Configuration
The project supports the following standard configurations via command line or YAML files in `configs/`:

| Model | Layers | d_model | Heads | Parameters |
| ----- | ------ | ------- | ----- | ---------- |
| gpt2 | 12 | 768 | 12 | 124M |
| gpt2-medium | 24 | 1024 |16 | 355M |
| gpt2-large | 36 | 1280 | 20 | 774M |
| gpt2-xl | 48 | 1600 | 25 | 1558M |
