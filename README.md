# Transformer from Scratch

A GPT-style decoder-only transformer implemented from scratch in PyTorch, including modern architectural components used in large language models.

## Features

- Multi-head self-attention
- KV caching (prefill + decode) for efficient autoregressive generation
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)
- Latent key/value representations for reduced KV cache memory
- Mixture-of-Experts (MoE) layer with top-k routing and load balancing

## Repository Structure

    .
    ├── model.py        # Transformer architecture, attention variants, MoE
    ├── train.py        # Training loop
    ├── sample.py       # Autoregressive text generation
    ├── config.py       # Hyperparameters
    ├── data.py         # Data loading and batching
    ├── input_mod.txt   # Training dataset
    ├── README.md

## Setup

Install dependencies:

    pip install torch

## Train the Model

    python train.py

This trains a decoder-only transformer language model on the provided dataset.

## Generate Samples

    python sample.py

Generates text autoregressively using the trained model with KV caching.

## Implemented Concepts

### KV Caching
Implements prefill and decode stages to avoid recomputing keys and values for past tokens during autoregressive generation, improving inference efficiency.

### Rotary Position Embeddings (RoPE)
Encodes positional information directly into attention via rotation in embedding space.

### Grouped-Query Attention (GQA)
Reduces KV cache memory by sharing key/value projections across groups of query heads.

### Latent Key/Value Representations
Projects keys and values into a lower-dimensional latent space to reduce memory usage during inference.

### Mixture-of-Experts (MoE)
Implements a sparse MoE layer with:
- top-k routing
- capacity constraints
- load balancing loss

## Notes

This project was built to understand and implement modern large language model architectures from first principles, without relying on high-level abstractions.

## Future Improvements

- Add training loss plots
- Add sample outputs
- Extend to larger datasets and models
