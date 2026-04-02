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
Projects hidden states into a lower-dimensional latent space and computes keys and values from that representation to reduce KV cache memory during inference.

### Mixture-of-Experts (MoE)
Implements a sparse MoE layer with:
- top-k routing
- capacity constraints
- load balancing loss

## Notes

This project was built to understand and implement modern large language model architectures from first principles, without relying on high-level abstractions.

## Sample Output

After training on `input_mod.txt` using character-level tokens, the model generated:

*Note: This sample is from a short 500-step training run (~10 minutes) for demonstration purposes. The model has not converged; longer training will yield higher-quality generations.*

```text
First Warwick and all the demands of her lives,
Which had not promised us and most deserve.

Boy:
But say'st we are lion, and will not confirm
Than a daughter's walk floucester, with his hand,
Than he store father and the counterior,
By that minate duke may be seen to sleep:
The nuptial dalm, or white we may be set encomplexion,
Thou hadst pull up his sweet soldier and dug as welcome.

ANGELO:
She had a foot, now my gentle give better
Which the power dugs and trumpets the flatter than
To Warwick, that devices stood my face elter.

AUFIDIUS:
I have a word of light up that repeal thought,
To unruly never walls have been him for me.

ROMEO:
At God.

Gross:
How should say the senator, the gates like,
For this accusation by the city, and be sight;
Who, that we'll we have thee day from Juliet,
Where thou swear from his past in thing and bed
The kindness but they have fresh before the ramp.

STANLEY:
O lay to chide where he shall know the crown
Against the truth: I do there that I have
Is thy
```

## Future Improvements

- Add training loss plots
- Add sample outputs
- Extend to larger datasets and models
