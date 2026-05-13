# Flow-SeqRec

**Flow Matching based Generative Sequential Recommendation**

Flow-SeqRec is an end-to-end generative sequential recommendation system built on continuous flow matching. It addresses the key deployment bottlenecks of existing diffusion-based recommendation models -- high inference latency and large sampling truncation errors when handling discrete user behavior data.

## Key Features

- **Continuous-Time ODE Trajectory**: Maps user interaction histories to target item latent representations via an optimal-transport conditional flow, replacing noisy multi-step diffusion with a smooth deterministic path.
- **Single-Step Euler Inference**: Achieves ~3.5x faster online inference compared to state-of-the-art generative baselines by leveraging a one-step ODE solver.
- **Mixture-of-Experts (MoE) Sparse Feature Extraction**: A top-k gated MoE module processes highly sparse item features, significantly improving long-tail item discovery and distribution.
- **Full-Rank Evaluation**: Replaces beam search with full-rank scoring over all items, fully exploiting the continuous latent space structure.

## Benchmark Results

| Dataset | Recall@10 | NDCG@10 |
|---|---|---|
| Amazon-Books | 0.1245 | 0.0812 |
| MovieLens-25M | 0.2314 | 0.2058 |

## Project Structure

```
Flow-SeqRec/
├── configs/                  # YAML configuration files
│   ├── amazon_books.yaml
│   └── movielens_25m.yaml
├── data/                     # Dataset loading and preprocessing
│   ├── dataset.py
│   └── preprocess.py
├── models/                   # Core model components
│   ├── encoder.py            # Transformer sequence encoder
│   ├── flow_matching.py      # Flow matching ODE module
│   ├── flow_seqrec.py        # Main model integrating all components
│   ├── losses.py             # Loss functions
│   └── moe.py                # Mixture-of-Experts module
├── trainers/
│   └── trainer.py            # Training loop with early stopping
├── utils/
│   ├── metrics.py            # Recall@K, NDCG@K (full-rank)
│   └── utils.py              # Seed, config, early stopping utilities
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
├── inference.py              # Single-user inference entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

### Amazon-Books

Download the Amazon-Books review dataset and run:

```bash
python -m data.preprocess --dataset amazon_books \
    --raw_path /path/to/Books.csv \
    --output_dir ./data/amazon_books
```

### MovieLens-25M

Download MovieLens-25M from https://grouplens.org/datasets/movielens/25m/ and run:

```bash
python -m data.preprocess --dataset movielens_25m \
    --raw_path /path/to/ml-25m/ratings.csv \
    --output_dir ./data/movielens_25m
```

## Training

```bash
# Amazon-Books
python train.py --config configs/amazon_books.yaml

# MovieLens-25M
python train.py --config configs/movielens_25m.yaml
```

Training logs are written to TensorBoard. Monitor with:

```bash
tensorboard --logdir ./logs
```

## Evaluation

Full-rank evaluation on the test set:

```bash
python evaluate.py \
    --config configs/amazon_books.yaml \
    --checkpoint ./checkpoints/amazon_books/best_model.pt \
    --split test \
    --measure_speed
```

## Inference

Generate top-K recommendations for a user given their interaction history:

```bash
python inference.py \
    --config configs/amazon_books.yaml \
    --checkpoint ./checkpoints/amazon_books/best_model.pt \
    --items 42 107 256 389 512 \
    --top_k 10
```

## Architecture Overview

```
User History [i1, i2, ..., iN]
         |
         v
 +------------------+
 | Item Embedding   |
 +------------------+
         |
         v
 +------------------+
 | Transformer      |    +--------------------+
 | Sequence Encoder | -> | User Representation|
 +------------------+    +--------------------+
                                  |
                                  v (condition)
                    +----------------------------+
                    | Flow Matching ODE          |
                    | x_0 ~ N(0,I) --v(x,t,c)-> |
                    | Single Euler Step          |
                    +----------------------------+
                                  |
                                  v
                    +----------------------------+
                    | Predicted Item Embedding   |
                    +----------------------------+
                                  |
         +------------------------+
         |                        |
         v                        v
 +--------------+     +---------------------+
 | Inner Product|     | MoE Sparse Feature  |
 | with All     |     | Fusion (Training)   |
 | Item Embeds  |     +---------------------+
 +--------------+
         |
         v
   Top-K Items
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{flow-seqrec-2026,
  title={Flow-SeqRec: Flow Matching based Generative Sequential Recommendation},
  year={2026}
}
```

## License

MIT License
