"""Inference script for Flow-SeqRec.

Given a user's historical interaction sequence, generates top-K
item recommendations using the trained flow matching model.
"""

import argparse
from typing import List

import torch
import numpy as np

from models.flow_seqrec import FlowSeqRec
from train import build_model
from utils.utils import load_config, get_device


def recommend(
    model: FlowSeqRec,
    item_sequence: List[int],
    max_seq_len: int,
    device: torch.device,
    top_k: int = 10,
    num_ode_steps: int = 1,
    exclude_seen: bool = True,
) -> List[int]:
    """Generate top-K recommendations for a single user.

    Args:
        model: Trained FlowSeqRec model.
        item_sequence: List of historical item IDs.
        max_seq_len: Maximum sequence length (for padding).
        device: Torch device.
        top_k: Number of recommendations.
        num_ode_steps: ODE solver steps.
        exclude_seen: Whether to exclude already-interacted items.

    Returns:
        List of recommended item IDs.
    """
    model.eval()

    # Prepare input
    seq_len = min(len(item_sequence), max_seq_len)
    if len(item_sequence) > max_seq_len:
        item_sequence = item_sequence[-max_seq_len:]

    padded = [0] * (max_seq_len - seq_len) + item_sequence

    input_seq = torch.tensor([padded], dtype=torch.long, device=device)
    seq_len_t = torch.tensor([seq_len], dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model.full_rank_scores(input_seq, seq_len_t, num_ode_steps=num_ode_steps)

    if exclude_seen:
        seen = torch.tensor(item_sequence, dtype=torch.long, device=device)
        scores[0, seen] = float("-inf")

    _, topk_ids = torch.topk(scores[0], top_k)
    return topk_ids.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Flow-SeqRec Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--items",
        type=int,
        nargs="+",
        required=True,
        help="Historical item IDs (space-separated)",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_ode_steps", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"]["device"])

    # Build a minimal dataset object to get metadata
    from data.dataset import SequentialRecDataset
    dataset_cfg = config["dataset"]
    dataset = SequentialRecDataset(
        dataset_cfg["data_dir"],
        split="test",
        max_seq_len=dataset_cfg["max_seq_len"],
    )

    model = build_model(config, dataset)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Input sequence: {args.items}")
    recommendations = recommend(
        model=model,
        item_sequence=args.items,
        max_seq_len=dataset_cfg["max_seq_len"],
        device=device,
        top_k=args.top_k,
        num_ode_steps=args.num_ode_steps,
    )
    print(f"Top-{args.top_k} recommendations: {recommendations}")


if __name__ == "__main__":
    main()
