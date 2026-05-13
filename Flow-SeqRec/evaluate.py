"""Entry point for evaluating a trained Flow-SeqRec model."""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from data.dataset import SequentialRecDataset, collate_fn
from models.flow_seqrec import FlowSeqRec
from train import build_model
from utils.metrics import compute_metrics
from utils.utils import load_config, set_seed, get_device


def evaluate(
    model: FlowSeqRec,
    data_loader: DataLoader,
    device: torch.device,
    ks: list,
) -> dict:
    """Full-rank evaluation on the given data loader."""
    model.eval()
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            scores = model.full_rank_scores(
                input_seq=batch["input_seq"],
                seq_len=batch["seq_len"],
            )

            # Mask seen items
            for i in range(scores.size(0)):
                seen = batch["input_seq"][i]
                scores[i, seen] = float("-inf")

            all_scores.append(scores.cpu())
            all_targets.append(batch["target"].cpu())

    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return compute_metrics(all_scores, all_targets, ks=ks)


def measure_inference_speed(
    model: FlowSeqRec,
    data_loader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> float:
    """Measure average inference latency per batch in milliseconds."""
    model.eval()
    sample_batch = next(iter(data_loader))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model.full_rank_scores(
                input_seq=sample_batch["input_seq"],
                seq_len=sample_batch["seq_len"],
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model.full_rank_scores(
                input_seq=sample_batch["input_seq"],
                seq_len=sample_batch["seq_len"],
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow-SeqRec")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--measure_speed", action="store_true", help="Measure inference speed")
    parser.add_argument("--num_ode_steps", type=int, default=None, help="Override ODE steps")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    dataset_cfg = config["dataset"]
    dataset = SequentialRecDataset(
        dataset_cfg["data_dir"],
        split=args.split,
        max_seq_len=dataset_cfg["max_seq_len"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build and load model
    model = build_model(config, dataset)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if args.num_ode_steps is not None:
        model.num_ode_steps = args.num_ode_steps

    print(f"Loaded model from {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Evaluating on {args.split} set ({len(dataset)} samples) ...")

    # Evaluate
    ks = config["evaluation"].get("ks", [5, 10, 20])
    metrics = evaluate(model, data_loader, device, ks)

    print("\nResults (Full-Rank Evaluation):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Speed benchmark
    if args.measure_speed:
        avg_ms = measure_inference_speed(model, data_loader, device)
        print(f"\nAverage inference latency: {avg_ms:.2f} ms/batch")
        print(f"ODE steps: {model.num_ode_steps}")


if __name__ == "__main__":
    main()
