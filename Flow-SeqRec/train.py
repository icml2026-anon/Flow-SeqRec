"""Entry point for training Flow-SeqRec."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from data.dataset import SequentialRecDataset, collate_fn
from models.flow_seqrec import FlowSeqRec
from trainers.trainer import Trainer
from utils.utils import load_config, set_seed, get_device


def build_model(config: dict, dataset: SequentialRecDataset) -> FlowSeqRec:
    """Construct the FlowSeqRec model from config and dataset metadata."""
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    sparse_dim = None
    if dataset.sparse_features is not None:
        sparse_dim = dataset.sparse_features.shape[1]

    model = FlowSeqRec(
        num_items=dataset.num_items,
        item_embed_dim=model_cfg["item_embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        max_seq_len=dataset_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        flow_hidden_dim=model_cfg["flow_hidden_dim"],
        flow_num_layers=model_cfg["flow_num_layers"],
        num_ode_steps=model_cfg["num_ode_steps"],
        sigma_min=model_cfg["sigma_min"],
        sparse_feature_dim=sparse_dim,
        moe_num_experts=model_cfg["moe_num_experts"],
        moe_top_k=model_cfg["moe_top_k"],
        moe_hidden_dim=model_cfg["moe_hidden_dim"],
        moe_load_balance_weight=model_cfg["moe_load_balance_weight"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Flow-SeqRec")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["training"]["seed"]
    set_seed(seed)

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    # Load datasets
    dataset_cfg = config["dataset"]
    data_dir = dataset_cfg["data_dir"]
    max_seq_len = dataset_cfg["max_seq_len"]

    train_dataset = SequentialRecDataset(data_dir, split="train", max_seq_len=max_seq_len)
    valid_dataset = SequentialRecDataset(data_dir, split="valid", max_seq_len=max_seq_len)
    test_dataset = SequentialRecDataset(data_dir, split="test", max_seq_len=max_seq_len)

    print(
        f"Dataset: {dataset_cfg['name']} | "
        f"Items: {train_dataset.num_items} | "
        f"Users: {train_dataset.num_users}"
    )
    print(
        f"Train: {len(train_dataset)} | "
        f"Valid: {len(valid_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build model
    model = build_model(config, train_dataset)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
    )

    results = trainer.train()
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
