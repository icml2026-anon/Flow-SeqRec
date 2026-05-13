"""Training loop for Flow-SeqRec."""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.flow_seqrec import FlowSeqRec
from utils.metrics import compute_metrics
from utils.utils import AverageMeter, EarlyStopping


class Trainer:
    """End-to-end trainer for Flow-SeqRec.

    Parameters
    ----------
    model : FlowSeqRec
        The model to train.
    config : dict
        Training configuration (from YAML).
    train_loader : DataLoader
        Training data loader.
    valid_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader, optional
        Test data loader.
    device : torch.device
        Device to train on.
    """

    def __init__(
        self,
        model: FlowSeqRec,
        config: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

        train_cfg = config["training"]
        eval_cfg = config["evaluation"]

        self.num_epochs = train_cfg["num_epochs"]
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.ks = eval_cfg.get("ks", [5, 10, 20])
        self.eval_batch_size = eval_cfg.get("eval_batch_size", 128)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR scheduler: linear warmup + cosine annealing
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("patience", 20),
            mode="max",
        )

        # Logging
        log_dir = train_cfg.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Checkpointing
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_metric = 0.0
        self.best_epoch = 0

    def train(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Best validation metrics.
        """
        print(f"Starting training for {self.num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_metrics = {}

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss = self._train_epoch(epoch)

            # Validate
            val_metrics = self._evaluate(self.valid_loader)

            # Log
            self.writer.add_scalar("train/loss", train_loss, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f"valid/{name}", value, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Print progress
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Loss: {train_loss:.4f} | {metric_str}"
            )

            # Checkpoint on best NDCG@10
            primary_metric = val_metrics.get("NDCG@10", 0.0)
            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                self.best_epoch = epoch
                best_metrics = val_metrics.copy()
                self._save_checkpoint(epoch, val_metrics)

            # LR scheduler step
            self.scheduler.step()

            # Early stopping
            if self.early_stopping.step(primary_metric):
                print(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break

        self.writer.close()

        # Final test evaluation
        if self.test_loader is not None:
            print("\nLoading best model for test evaluation ...")
            self._load_best_checkpoint()
            test_metrics = self._evaluate(self.test_loader)
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
            print(f"Test Results | {metric_str}")
            return test_metrics

        return best_metrics

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_seq=batch["input_seq"],
                seq_len=batch["seq_len"],
                target=batch["target"],
                sparse_features=batch.get("sparse_features"),
            )

            loss = outputs["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            loss_meter.update(loss.item(), batch["input_seq"].size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg

    @torch.no_grad()
    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model using full-rank protocol."""
        self.model.eval()

        all_scores = []
        all_targets = []

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            scores = self.model.full_rank_scores(
                input_seq=batch["input_seq"],
                seq_len=batch["seq_len"],
            )

            # Mask items already in the input sequence to avoid data leakage
            for i in range(scores.size(0)):
                seen_items = batch["input_seq"][i]
                scores[i, seen_items] = float("-inf")

            all_scores.append(scores.cpu())
            all_targets.append(batch["target"].cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(all_scores, all_targets, ks=self.ks)
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        print(f"  -> Saved best model (epoch {epoch}) to {path}")

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  -> Loaded best model from epoch {checkpoint['epoch']}")
