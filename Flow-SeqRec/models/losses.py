"""Loss functions for Flow-SeqRec."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    """Wrapper for conditional flow matching loss (MSE between predicted
    and ground-truth velocity)."""

    def __init__(self):
        super().__init__()

    def forward(
        self, v_pred: torch.Tensor, v_target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(v_pred, v_target)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss as an optional auxiliary
    objective for direct ranking supervision."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            pos_scores: (batch,) scores for positive items
            neg_scores: (batch,) scores for negative items
        """
        return -F.logsigmoid(pos_scores - neg_scores).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for embedding alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> torch.Tensor:
        """In-batch contrastive loss.

        Args:
            pred_embed: (batch, dim) predicted embeddings
            target_embed: (batch, dim) target embeddings
        """
        pred_embed = F.normalize(pred_embed, dim=-1)
        target_embed = F.normalize(target_embed, dim=-1)

        logits = torch.matmul(pred_embed, target_embed.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
