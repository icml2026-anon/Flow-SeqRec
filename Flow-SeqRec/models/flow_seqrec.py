"""Flow-SeqRec: Flow Matching based Generative Sequential Recommendation.

End-to-end architecture that combines:
  1. Transformer sequence encoder for user history
  2. Mixture-of-Experts for sparse feature extraction
  3. Continuous-time flow matching (ODE) for item generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import SequenceEncoder
from models.flow_matching import FlowMatchingModule
from models.moe import SparseFeatureMoE


class FlowSeqRec(nn.Module):
    """Flow-SeqRec model.

    Parameters
    ----------
    num_items : int
        Number of items in the catalog.
    item_embed_dim : int
        Dimension of item embeddings (also the flow data dim).
    hidden_dim : int
        Hidden dimension for the sequence encoder and velocity network.
    num_heads : int
        Number of attention heads in the sequence encoder.
    num_encoder_layers : int
        Number of Transformer layers in the sequence encoder.
    max_seq_len : int
        Maximum input sequence length.
    dropout : float
        Dropout rate.
    flow_hidden_dim : int
        Hidden dimension in the velocity network.
    flow_num_layers : int
        Number of residual blocks in the velocity network.
    num_ode_steps : int
        Number of Euler steps during inference.
    sigma_min : float
        Minimum noise for the flow prior.
    sparse_feature_dim : int or None
        Dimension of sparse item features. None to disable MoE.
    moe_num_experts : int
        Number of experts in MoE.
    moe_top_k : int
        Top-k experts activated.
    moe_hidden_dim : int
        Hidden dim inside each expert.
    moe_load_balance_weight : float
        Load balancing loss weight.
    """

    def __init__(
        self,
        num_items: int,
        item_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        flow_hidden_dim: int = 256,
        flow_num_layers: int = 3,
        num_ode_steps: int = 1,
        sigma_min: float = 1e-4,
        sparse_feature_dim: int = None,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_items = num_items
        self.item_embed_dim = item_embed_dim
        self.hidden_dim = hidden_dim
        self.num_ode_steps = num_ode_steps

        # Item embedding table (shared between encoder input and flow target)
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)

        # Sequence encoder
        self.encoder = SequenceEncoder(
            num_items=num_items,
            embed_dim=item_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Sparse feature MoE (optional)
        self.use_moe = sparse_feature_dim is not None
        if self.use_moe:
            self.sparse_moe = SparseFeatureMoE(
                sparse_dim=sparse_feature_dim,
                embed_dim=item_embed_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                moe_hidden_dim=moe_hidden_dim,
                load_balance_weight=moe_load_balance_weight,
            )

        # Flow matching module
        self.flow = FlowMatchingModule(
            data_dim=item_embed_dim,
            cond_dim=hidden_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
            sigma_min=sigma_min,
        )

    def _get_target_embedding(
        self,
        target: torch.Tensor,
        sparse_features: torch.Tensor = None,
    ) -> tuple:
        """Get target item embedding, optionally fused with sparse features.

        Returns:
            target_embed: (batch, item_embed_dim)
            moe_loss: scalar or 0
        """
        target_embed = self.item_embedding(target)  # (B, embed_dim)
        moe_loss = torch.tensor(0.0, device=target.device)

        if self.use_moe and sparse_features is not None:
            target_embed, moe_loss = self.sparse_moe(target_embed, sparse_features)

        return target_embed, moe_loss

    def forward(
        self,
        input_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
        sparse_features: torch.Tensor = None,
    ) -> dict:
        """Training forward pass.

        Args:
            input_seq: (batch, max_seq_len) padded item ids
            seq_len: (batch,) actual lengths
            target: (batch,) target item ids
            sparse_features: (batch, sparse_dim) optional sparse features

        Returns:
            dict with 'flow_loss', 'moe_loss', 'total_loss'
        """
        # Encode user history
        user_repr = self.encoder(input_seq, seq_len)  # (B, hidden_dim)

        # Target embedding
        target_embed, moe_loss = self._get_target_embedding(target, sparse_features)

        # Flow matching loss
        flow_loss = self.flow.compute_loss(target_embed, user_repr)

        total_loss = flow_loss + moe_loss

        return {
            "flow_loss": flow_loss,
            "moe_loss": moe_loss,
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        input_seq: torch.Tensor,
        seq_len: torch.Tensor,
        num_ode_steps: int = None,
    ) -> torch.Tensor:
        """Generate predicted item embeddings for ranking.

        Args:
            input_seq: (batch, max_seq_len)
            seq_len: (batch,)
            num_ode_steps: override default ODE steps

        Returns:
            pred_embed: (batch, item_embed_dim)
        """
        steps = num_ode_steps if num_ode_steps is not None else self.num_ode_steps
        user_repr = self.encoder(input_seq, seq_len)
        pred_embed = self.flow.sample(user_repr, num_steps=steps)
        return pred_embed

    @torch.no_grad()
    def full_rank_scores(
        self,
        input_seq: torch.Tensor,
        seq_len: torch.Tensor,
        num_ode_steps: int = None,
    ) -> torch.Tensor:
        """Compute scores for ALL items (full-rank evaluation).

        Instead of beam search, we leverage the continuous latent space
        by computing inner products between the predicted embedding and
        all item embeddings.

        Args:
            input_seq: (batch, max_seq_len)
            seq_len: (batch,)

        Returns:
            scores: (batch, num_items+1) -- score for each item
        """
        pred_embed = self.predict(input_seq, seq_len, num_ode_steps)  # (B, embed_dim)

        # All item embeddings (skip padding at index 0)
        all_item_embed = self.item_embedding.weight  # (num_items+1, embed_dim)

        # Inner-product scores
        scores = torch.matmul(pred_embed, all_item_embed.t())  # (B, num_items+1)

        # Mask padding item
        scores[:, 0] = float("-inf")

        return scores
