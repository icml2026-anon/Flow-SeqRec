"""Mixture-of-Experts (MoE) module for sparse feature extraction.

Designed to handle highly sparse item features, improving the model's
ability to capture long-tail item characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single feed-forward expert network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Top-k gated Mixture-of-Experts layer.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension inside each expert.
    output_dim : int
        Output dimension.
    num_experts : int
        Number of expert networks.
    top_k : int
        Number of experts activated per token.
    load_balance_weight : float
        Coefficient for the auxiliary load-balancing loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple:
        """Forward pass.

        Args:
            x: (batch, input_dim)

        Returns:
            output: (batch, output_dim)
            aux_loss: scalar load-balancing loss
        """
        # Gating scores
        logits = self.gate(x)  # (B, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, top_k)

        # Compute expert outputs for selected experts
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=x.device)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (B,)
            weight = top_k_weights[:, k].unsqueeze(-1)  # (B, 1)

            for e_id in range(self.num_experts):
                mask = expert_idx == e_id
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e_id](expert_input)
                    output[mask] += weight[mask] * expert_output

        # Auxiliary load-balancing loss
        # Fraction of tokens routed to each expert
        gates_softmax = F.softmax(logits, dim=-1)  # (B, num_experts)
        density = gates_softmax.mean(dim=0)  # (num_experts,)
        # Fraction of routing probability assigned to each expert
        routing_prob = F.softmax(logits, dim=-1).mean(dim=0)

        aux_loss = self.load_balance_weight * self.num_experts * (
            density * routing_prob
        ).sum()

        return output, aux_loss


class SparseFeatureMoE(nn.Module):
    """MoE-based sparse feature processor that projects raw sparse item
    features into a dense representation, then fuses with the main item
    embedding via a gating mechanism.

    Parameters
    ----------
    sparse_dim : int
        Dimension of raw sparse features.
    embed_dim : int
        Target item embedding dimension.
    num_experts : int
        Number of experts.
    top_k : int
        Top-k experts per token.
    moe_hidden_dim : int
        Hidden dim inside each expert.
    load_balance_weight : float
        Load balance loss coefficient.
    """

    def __init__(
        self,
        sparse_dim: int,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.moe = MixtureOfExperts(
            input_dim=sparse_dim,
            hidden_dim=moe_hidden_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            load_balance_weight=load_balance_weight,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        item_embed: torch.Tensor,
        sparse_features: torch.Tensor,
    ) -> tuple:
        """Fuse item embedding with sparse features via gated MoE.

        Args:
            item_embed: (batch, embed_dim)
            sparse_features: (batch, sparse_dim)

        Returns:
            fused: (batch, embed_dim)
            aux_loss: scalar
        """
        sparse_repr, aux_loss = self.moe(sparse_features)
        gate = self.fusion_gate(torch.cat([item_embed, sparse_repr], dim=-1))
        fused = gate * item_embed + (1 - gate) * sparse_repr
        return fused, aux_loss
