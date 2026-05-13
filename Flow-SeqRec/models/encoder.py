"""Transformer-based sequence encoder for user interaction histories."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x -- (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    """Transformer encoder that maps a padded item-id sequence into a
    fixed-length user representation vector.

    Parameters
    ----------
    num_items : int
        Total number of items (id 0 is reserved for padding).
    embed_dim : int
        Item embedding dimension.
    hidden_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    max_seq_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # +1 for padding token (index 0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_seq: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded sequence of item ids.

        Args:
            input_seq: (batch, max_seq_len) -- left-padded item ids
            seq_len: (batch,) -- actual sequence lengths

        Returns:
            (batch, hidden_dim) -- user representation
        """
        padding_mask = input_seq == 0  # True where padded

        x = self.item_embedding(input_seq)  # (B, L, embed_dim)
        x = self.proj(x)  # (B, L, hidden_dim)
        x = self.pos_enc(x)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        # Gather the representation at the last valid position for each sample
        batch_size = x.size(0)
        last_idx = (input_seq.size(1) - 1) * torch.ones(
            batch_size, dtype=torch.long, device=x.device
        )
        # last valid position = max_seq_len - 1 (since left-padded, last token is always valid)
        user_repr = x[torch.arange(batch_size, device=x.device), last_idx]

        return user_repr
