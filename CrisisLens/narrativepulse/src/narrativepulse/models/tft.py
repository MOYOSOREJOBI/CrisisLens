from __future__ import annotations

import torch
from torch import nn


class GatedResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = torch.relu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(residual + gated)


class SimpleTFT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizons: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.grn = GatedResidualBlock(hidden_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(hidden_dim, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        hidden = self.grn(hidden)
        encoded = self.encoder(hidden)
        return self.out(encoded[:, -1, :])
