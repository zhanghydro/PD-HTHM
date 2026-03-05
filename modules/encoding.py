# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _check_3d(x: torch.Tensor, name: str = "x") -> None:
    if x.dim() != 3:
        raise ValueError(f"{name} must be [B,T,D], got {tuple(x.shape)}")


def _check_last_dim(x: torch.Tensor, d_model: int, name: str = "x") -> None:
    if x.size(-1) != d_model:
        raise ValueError(f"{name} last dim must be {d_model}, got {x.size(-1)}")


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input:
      x: [B, T, D]
    Output:
      x + PE: [B, T, D]
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.dropout = nn.Dropout(p=float(dropout))

        pe = self._build_sinusoidal_pe(self.max_len, self.d_model)  # [max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))                 # [1, max_len, d_model]

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "x")
        _check_last_dim(x, self.d_model, "x")

        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        pe = self.pe[:, :T, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x + pe)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Each position has a learnable embedding.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.encoding = nn.Parameter(torch.randn(1, self.max_len, self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "x")
        _check_last_dim(x, self.d_model, "x")

        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        return x + self.encoding[:, :T, :].to(dtype=x.dtype, device=x.device)


class ZeroPositionalEncoding(nn.Module):
    """
    No-op positional encoding (adds zero vectors).
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.register_buffer("pe", torch.zeros(1, self.max_len, self.d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "x")
        _check_last_dim(x, self.d_model, "x")

        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        return x + self.pe[:, :T, :].to(dtype=x.dtype, device=x.device)


class Time2VecEncoding(nn.Module):
    """
    Time2Vec encoding.
    Input:
      t: [B, T, time_feature_dim]
    Output:
      [B, T, d_model]
    """
    def __init__(self, d_model: int, time_feature_dim: int):
        super().__init__()
        self.d_model = int(d_model)
        self.time_feature_dim = int(time_feature_dim)
        if self.d_model < 2:
            raise ValueError("d_model must be >= 2 for Time2VecEncoding")

        self.linear = nn.Linear(self.time_feature_dim, 1)
        self.freq = nn.Parameter(torch.randn(self.time_feature_dim, self.d_model - 1))
        self.phase = nn.Parameter(torch.randn(self.time_feature_dim, self.d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        _check_3d(t, "t")
        if t.size(-1) != self.time_feature_dim:
            raise ValueError(f"t last dim must be {self.time_feature_dim}, got {t.size(-1)}")

        linear_term = self.linear(t)  # [B,T,1]
        periodic = torch.sin(torch.matmul(t, self.freq) + self.phase)  # [B,T,d_model-1]
        return torch.cat([linear_term, periodic], dim=-1)              # [B,T,d_model]


class RotaryPositionalEmbedding(nn.Module):
    """
    Stub for rotary positional embedding.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x