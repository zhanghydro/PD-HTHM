# -*- coding: utf-8 -*-
"""
data_processing.py
- Build sliding-window datasets and PyTorch DataLoaders from numpy arrays.
- Fit scaler on TRAIN, reuse for VAL/TEST.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class StandardScaler:
    """Standardize X and y using mean/std from training data."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "StandardScaler":
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0) + 1e-8
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean


class WindowDataset(Dataset):
    """
    Sliding-window dataset.

    X: [N, F], y: [N] or [N, 1]
    For each window of length T:
      - input  = X[t-T+1 : t+1]
      - target = y[t]  (the last day in the window)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int,
        scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        if X.ndim != 2:
            raise ValueError(f"X must be [N, F], got shape={X.shape}")

        X = X.astype(np.float32)
        y = y.reshape(-1, 1).astype(np.float32)

        self.seq_length = int(seq_length)

        if scaler is None:
            scaler = StandardScaler.fit(X, y)
        self.scaler = scaler

        self.Xn = scaler.transform_x(X).astype(np.float32)
        self.yn = scaler.transform_y(y).astype(np.float32)

        N = self.Xn.shape[0]
        self.starts = np.arange(0, max(0, N - self.seq_length + 1), dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + self.seq_length
        x_win = self.Xn[s:e, :]        # [T, F]
        y_t = self.yn[e - 1, 0:1]      # [1]
        return torch.from_numpy(x_win), torch.from_numpy(y_t)


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seq_length: int,
    batch_train: int = 256,
    batch_eval: int = 1024,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Create DataLoaders. Scaler is fitted on TRAIN and reused for VAL/TEST."""
    train_ds = WindowDataset(X_train, y_train, seq_length=seq_length, scaler=None)
    scaler = train_ds.scaler
    val_ds = WindowDataset(X_val, y_val, seq_length=seq_length, scaler=scaler)
    test_ds = WindowDataset(X_test, y_test, seq_length=seq_length, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False)
    return train_loader, val_loader, test_loader, scaler