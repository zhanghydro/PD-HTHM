# -*- coding: utf-8 -*-
"""
Data pipeline utilities.

Provides:
- CSV reading with column checks
- Date-based split with optional warmup concatenation
- Conversion to numpy arrays (x, y) with optional missing-y masking
- Wrap-window dataset for training (stride-based)
- Eval tensors helper (single-batch full sequence)

No file paths are hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =========================================================
# Config
# =========================================================
@dataclass(frozen=True)
class DataColumns:
    date: str = "date"
    temp: str = "temperature"
    prec: str = "precipitation"
    pet: str = "pet"
    q: str = "discharge_spec"


@dataclass(frozen=True)
class DateRange:
    start: str  # inclusive
    end: str    # inclusive


@dataclass(frozen=True)
class SplitConfig:
    cols: DataColumns = DataColumns()

    # If you only need simple train/val/test without warmup, set warmup ranges to None.
    train: DateRange = DateRange("1975-01-01", "1999-12-31")
    val_warmup: Optional[DateRange] = None
    val_main: Optional[DateRange] = None
    test_warmup: Optional[DateRange] = None
    test_main: Optional[DateRange] = None

    # Missing target handling
    mask_missing_y: bool = False  # True: keep rows with missing y and provide mask
    drop_missing_forcings: bool = True  # drop rows missing temp/prec/pet


# =========================================================
# IO + basic checks
# =========================================================
def read_time_series_csv(path: str, cfg: SplitConfig) -> pd.DataFrame:
    df = pd.read_csv(path)

    c = cfg.cols
    required = [c.date, c.temp, c.prec, c.pet, c.q]
    missing = [k for k in required if k not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[c.date] = pd.to_datetime(df[c.date], errors="coerce")
    df = df.dropna(subset=[c.date])

    if cfg.drop_missing_forcings:
        df = df.dropna(subset=[c.temp, c.prec, c.pet])

    if not cfg.mask_missing_y:
        df = df.dropna(subset=[c.q])

    df = df.sort_values(c.date).reset_index(drop=True)
    return df


# =========================================================
# Splitting
# =========================================================
def split_by_ranges(df: pd.DataFrame, cfg: SplitConfig):
    c = cfg.cols

    train = _slice_range(df, c.date, cfg.train)

    val = None
    val_w = None
    test = None
    test_w = None

    if cfg.val_main is not None:
        val_m = _slice_range(df, c.date, cfg.val_main)
        if cfg.val_warmup is not None:
            val_w = _slice_range(df, c.date, cfg.val_warmup)
            val = pd.concat([val_w, val_m], ignore_index=True)
        else:
            val = val_m

    if cfg.test_main is not None:
        test_m = _slice_range(df, c.date, cfg.test_main)
        if cfg.test_warmup is not None:
            test_w = _slice_range(df, c.date, cfg.test_warmup)
            test = pd.concat([test_w, test_m], ignore_index=True)
        else:
            test = test_m

    _assert_non_empty(train, "train")
    if cfg.val_main is not None:
        _assert_non_empty(val, "val")
    if cfg.test_main is not None:
        _assert_non_empty(test, "test")

    return train, val, test, val_w, test_w


def _slice_range(df: pd.DataFrame, date_col: str, r: DateRange) -> pd.DataFrame:
    s = pd.to_datetime(r.start)
    e = pd.to_datetime(r.end)
    out = df[(df[date_col] >= s) & (df[date_col] <= e)].copy()
    return out.reset_index(drop=True)


def _assert_non_empty(df: Optional[pd.DataFrame], name: str):
    if df is None or len(df) == 0:
        raise ValueError(f"Split '{name}' is empty. Check date ranges.")


# =========================================================
# Conversion to arrays with mask
# =========================================================
def df_to_arrays(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x: [T, 3]
      y: [T, 1]
      m: [T, 1] (1=valid, 0=missing) if mask_missing_y else all ones
    """
    c = cfg.cols

    x = df[[c.temp, c.prec, c.pet]].to_numpy(dtype=np.float32)
    y = df[[c.q]].to_numpy(dtype=np.float32)

    if cfg.mask_missing_y:
        m = (~np.isnan(y[:, 0])).astype(np.float32)[:, None]
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    else:
        m = np.ones_like(y, dtype=np.float32)

    return x, y, m


def warmup_length(df_warmup: Optional[pd.DataFrame]) -> int:
    return 0 if df_warmup is None else int(len(df_warmup))


# =========================================================
# Training windows
# =========================================================
class WrapWindowDataset(Dataset):
    """
    Stride-based wrap windows for training.

    Each item:
      x [L, F], y [L, 1], m [L, 1]
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, m: np.ndarray, window: int, stride: int):
        if x.shape[0] != y.shape[0] or x.shape[0] != m.shape[0]:
            raise ValueError("x, y, m must share the same length T")

        self.x = x
        self.y = y
        self.m = m
        self.window = int(window)
        self.stride = int(stride)

        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be positive")

        self.n = max(0, (x.shape[0] - self.window) // self.stride + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        s = idx * self.stride
        e = s + self.window
        return (
            torch.from_numpy(self.x[s:e]),
            torch.from_numpy(self.y[s:e]),
            torch.from_numpy(self.m[s:e]),
        )


# =========================================================
# Eval helper
# =========================================================
def to_eval_tensors(
    x: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    device: torch.device,
):
    """
    Convert arrays to eval tensors as a single batch:
      x: [1, T, F], y: [1, T, 1], m: [1, T, 1]
    """
    xt = torch.from_numpy(x)[None, ...].to(device)
    yt = torch.from_numpy(y)[None, ...].to(device)
    mt = torch.from_numpy(m)[None, ...].to(device)
    return xt, yt, mt
