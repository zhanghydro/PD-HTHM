# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch


@dataclass
class SplitConfig:
    # ----------------- TRAIN -----------------
    train_start: str = "1971-01-01"
    train_end: str = "1999-12-31"

    # ----------------- VAL (warmup + main) ---
    val_warmup_start: str = "2000-01-01"
    val_warmup_end: str = "2000-12-31"
    val_main_start: str = "2001-01-01"
    val_main_end: str = "2007-12-31"

    # ----------------- TEST (warmup + main) --
    test_warmup_start: str = "2008-01-01"
    test_warmup_end: str = "2008-12-31"
    test_main_start: str = "2009-01-01"
    test_main_end: str = "2014-12-31"

    date_col: str = "date"
    x_cols: Tuple[str, str, str] = ("temperature", "precipitation", "pet")
    y_col: str = "discharge_spec"


# =========================================================
# I/O and checks
# =========================================================
def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def _require_columns(df: pd.DataFrame, cols: Tuple[str, ...], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        head = f"{where}: " if where else ""
        raise ValueError(f"{head}Missing columns: {missing}. Existing: {list(df.columns)}")


def _to_datetime_sorted(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    return df


def read_camels_csv(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    _require_file(csv_path)
    df = pd.read_csv(csv_path)
    _require_columns(df, (date_col,), where="read_camels_csv")
    df = _to_datetime_sorted(df, date_col=date_col)
    return df


# =========================================================
# splitting
# =========================================================
def _slice_inclusive(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return df[(df[date_col] >= s) & (df[date_col] <= e)].copy()


def split_train_val_test_with_warmup(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    Returns:
      train_df: [train_start, train_end]
      val_df:   [val_warmup + val_main] concatenated
      test_df:  [test_warmup + test_main] concatenated
      warmup_len_val: warmup rows inside val_df
      warmup_len_test: warmup rows inside test_df
    """
    d = df.copy()

    train_df = _slice_inclusive(d, cfg.date_col, cfg.train_start, cfg.train_end)

    val_warmup = _slice_inclusive(d, cfg.date_col, cfg.val_warmup_start, cfg.val_warmup_end)
    val_main = _slice_inclusive(d, cfg.date_col, cfg.val_main_start, cfg.val_main_end)
    val_df = pd.concat([val_warmup, val_main], ignore_index=True)
    warmup_len_val = int(len(val_warmup))

    test_warmup = _slice_inclusive(d, cfg.date_col, cfg.test_warmup_start, cfg.test_warmup_end)
    test_main = _slice_inclusive(d, cfg.date_col, cfg.test_main_start, cfg.test_main_end)
    test_df = pd.concat([test_warmup, test_main], ignore_index=True)
    warmup_len_test = int(len(test_warmup))

    if train_df.empty:
        raise ValueError("Train period is empty after splitting.")
    if val_main.empty:
        raise ValueError("Val main period is empty after splitting.")
    if test_main.empty:
        raise ValueError("Test main period is empty after splitting.")

    return train_df, val_df, test_df, warmup_len_val, warmup_len_test


# =========================================================
# feature extraction
# =========================================================
def extract_xy(df: pd.DataFrame, x_cols: Tuple[str, ...], y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    _require_columns(df, tuple(x_cols) + (y_col,), where="extract_xy")

    X = df.loc[:, list(x_cols)].to_numpy(dtype=np.float32)
    y = df.loc[:, [y_col]].to_numpy(dtype=np.float32)  # keep [N,1]
    return X, y


# =========================================================
# window building
# =========================================================
def _compute_num_wraps(n_days: int, wrap_length: int, stride: int) -> int:
    if n_days < wrap_length:
        raise ValueError(f"Not enough days for wrap_length={wrap_length}. N={n_days}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return (n_days - wrap_length) // stride + 1


def _fill_wrap_arrays(
    X: np.ndarray,
    y: np.ndarray,
    wrap_length: int,
    stride: int,
    Xw: np.ndarray,
    yw: np.ndarray,
) -> None:
    n_wrap = Xw.shape[0]
    for i in range(n_wrap):
        s = i * stride
        e = s + wrap_length
        Xw[i, :, :] = X[s:e, :]
        yw[i, :, :] = y[s:e, :]


def build_train_wrap_windows(
    X: np.ndarray,
    y: np.ndarray,
    wrap_length: int = 2190,
    stride: int = 365,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected X,y as 2D arrays. Got X{X.shape}, y{y.shape}")
    if y.shape[1] != 1:
        raise ValueError(f"y must be [N,1]. Got y{y.shape}")

    n_wrap = _compute_num_wraps(X.shape[0], wrap_length, stride)

    Xw = np.empty((n_wrap, wrap_length, X.shape[1]), dtype=np.float32)
    yw = np.empty((n_wrap, wrap_length, y.shape[1]), dtype=np.float32)

    _fill_wrap_arrays(X, y, wrap_length, stride, Xw, yw)
    return Xw, yw


def build_full_sequence_batch(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_seq = np.expand_dims(X, axis=0).astype(np.float32)
    y_seq = np.expand_dims(y, axis=0).astype(np.float32)
    return X_seq, y_seq


# =========================================================
# torch conversion
# =========================================================
def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(X: np.ndarray, y: np.ndarray, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = _resolve_device(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    return Xt, yt


# =========================================================
# pipeline
# =========================================================
def prepare_datasets_from_csv(
    csv_path: str,
    cfg: Optional[SplitConfig] = None,
    wrap_length: int = 2190,
    stride: int = 365,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = SplitConfig()

    df = read_camels_csv(csv_path, date_col=cfg.date_col)

    # drop rows with any NaN in required columns
    required_cols = (cfg.date_col,) + tuple(cfg.x_cols) + (cfg.y_col,)
    _require_columns(df, required_cols, where="prepare_datasets_from_csv")
    df = df.dropna(subset=list(required_cols))

    train_df, val_df, test_df, warmup_len_val, warmup_len_test = split_train_val_test_with_warmup(df, cfg)

    X_train, y_train = extract_xy(train_df, cfg.x_cols, cfg.y_col)
    X_val, y_val = extract_xy(val_df, cfg.x_cols, cfg.y_col)
    X_test, y_test = extract_xy(test_df, cfg.x_cols, cfg.y_col)

    train_x_np, train_y_np = build_train_wrap_windows(X_train, y_train, wrap_length=wrap_length, stride=stride)
    val_x_np, val_y_np = build_full_sequence_batch(X_val, y_val)
    test_x_np, test_y_np = build_full_sequence_batch(X_test, y_test)

    train_x, train_y = to_torch(train_x_np, train_y_np, device=device)
    val_x, val_y = to_torch(val_x_np, val_y_np, device=device)
    test_x, test_y = to_torch(test_x_np, test_y_np, device=device)

    val_dates = val_df[cfg.date_col].to_numpy()
    test_dates = test_df[cfg.date_col].to_numpy()

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "warmup_len_val": warmup_len_val,
        "warmup_len_test": warmup_len_test,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }