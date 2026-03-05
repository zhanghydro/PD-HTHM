# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import time
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_processing import make_loaders
from model import build_model


# DATA_DIR = "/data"
# CSV_GLOB = "CAMELS_GB_hydromet_timeseries_*.csv"

DATA_DIR = r"E:\zhanghongwei\投稿\上传\Severn_data"
CSV_GLOB = "CAMELS_GB_hydromet_timeseries_53005_19701001-20150930.csv"   # 只会选第一个匹配文件

MODEL_NAME = "gru"          # "lstm" / "gru" / "transformer_ed"
SEQ_LENGTH = 30

X_COLS = ["precipitation", "pet", "temperature"]
Y_COL = "discharge_spec"

TRAIN_PRED = ("1971-01-01", "1999-12-31")
VAL_PRED   = ("2000-01-01", "2007-12-31")
TEST_PRED  = ("2009-01-01", "2014-12-31")

EPOCHS = 100
LR = 1e-3
PATIENCE = 10
MIN_DELTA = 1e-4
BATCH_TRAIN = 256
BATCH_EVAL = 1024

SEED = 123
SAVE_DIR = "runs_output"


def get_window_range(start_pred: str, end_pred: str, seq_length: int):
    """Extend start to include (seq_length-1) days for windowing."""
    start = pd.to_datetime(start_pred) - pd.Timedelta(days=seq_length - 1)
    end = pd.to_datetime(end_pred)
    return start, end


def nse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    den = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / den)


def nse_loss_torch(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """loss = 1 - NSE (batch)."""
    num = torch.sum((y - y_hat) ** 2)
    den = torch.sum((y - torch.mean(y)) ** 2) + eps
    nse = 1.0 - num / den
    return 1.0 - nse


@torch.no_grad()
def eval_nse(model: nn.Module, loader, device, scaler) -> float:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)["y_hat"]
        ys.append(y.detach().cpu().numpy())
        ps.append(y_hat.detach().cpu().numpy())

    y_norm = np.concatenate(ys, axis=0).reshape(-1)
    p_norm = np.concatenate(ps, axis=0).reshape(-1)
    y_true = scaler.inverse_y(y_norm)
    y_pred = scaler.inverse_y(p_norm)
    return nse_np(y_true, y_pred)


def train_one(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    scaler,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1e18
    best_state = None
    best_epoch = 0
    no_improve = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_iter = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)["y_hat"]
            loss = nse_loss_torch(y_hat, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            n_iter += 1

        loss_ep = loss_sum / max(1, n_iter)
        val_nse = eval_nse(model, val_loader, device, scaler)

        if val_nse > best_val + min_delta:
            best_val = float(val_nse)
            best_epoch = int(ep)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best_val={best_val:.4f} @ {best_epoch})")
                break

        if ep == 1 or ep % 10 == 0:
            print(
                f"Epoch {ep:03d} | loss(1-NSE)={loss_ep:.4f} | "
                f"val_NSE={val_nse:.4f} | time={time.time()-t0:.2f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def pick_one_csv() -> str:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No file matched: {os.path.join(DATA_DIR, CSV_GLOB)}")
    return paths[0]


def infer_basin_id(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    parts = base.split("_")
    return parts[4] if len(parts) > 4 else os.path.splitext(base)[0]


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    needed = set(X_COLS + [Y_COL])
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns {miss}. Got columns={list(df.columns)}")

    tr_s, tr_e = get_window_range(TRAIN_PRED[0], TRAIN_PRED[1], SEQ_LENGTH)
    va_s, va_e = get_window_range(VAL_PRED[0], VAL_PRED[1], SEQ_LENGTH)
    te_s, te_e = get_window_range(TEST_PRED[0], TEST_PRED[1], SEQ_LENGTH)

    df_tr = df.loc[tr_s:tr_e, X_COLS + [Y_COL]].dropna()
    df_va = df.loc[va_s:va_e, X_COLS + [Y_COL]].dropna()
    df_te = df.loc[te_s:te_e, X_COLS + [Y_COL]].dropna()

    if len(df_tr) < SEQ_LENGTH or len(df_va) < SEQ_LENGTH or len(df_te) < SEQ_LENGTH:
        raise ValueError(
            f"Not enough valid days after split. "
            f"train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}, seq_length={SEQ_LENGTH}"
        )

    X_train = df_tr[X_COLS].values.astype(np.float32)
    y_train = df_tr[Y_COL].values.astype(np.float32)

    X_val = df_va[X_COLS].values.astype(np.float32)
    y_val = df_va[Y_COL].values.astype(np.float32)

    X_test = df_te[X_COLS].values.astype(np.float32)
    y_test = df_te[Y_COL].values.astype(np.float32)

    return (X_train, y_train, X_val, y_val, X_test, y_test, df_tr, df_va, df_te)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    csv_path = pick_one_csv()
    basin_id = infer_basin_id(csv_path)
    print("Using CSV:", csv_path)
    print("Basin ID:", basin_id)

    X_train, y_train, X_val, y_val, X_test, y_test, df_tr, df_va, df_te = load_and_prepare(csv_path)
    print(f"Split sizes (days): train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")

    train_loader, val_loader, test_loader, scaler = make_loaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        seq_length=SEQ_LENGTH,
        batch_train=BATCH_TRAIN,
        batch_eval=BATCH_EVAL,
    )

    if MODEL_NAME in ["lstm", "gru"]:
        model_kwargs = dict(hidden_size=64, num_layers=1, dropout=0.3)
    else:
        model_kwargs = dict(
            d_model=128, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=256, dropout=0.1
        )

    model = build_model(MODEL_NAME, input_size=len(X_COLS), **model_kwargs).to(device)
    print(f"Model: {MODEL_NAME} | seq_length={SEQ_LENGTH} | input_size={len(X_COLS)}")

    best_val, best_epoch, best_state = train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaler=scaler,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
    )

    test_nse = eval_nse(model, test_loader, device, scaler)
    print(f"\nDONE | best_val_NSE={best_val:.4f} @ epoch {best_epoch} | test_NSE={test_nse:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    outdir = os.path.join(SAVE_DIR, f"{basin_id}_{MODEL_NAME}_T{SEQ_LENGTH}")
    os.makedirs(outdir, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(outdir, "best_model_state.pt"))

    meta = {
        "csv_path": csv_path,
        "basin_id": basin_id,
        "model": MODEL_NAME,
        "seq_length": SEQ_LENGTH,
        "train_pred": TRAIN_PRED,
        "val_pred": VAL_PRED,
        "test_pred": TEST_PRED,
        "best_val_nse": float(best_val),
        "best_epoch": int(best_epoch),
        "test_nse": float(test_nse),
        "x_cols": X_COLS,
        "y_col": Y_COL,
        "seed": SEED,
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved to:", outdir)


if __name__ == "__main__":
    main()