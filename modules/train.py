# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import copy
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from modules.data_processing import prepare_datasets_from_csv
from PDHTHM import build_model

from optim_decoupling import (
    DecoupledOptimizerConfig,
    setup_optimization,
    clip_gradients,
)

from loss import nse_loss

from modules.config import (
    device,
    CSV_PATH,
    OUTPUT_ROOT, LOGS_DIR, PLOTS_DIR, WEIGHTS_DIR, RESULTS_DIR, PARAMS_PLOTS_DIR,
    WRAP_LENGTH, STRIDE,
    NUM_EPOCHS, STATIC_LR, DYNAMIC_LR, WEIGHT_DECAY, CLIP_NORM,
    SCHED_FACTOR, SCHED_PATIENCE, MIN_LR,
    EARLY_STOP_PATIENCE,
    PARAM_NAMES,
    MODEL_NAME, INPUT_SIZE, TRANSFORMER_SEQ_LENGTH,
    SPLIT,
)


# =========================================================
# logging
# =========================================================
def setup_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def attach_file_logging(console_logger: logging.Logger, log_file: str, level: int = logging.INFO) -> None:
    """
    Attach a file handler to an existing logger, avoiding duplicate file handlers.
    """
    log_file = os.path.abspath(log_file)
    for h in console_logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(h.baseFilename) == log_file:
                    return
            except Exception:
                pass

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    console_logger.addHandler(fh)


# =========================================================
# filesystem helpers
# =========================================================
def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)


def ensure_dirs_strict(*dirs: str) -> None:
    for d in dirs:
        ensure_dir(d)
        if not os.path.isdir(d):
            raise RuntimeError(f"Failed to create directory: {d}")


def atomic_torch_save(state: Dict[str, Any], path: str) -> None:
    """
    Robust save for Windows: write tmp then replace.
    Uses _use_new_zipfile_serialization=False for broader compatibility.
    """
    ensure_dir_for_file(path)
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp, _use_new_zipfile_serialization=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def safe_json_dump(obj: Dict[str, Any], path: str) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# tensor/metric helpers
# =========================================================
def ensure_2d_y(y: torch.Tensor) -> torch.Tensor:
    """[B,T,1] -> [B,T]"""
    if y.ndim == 3 and y.size(-1) == 1:
        return y.squeeze(-1)
    return y


def compute_nse_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    den = np.sum((y_true - np.mean(y_true)) ** 2) + eps
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / den)


def safe_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))
    except Exception:
        return float("nan")


def as_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)


def slice_main_period(
    yhat: torch.Tensor,
    y: torch.Tensor,
    dates: np.ndarray,
    warmup_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    yhat, y: [1,T] or [B,T] (here B=1 for val/test batches in your pipeline)
    dates: [T]
    returns flattened arrays for main period.
    """
    yhat_main = yhat[:, warmup_len:].squeeze(0)
    y_main = y[:, warmup_len:].squeeze(0)
    d_main = dates[warmup_len:].flatten()
    return as_numpy_1d(yhat_main), as_numpy_1d(y_main), d_main


# =========================================================
# model forward adaptor
# =========================================================
def forward_model(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Accept:
      (yhat, params_seq) OR {"y_hat": yhat, "params": params_seq}
    Return:
      yhat: [B,T]
      params_seq: [B,T,7] or None
    """
    out = model(x)
    if isinstance(out, tuple) and len(out) == 2:
        yhat, params_seq = out
    elif isinstance(out, dict) and "y_hat" in out:
        yhat = out["y_hat"]
        params_seq = out.get("params", None)
    else:
        raise ValueError("Model output must be (outputs, params_seq) or dict with 'y_hat'.")
    yhat = ensure_2d_y(yhat)
    return yhat, params_seq


# =========================================================
# plotting
# =========================================================
def plot_timeseries(date_arr: np.ndarray, obs: np.ndarray, pred: np.ndarray, save_path: str) -> None:
    ensure_dir_for_file(save_path)
    plt.figure(figsize=(12, 6))
    plt.plot(date_arr, obs, label="Observed")
    plt.plot(date_arr, pred, label="Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Trainer
# =========================================================
class Trainer:
    def __init__(self, csv_path: str, logger: logging.Logger):
        self.csv_path = csv_path
        self.logger = logger

        stem = os.path.splitext(os.path.basename(csv_path))[0]
        self.file_stem = stem

        self.log_txt = os.path.join(LOGS_DIR, f"{stem}_train.log")
        self.weight_path = os.path.join(WEIGHTS_DIR, f"{stem}_best_weights.pth")
        self.results_csv = os.path.join(RESULTS_DIR, f"{stem}_test_results.csv")
        self.params_csv = os.path.join(RESULTS_DIR, f"{stem}_test_parameters.csv")
        self.plot_png = os.path.join(PLOTS_DIR, f"{stem}_test_plot.png")
        self.meta_json = os.path.join(OUTPUT_ROOT, f"{stem}_meta.json")

        ensure_dirs_strict(OUTPUT_ROOT, LOGS_DIR, PLOTS_DIR, WEIGHTS_DIR, RESULTS_DIR, PARAMS_PLOTS_DIR)

    def load_data(self) -> Dict[str, Any]:
        return prepare_datasets_from_csv(
            self.csv_path,
            cfg=SPLIT,
            wrap_length=WRAP_LENGTH,
            stride=STRIDE,
            device=device,
        )

    def build_model(self) -> nn.Module:
        model = build_model(
            MODEL_NAME,
            input_size=INPUT_SIZE,
            device=device,
            transformer_seq_length=TRANSFORMER_SEQ_LENGTH,
        ).to(device)
        return model

    def build_optim(self, model: nn.Module):
        optim_cfg = DecoupledOptimizerConfig(
            weight_decay=WEIGHT_DECAY,
            prefer_name_grouping=True,
            static_param_leaf_names=("SUB", "CRAK", "SQ", "COEFF", "INSC"),
            lr_static_named=STATIC_LR,
            lr_dynamic_named=DYNAMIC_LR,
            use_scheduler=True,
            sched_mode="max",
            sched_factor=SCHED_FACTOR,
            sched_patience=SCHED_PATIENCE,
            sched_min_lr=MIN_LR,
            sched_verbose=False,
            grad_clip_norm=CLIP_NORM,
        )
        optimizer, scheduler, optim_summary = setup_optimization(model, optim_cfg)
        return optim_cfg, optimizer, scheduler, optim_summary

    def train(self) -> None:
        attach_file_logging(self.logger, self.log_txt, level=logging.INFO)

        pack = self.load_data()

        train_x = pack["train_x"]
        train_y = pack["train_y"]
        val_x = pack["val_x"]
        val_y = pack["val_y"]
        test_x = pack["test_x"]
        test_y = pack["test_y"]

        warmup_val = int(pack["warmup_len_val"])
        warmup_test = int(pack["warmup_len_test"])
        test_dates = pack["test_dates"]

        model = self.build_model()
        optim_cfg, optimizer, scheduler, optim_summary = self.build_optim(model)

        meta: Dict[str, Any] = {
            "csv_path": self.csv_path,
            "model_name": MODEL_NAME,
            "seq_length": int(TRANSFORMER_SEQ_LENGTH),
            "wrap_length": int(WRAP_LENGTH),
            "stride": int(STRIDE),
            "optimizer_groups": optim_summary.get("optimizer_groups", []),
        }

        best_val_nse = -float("inf")
        best_epoch = 0
        best_model_wts = None
        early_counter = 0

        t0 = time.time()

        for epoch in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()

            yhat_train, _ = forward_model(model, train_x)
            y_train_2d = ensure_2d_y(train_y)

            loss_train = nse_loss(yhat_train, y_train_2d, warmup_len=365)
            loss_train.backward()

            clip_gradients(model, optim_cfg)
            optimizer.step()

            train_loss = float(loss_train.item())
            train_nse = float(1.0 - train_loss)

            model.eval()
            with torch.no_grad():
                yhat_val, _ = forward_model(model, val_x)
                y_val_2d = ensure_2d_y(val_y)
                loss_val_t = nse_loss(yhat_val, y_val_2d, warmup_len=warmup_val)
                val_loss = float(loss_val_t.item())
                val_nse = float(1.0 - val_loss)

            if scheduler is not None:
                scheduler.step(val_nse)

            if val_nse > best_val_nse:
                best_val_nse = val_nse
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                early_counter = 0
            else:
                early_counter += 1

            self.logger.info(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                f"TrainLoss={train_loss:.4f} TrainNSE={train_nse:.4f} | "
                f"ValLoss={val_loss:.4f} ValNSE={val_nse:.4f}"
            )

            if early_counter >= EARLY_STOP_PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        meta["best_val_nse"] = float(best_val_nse)
        meta["best_epoch"] = int(best_epoch)
        meta["train_seconds"] = float(time.time() - t0)

        if best_model_wts is None:
            self.logger.warning("No best model weights found; skip saving/testing.")
            safe_json_dump(meta, self.meta_json)
            return

        model.load_state_dict(best_model_wts)
        atomic_torch_save(best_model_wts, self.weight_path)
        self.logger.info(f"Saved best weights: {os.path.abspath(self.weight_path)}")

        model.eval()
        with torch.no_grad():
            yhat_test, params_seq = forward_model(model, test_x)
            y_test_2d = ensure_2d_y(test_y)

        pred_main, obs_main, dates_main = slice_main_period(yhat_test, y_test_2d, test_dates, warmup_test)

        test_nse = compute_nse_np(obs_main, pred_main)
        test_r2 = safe_r2_np(obs_main, pred_main)

        meta["test_nse"] = float(test_nse)
        meta["test_r2"] = float(test_r2)

        self.logger.info(f"Test NSE={test_nse:.4f} | R2={test_r2:.4f}")

        ensure_dir_for_file(self.results_csv)
        pd.DataFrame({"Date": dates_main, "Observed": obs_main, "Predicted": pred_main}).to_csv(self.results_csv, index=False)

        if params_seq is not None:
            params_main = params_seq[:, warmup_test:, :].squeeze(0).detach().cpu().numpy()  # [T,7]
            ensure_dir_for_file(self.params_csv)
            dfp = pd.DataFrame(params_main.reshape(-1, len(PARAM_NAMES)), columns=PARAM_NAMES)
            dfp["Date"] = dates_main
            dfp.to_csv(self.params_csv, index=False)

        plot_timeseries(dates_main, obs_main, pred_main, self.plot_png)
        safe_json_dump(meta, self.meta_json)

        self.logger.info(f"DONE. Outputs under: {os.path.abspath(OUTPUT_ROOT)}")


def main():
    logger = setup_logger("train_console", level=logging.INFO, log_file=None)
    logger.info(f"RUNNING FILE: {os.path.abspath(__file__)}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    Trainer(CSV_PATH, logger).train()


if __name__ == "__main__":
    main()