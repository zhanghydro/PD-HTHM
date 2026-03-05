# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Tuple
import torch
from modules.data_processing import SplitConfig


# -------------------------
# small parsers (env -> type)
# -------------------------
def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v == "" else v

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    return default if v is None or v == "" else int(v)

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    return default if v is None or v == "" else float(v)

def _env_tuple3(key: str, default: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Example env: X_COLS="temperature,precipitation,pet"
    """
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"{key} must have 3 comma-separated items, got: {v}")
    return (parts[0], parts[1], parts[2])


# -------------------------
# main config builder
# -------------------------
def get_config() -> dict:
    # DEVICE
    device_str = _env_str("DEVICE", "cuda")
    if device_str.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA / PATHS
    DATA_DIR = _env_str("DATA_DIR", "/data")
    CSV_FILE = _env_str("CSV_FILE", "*.csv")
    CSV_PATH = os.path.join(DATA_DIR, CSV_FILE)

    OUTPUT_ROOT = _env_str("OUTPUT_ROOT", os.path.join(DATA_DIR, "output"))
    LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")
    PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
    WEIGHTS_DIR = os.path.join(OUTPUT_ROOT, "weights")
    RESULTS_DIR = os.path.join(OUTPUT_ROOT, "test_results")
    PARAMS_PLOTS_DIR = os.path.join(PLOTS_DIR, "parameters")

    # WRAP TRAINING
    WRAP_LENGTH = _env_int("WRAP_LENGTH", 2190)
    STRIDE = _env_int("STRIDE", 365)

    # TRAINING HYPERPARAMETERS
    NUM_EPOCHS = _env_int("NUM_EPOCHS", 300)
    STATIC_LR = _env_float("STATIC_LR", 0.01)
    DYNAMIC_LR = _env_float("DYNAMIC_LR", 1e-3)
    WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 1e-5)
    CLIP_NORM = _env_float("CLIP_NORM", 1.0)

    # scheduler
    SCHED_FACTOR = _env_float("SCHED_FACTOR", 0.8)
    SCHED_PATIENCE = _env_int("SCHED_PATIENCE", 5)
    MIN_LR = _env_float("MIN_LR", 1e-5)

    # early stopping
    EARLY_STOP_PATIENCE = _env_int("EARLY_STOP_PATIENCE", 10)

    # MODEL SETTINGS
    PARAM_NAMES = ["SMSC", "K", "SUB", "CRAK", "SQ", "COEFF", "INSC"]
    MODEL_NAME = _env_str("MODEL_NAME", "simhyd_hyper_tf")
    INPUT_SIZE = _env_int("INPUT_SIZE", 3)
    TRANSFORMER_SEQ_LENGTH = _env_int("SEQ_LENGTH", 30)

    # SPLIT CONFIG (warmup+main) - allow override via env if needed
    SPLIT = SplitConfig(
        train_start=_env_str("TRAIN_START", "1971-01-01"),
        train_end=_env_str("TRAIN_END", "1999-12-31"),

        val_warmup_start=_env_str("VAL_WARMUP_START", "2000-01-01"),
        val_warmup_end=_env_str("VAL_WARMUP_END", "2000-12-31"),
        val_main_start=_env_str("VAL_MAIN_START", "2001-01-01"),
        val_main_end=_env_str("VAL_MAIN_END", "2007-12-31"),

        test_warmup_start=_env_str("TEST_WARMUP_START", "2008-01-01"),
        test_warmup_end=_env_str("TEST_WARMUP_END", "2008-12-31"),
        test_main_start=_env_str("TEST_MAIN_START", "2009-01-01"),
        test_main_end=_env_str("TEST_MAIN_END", "2014-12-31"),

        date_col=_env_str("DATE_COL", "date"),
        x_cols=_env_tuple3("X_COLS", ("temperature", "precipitation", "pet")),
        y_col=_env_str("Y_COL", "discharge_spec"),
    )

    return dict(
        device=device,

        DATA_DIR=DATA_DIR,
        CSV_FILE=CSV_FILE,
        CSV_PATH=CSV_PATH,

        OUTPUT_ROOT=OUTPUT_ROOT,
        LOGS_DIR=LOGS_DIR,
        PLOTS_DIR=PLOTS_DIR,
        WEIGHTS_DIR=WEIGHTS_DIR,
        RESULTS_DIR=RESULTS_DIR,
        PARAMS_PLOTS_DIR=PARAMS_PLOTS_DIR,

        WRAP_LENGTH=WRAP_LENGTH,
        STRIDE=STRIDE,

        NUM_EPOCHS=NUM_EPOCHS,
        STATIC_LR=STATIC_LR,
        DYNAMIC_LR=DYNAMIC_LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        CLIP_NORM=CLIP_NORM,

        SCHED_FACTOR=SCHED_FACTOR,
        SCHED_PATIENCE=SCHED_PATIENCE,
        MIN_LR=MIN_LR,

        EARLY_STOP_PATIENCE=EARLY_STOP_PATIENCE,

        PARAM_NAMES=PARAM_NAMES,
        MODEL_NAME=MODEL_NAME,
        INPUT_SIZE=INPUT_SIZE,
        TRANSFORMER_SEQ_LENGTH=TRANSFORMER_SEQ_LENGTH,

        SPLIT=SPLIT,
    )

CFG = get_config()

device = CFG["device"]
DATA_DIR = CFG["DATA_DIR"]
CSV_FILE = CFG["CSV_FILE"]
CSV_PATH = CFG["CSV_PATH"]

OUTPUT_ROOT = CFG["OUTPUT_ROOT"]
LOGS_DIR = CFG["LOGS_DIR"]
PLOTS_DIR = CFG["PLOTS_DIR"]
WEIGHTS_DIR = CFG["WEIGHTS_DIR"]
RESULTS_DIR = CFG["RESULTS_DIR"]
PARAMS_PLOTS_DIR = CFG["PARAMS_PLOTS_DIR"]

WRAP_LENGTH = CFG["WRAP_LENGTH"]
STRIDE = CFG["STRIDE"]

NUM_EPOCHS = CFG["NUM_EPOCHS"]
STATIC_LR = CFG["STATIC_LR"]
DYNAMIC_LR = CFG["DYNAMIC_LR"]
WEIGHT_DECAY = CFG["WEIGHT_DECAY"]
CLIP_NORM = CFG["CLIP_NORM"]

SCHED_FACTOR = CFG["SCHED_FACTOR"]
SCHED_PATIENCE = CFG["SCHED_PATIENCE"]
MIN_LR = CFG["MIN_LR"]

EARLY_STOP_PATIENCE = CFG["EARLY_STOP_PATIENCE"]

PARAM_NAMES = CFG["PARAM_NAMES"]
MODEL_NAME = CFG["MODEL_NAME"]
INPUT_SIZE = CFG["INPUT_SIZE"]
TRANSFORMER_SEQ_LENGTH = CFG["TRANSFORMER_SEQ_LENGTH"]

SPLIT = CFG["SPLIT"]