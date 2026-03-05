# analysis/shap_hypernet.py
# -*- coding: utf-8 -*-
"""
SHAP analysis for a hypernetwork (PyTorch) that maps meteorological windows -> model parameters.

Design
------
- This module does NOT define the model architecture.
- You pass in a trained PyTorch module (e.g., model.hyper_net) and data windows.
- Uses shap.GradientExplainer for gradient-based SHAP values.

Outputs
-------
- SHAP summary plot (tif, optional)
- Top-k feature importance CSV per output dimension (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import shap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =========================
# Config
# =========================
@dataclass
class ShapConfig:
    """Configuration for SHAP computation and plotting."""
    window_length: int
    input_size: int = 3                       # e.g., [T, P, PET]
    feature_labels: Sequence[str] = ("T", "P", "PET")
    reverse_time_labels: bool = True          # match your current naming order
    background_size: Optional[int] = None     # None -> use all as background
    explain_size: Optional[int] = None        # None -> explain all
    max_display: int = 10                     # top-k features in plots/CSVs
    dpi: int = 600
    save_tif: bool = True
    tiff_lzw: bool = True
    font_family: str = "Times New Roman"
    x_ticks_max: int = 6                      # max x-ticks for summary plot
    plot_alpha: float = 0.8


# =========================
# Utilities: windows & names
# =========================
def create_sliding_windows(features_2d: np.ndarray, window_length: int) -> np.ndarray:
    """
    Create sliding windows from a 2D feature array.

    Parameters
    ----------
    features_2d : np.ndarray
        Shape [T, F].
    window_length : int
        Window length.

    Returns
    -------
    np.ndarray
        Shape [N, window_length, F], where N = T - window_length + 1
    """
    features_2d = np.asarray(features_2d, dtype=np.float32)
    if features_2d.ndim != 2:
        raise ValueError("features_2d must be a 2D array of shape [T, F].")
    T, F = features_2d.shape
    if T < window_length:
        raise ValueError(f"Not enough timesteps T={T} for window_length={window_length}.")
    n = T - window_length + 1
    out = np.empty((n, window_length, F), dtype=np.float32)
    for i in range(n):
        out[i] = features_2d[i:i + window_length]
    return out


def make_feature_names(cfg: ShapConfig) -> List[str]:
    """
    Generate feature names for flattened window inputs.
    Example: ["T30","P30","PET30", ..., "T1","P1","PET1"] if reverse_time_labels=True.
    """
    if len(cfg.feature_labels) != cfg.input_size:
        raise ValueError("len(feature_labels) must match input_size.")

    # time indices: 1..L
    time_ids = list(range(1, cfg.window_length + 1))
    if cfg.reverse_time_labels:
        time_ids = list(reversed(time_ids))

    names: List[str] = []
    for t in time_ids:
        for feat in cfg.feature_labels:
            names.append(f"{feat}{t}")
    return names


# =========================
# Wrapper: flatten -> window -> hypernet outputs
# =========================
class WindowFlattenWrapper(nn.Module):
    """
    Wrapper for SHAP GradientExplainer.

    Input : [B, window_length*input_size] (flattened)
    Output: hyper_net(window) -> [B, D] (D = output dim of hypernetwork)
    """
    def __init__(self, hyper_net: nn.Module, window_length: int, input_size: int):
        super().__init__()
        self.hyper_net = hyper_net
        self.window_length = int(window_length)
        self.input_size = int(input_size)

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        if flat_input.ndim != 2:
            raise ValueError("flat_input must have shape [B, window_length*input_size].")
        B, K = flat_input.shape
        expected = self.window_length * self.input_size
        if K != expected:
            raise ValueError(f"Expected input dim {expected}, got {K}.")
        x = flat_input.view(B, self.window_length, self.input_size)
        return self.hyper_net(x)


# =========================
# SHAP core
# =========================
def compute_shap_values(
    hyper_net: nn.Module,
    windows: np.ndarray,
    cfg: ShapConfig,
    device: Optional[torch.device] = None,
) -> Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray, List[str]]:
    """
    Compute SHAP values for hyper_net using GradientExplainer.

    Parameters
    ----------
    hyper_net : nn.Module
        Trained hypernetwork module.
    windows : np.ndarray
        Shape [N, L, F].
    cfg : ShapConfig
        SHAP configuration.
    device : torch.device, optional
        Device to run on.

    Returns
    -------
    shap_values : np.ndarray or list[np.ndarray]
        SHAP values returned by shap.GradientExplainer.
        Often a list: one array per output dimension.
    explain_flat : np.ndarray
        The flattened explain data used for SHAP, shape [Ne, L*F].
    feature_names : list[str]
        Flattened feature names aligned with explain_flat columns.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError("windows must be a 3D array [N, L, F].")
    N, L, F = windows.shape
    if L != cfg.window_length or F != cfg.input_size:
        raise ValueError(f"windows shape mismatch: got [N,{L},{F}], expected [N,{cfg.window_length},{cfg.input_size}].")

    # flatten windows: [N, L*F]
    flat = windows.reshape(N, -1)

    # select background / explain subsets
    bg = flat if cfg.background_size is None else flat[: int(cfg.background_size)]
    ex = flat if cfg.explain_size is None else flat[: int(cfg.explain_size)]

    # to torch
    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)
    ex_t = torch.tensor(ex, dtype=torch.float32, device=device)

    hyper_net = hyper_net.to(device)
    hyper_net.eval()

    wrapper = WindowFlattenWrapper(hyper_net, cfg.window_length, cfg.input_size).to(device)
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, bg_t)
    shap_values = explainer.shap_values(ex_t)

    feature_names = make_feature_names(cfg)
    return shap_values, ex, feature_names


# =========================
# Plot & save
# =========================
def save_summary_plots(
    shap_values: Union[np.ndarray, List[np.ndarray]],
    explain_flat: np.ndarray,
    feature_names: List[str],
    save_dir: str,
    cfg: ShapConfig,
    *,
    prefix: str = "shap_summary",
) -> None:
    """
    Save SHAP summary plots (dot) for each output dimension (if list).

    Parameters
    ----------
    shap_values : np.ndarray or list[np.ndarray]
        From compute_shap_values.
    explain_flat : np.ndarray
        Flattened explain samples, shape [Ne, n_features].
    feature_names : list[str]
        Names of flattened features.
    save_dir : str
        Output directory.
    cfg : ShapConfig
        Plot configuration.
    prefix : str
        Filename prefix.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams["font.family"] = cfg.font_family

    def _save_one(sv: np.ndarray, tag: str):
        fig, ax = plt.subplots(figsize=(10, 6))

        shap.summary_plot(
            sv,
            features=explain_flat,
            feature_names=feature_names,
            max_display=cfg.max_display,
            show=False,
            plot_type="dot",
            alpha=cfg.plot_alpha,
            auto_size_plot=False
        )

        axes = fig.get_axes()
        if len(axes) > 0:
            main_ax = axes[0]
            main_ax.set_xlabel("SHAP Value", fontsize=20)
            main_ax.xaxis.set_major_locator(MaxNLocator(cfg.x_ticks_max))

        if len(axes) > 1:
            # usually the colorbar axis
            axes[1].set_xlabel("")

        for ax_ in axes:
            for item in ([ax_.title, ax_.xaxis.label, ax_.yaxis.label] +
                         ax_.get_xticklabels() + ax_.get_yticklabels()):
                item.set_fontsize(18)

        fig.tight_layout()

        if cfg.save_tif:
            out = os.path.join(save_dir, f"{prefix}_{tag}.tif")
            pil_kwargs = {"compression": "tiff_lzw"} if cfg.tiff_lzw else None
            fig.savefig(out, format="tif", dpi=cfg.dpi, pil_kwargs=pil_kwargs)
        else:
            out = os.path.join(save_dir, f"{prefix}_{tag}.png")
            fig.savefig(out, dpi=cfg.dpi)

        plt.close(fig)

    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            _save_one(np.asarray(sv), f"out{i}")
    else:
        _save_one(np.asarray(shap_values), "out0")


def save_topk_importance_csv(
    shap_values: Union[np.ndarray, List[np.ndarray]],
    feature_names: List[str],
    save_dir: str,
    cfg: ShapConfig,
    *,
    prefix: str = "feature_importance",
) -> None:
    """
    Save top-k mean(|SHAP|) and ratio to CSV for each output dimension.

    Parameters
    ----------
    shap_values : np.ndarray or list[np.ndarray]
        From compute_shap_values.
    feature_names : list[str]
        Names of flattened features.
    save_dir : str
        Output directory.
    cfg : ShapConfig
        Controls top-k.
    prefix : str
        Filename prefix.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    def _save_one(sv: np.ndarray, tag: str):
        sv = np.asarray(sv)
        mean_abs = np.mean(np.abs(sv), axis=0)  # [n_features]
        total = float(np.sum(mean_abs)) + 1e-12
        ratio = mean_abs / total

        topk = min(cfg.max_display, mean_abs.shape[0])
        idx = np.argsort(mean_abs)[::-1][:topk]

        df = pd.DataFrame({
            "feature": np.array(feature_names)[idx],
            "mean_abs_shap": mean_abs[idx],
            "importance_ratio": ratio[idx],
        })

        out_csv = os.path.join(save_dir, f"{prefix}_{tag}.csv")
        df.to_csv(out_csv, index=False)

    import pandas as pd  # local import to keep module clean if only plotting is used

    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            _save_one(np.asarray(sv), f"out{i}")
    else:
        _save_one(np.asarray(shap_values), "out0")
