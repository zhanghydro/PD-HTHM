# -*- coding: utf-8 -*-
"""
Morris (Elementary Effects) sensitivity analysis utilities.

This module treats the target model as a black box: model_fn(x) -> scalar.
It provides:
- problem specification validation
- Morris sampling
- robust evaluation (nan/inf handling + optional parallelism)
- SALib Morris analysis (mu*, sigma, mu, and optional confidence intervals)
- tidy results as pandas DataFrame
- optional helpers for saving and plotting

Dependencies:
  numpy, pandas, SALib
Optional:
  joblib (parallel evaluation), matplotlib (plotting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze


# =========================================================
# Configuration
# =========================================================
@dataclass(frozen=True)
class MorrisConfig:
    """Configuration for Morris sampling and analysis."""
    N: int = 200
    num_levels: int = 10
    grid_jump: int = 2
    seed: int = 42

    # analysis bootstrap (for confidence intervals)
    num_resamples: int = 1000
    conf_level: float = 0.95

    # evaluation
    n_jobs: int = 1

    # bad-output handling
    penalty: Optional[float] = None  # if None -> auto penalty from finite outputs
    fail_on_nan: bool = False        # if True -> raise if any NaN/Inf appears


# =========================================================
# Problem specification
# =========================================================
def build_problem(
    names: Sequence[str],
    bounds: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """
    Build SALib 'problem' dict.

    names: list of parameter names, length D
    bounds: list of [lb, ub], length D
    """
    _validate_problem(names, bounds)
    return {
        "num_vars": len(names),
        "names": list(names),
        "bounds": [list(b) for b in bounds],
    }


def _validate_problem(
    names: Sequence[str],
    bounds: Sequence[Sequence[float]],
) -> None:
    if not isinstance(names, (list, tuple)) or not isinstance(bounds, (list, tuple)):
        raise TypeError("names and bounds must be sequences (list/tuple)")

    if len(names) != len(bounds):
        raise ValueError("names and bounds must have the same length")

    if len(names) < 2:
        raise ValueError("need at least 2 parameters")

    # names checks
    seen = set()
    for n in names:
        if not isinstance(n, str) or not n.strip():
            raise ValueError("all parameter names must be non-empty strings")
        if n in seen:
            raise ValueError(f"duplicate parameter name: {n}")
        seen.add(n)

    # bounds checks
    for i, b in enumerate(bounds):
        if not isinstance(b, (list, tuple)) or len(b) != 2:
            raise ValueError(f"bounds[{i}] must be [lb, ub]")
        lb, ub = float(b[0]), float(b[1])
        if not np.isfinite(lb) or not np.isfinite(ub):
            raise ValueError(f"bounds[{i}] must be finite numbers")
        if ub <= lb:
            raise ValueError(f"bounds[{i}] must satisfy ub > lb (got {lb}, {ub})")


# =========================================================
# Sampling
# =========================================================
def sample_morris(
    problem: Dict[str, Any],
    cfg: MorrisConfig,
) -> np.ndarray:
    """
    Generate Morris samples X with shape (n_samples, D).
    """
    X = morris_sample(
        problem,
        N=cfg.N,
        num_levels=cfg.num_levels,
        grid_jump=cfg.grid_jump,
        seed=cfg.seed,
        optimal_trajectories=None,
        local_optimization=False,
    )
    return np.asarray(X, dtype=np.float64)


def expected_evaluations(problem: Dict[str, Any], N: int) -> int:
    """
    Morris typically evaluates about N*(D+1) points.
    """
    D = int(problem["num_vars"])
    return int(N) * (D + 1)


# =========================================================
# Evaluation
# =========================================================
def evaluate_blackbox(
    X: np.ndarray,
    model_fn: Callable[[np.ndarray], Union[float, int]],
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Evaluate model_fn(x) for each row x in X.
    Returns y shape (n_samples,).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, D)")

    if n_jobs is None or n_jobs <= 1:
        y = []
        for x in X:
            y.append(_safe_float(model_fn(x)))
        return np.asarray(y, dtype=np.float64)

    # optional parallel via joblib
    try:
        from joblib import Parallel, delayed
        y = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_safe_float)(model_fn(x)) for x in X
        )
        return np.asarray(y, dtype=np.float64)
    except Exception:
        y = []
        for x in X:
            y.append(_safe_float(model_fn(x)))
        return np.asarray(y, dtype=np.float64)


def _safe_float(v: Union[float, int, np.number]) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def sanitize_outputs(
    y: np.ndarray,
    penalty: Optional[float] = None,
    fail_on_nan: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ensure y is finite for SALib. Replace NaN/Inf with penalty.
    Returns (y_clean, info_dict).
    """
    y = np.asarray(y, dtype=np.float64)
    bad = ~np.isfinite(y)
    info = {
        "n_total": int(y.size),
        "n_bad": int(bad.sum()),
        "penalty": None,
    }

    if not bad.any():
        return y, info

    if fail_on_nan:
        raise ValueError("NaN/Inf encountered in model outputs")

    finite = y[~bad]
    if finite.size == 0:
        raise ValueError("All model outputs are NaN/Inf. Check model_fn or bounds.")

    if penalty is None:
        # conservative "worse than worst" penalty
        penalty = np.nanmax(finite) + 10.0 * (np.nanstd(finite) + 1.0)

    y = y.copy()
    y[bad] = float(penalty)
    info["penalty"] = float(penalty)
    return y, info


# =========================================================
# Analysis + tidy results
# =========================================================
def morris_sensitivity(
    model_fn: Callable[[np.ndarray], Union[float, int]],
    names: Sequence[str],
    bounds: Sequence[Sequence[float]],
    cfg: Optional[MorrisConfig] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Run Morris sensitivity analysis for a black-box model.

    Returns:
      Si: SALib result dict
      df: tidy DataFrame sorted by mu_star desc
      meta: dict with sampling/evaluation stats
    """
    if cfg is None:
        cfg = MorrisConfig()

    problem = build_problem(names, bounds)
    X = sample_morris(problem, cfg)

    y_raw = evaluate_blackbox(X, model_fn, n_jobs=cfg.n_jobs)
    y, y_info = sanitize_outputs(y_raw, penalty=cfg.penalty, fail_on_nan=cfg.fail_on_nan)

    Si = morris_analyze(
        problem,
        X,
        y,
        num_levels=cfg.num_levels,
        print_to_console=False,
        num_resamples=cfg.num_resamples,
        conf_level=cfg.conf_level,
        seed=cfg.seed,
    )

    df = results_to_dataframe(Si).sort_values("mu_star", ascending=False).reset_index(drop=True)

    meta = {
        "num_vars": int(problem["num_vars"]),
        "N": int(cfg.N),
        "num_levels": int(cfg.num_levels),
        "grid_jump": int(cfg.grid_jump),
        "seed": int(cfg.seed),
        "expected_evals": expected_evaluations(problem, cfg.N),
        **y_info,
    }
    return Si, df, meta


def results_to_dataframe(Si: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SALib Morris output dict to a tidy DataFrame.
    """
    df = pd.DataFrame(
        {
            "Parameter": Si["names"],
            "mu_star": Si["mu_star"],
            "sigma": Si["sigma"],
            "mu": Si["mu"],
        }
    )
    if "mu_star_conf" in Si:
        df["mu_star_conf"] = Si["mu_star_conf"]
    return df


# =========================================================
# Optional I/O helpers
# =========================================================
def save_results_csv(df: pd.DataFrame, path: str) -> str:
    """
    Save results DataFrame to CSV. Returns the path.
    """
    df.to_csv(path, index=False)
    return path


def save_meta_json(meta: Dict[str, Any], path: str) -> str:
    """
    Save meta dict to JSON. Returns the path.
    """
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


# =========================================================
# Optional plotting helpers (matplotlib)
# =========================================================
def plot_mu_star_bar(df: pd.DataFrame, path: str, title: str = "Morris sensitivity (mu*)") -> str:
    """
    Save a horizontal bar plot of mu*.
    """

    d = df.sort_values("mu_star", ascending=True)  # for barh (top at bottom)
    plt.figure(figsize=(9, 6))
    plt.barh(d["Parameter"], d["mu_star"])
    plt.xlabel("mu*")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_mu_sigma(df: pd.DataFrame, path: str, title: str = "Morris: mu* vs sigma") -> str:
    """
    Save mu* vs sigma scatter plot.
    """

    plt.figure(figsize=(6.5, 6))
    plt.scatter(df["mu_star"], df["sigma"])
    for _, r in df.iterrows():
        plt.text(r["mu_star"], r["sigma"], str(r["Parameter"]), fontsize=9)
    plt.xlabel("mu*")
    plt.ylabel("sigma")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path
