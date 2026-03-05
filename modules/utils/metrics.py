# -*- coding: utf-8 -*-
"""
Notes
-----
- All metrics align observation/simulation pairs and ignore non-finite values (NaN/Inf).
- Error handling is configurable:
    * on_error="nan"   -> return np.nan for undefined cases
    * on_error="raise" -> raise RuntimeError with a short message
- R2 in this file is defined as r^2 (square of Pearson correlation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Literal

import numpy as np

EPS = 1e-8
OnError = Literal["nan", "raise"]


# =========================
# small helpers
# =========================
def _fail(msg: str, on_error: OnError) -> float:
    if on_error == "raise":
        raise RuntimeError(msg)
    return np.nan


def _align(obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float).flatten()
    sim = np.asarray(sim, dtype=float).flatten()
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def _require_same_shape(obs: np.ndarray, sim: np.ndarray, on_error: OnError) -> Optional[float]:
    if obs.shape != sim.shape:
        return _fail("obs and sim must be of the same length.", on_error)
    return None


def _prep_pair(obs: np.ndarray, sim: np.ndarray, on_error: OnError) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Align obs/sim, drop non-finite, then validate shapes.
    Returns:
      (obs_f, sim_f, bad)
    where bad is None if OK, otherwise a float failure value (NaN or exception already raised).
    """
    obs_f, sim_f = _align(obs, sim)
    bad = _require_same_shape(obs_f, sim_f, on_error)
    if bad is not None:
        return None, None, bad
    return obs_f, sim_f, None


# =========================
# metric functions
# =========================
def calc_nse(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Nash-Sutcliffe Efficiency (NSE)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 2:
        return _fail("NSE requires at least 2 valid paired samples.", on_error)

    denom = np.sum((obs_f - np.mean(obs_f)) ** 2)
    if denom < EPS:
        return _fail("NSE undefined when all observation values are (near) identical.", on_error)

    numer = np.sum((sim_f - obs_f) ** 2)
    return float(1.0 - numer / (denom + EPS))


def calc_mnse(obs: np.ndarray, sim: np.ndarray, *, alpha: float = 1.0, on_error: OnError = "nan") -> float:
    """Modified NSE (alpha-norm variant)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 2:
        return _fail("mNSE requires at least 2 valid paired samples.", on_error)

    denom = np.sum(np.abs(obs_f - np.mean(obs_f)) ** alpha)
    if denom < EPS:
        return _fail("mNSE undefined when denominator is (near) zero.", on_error)

    numer = np.sum(np.abs(obs_f - sim_f) ** alpha)
    return float(1.0 - numer / (denom + EPS))


def calc_rmse(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Root Mean Squared Error (RMSE)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 1:
        return _fail("RMSE requires at least 1 valid paired sample.", on_error)
    return float(np.sqrt(np.mean((sim_f - obs_f) ** 2)))


def calc_pbias(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Percent Bias (PBIAS)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad

    denom = np.sum(obs_f)
    if abs(denom) < EPS:
        return _fail("PBIAS undefined when sum(obs) is (near) zero.", on_error)
    return float(100.0 * (np.sum(sim_f) - np.sum(obs_f)) / (denom + EPS))


def calc_pearson_r(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Pearson correlation coefficient r."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 2:
        return _fail("Pearson r requires at least 2 valid paired samples.", on_error)

    if np.std(obs_f) < EPS or np.std(sim_f) < EPS:
        return _fail("Pearson r undefined when std(obs) or std(sim) is (near) zero.", on_error)

    r = np.corrcoef(obs_f, sim_f)[0, 1]
    if not np.isfinite(r):
        return _fail("Pearson r is not finite.", on_error)
    return float(r)


def calc_r2(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Correlation-based R2 (R2 = r^2)."""
    r = calc_pearson_r(obs, sim, on_error=on_error)
    if not np.isfinite(r):
        return np.nan
    return float(r * r)


def calc_kge(obs: np.ndarray, sim: np.ndarray, *, on_error: OnError = "nan") -> float:
    """Kling-Gupta Efficiency (KGE, 2009)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 2:
        return _fail("KGE requires at least 2 valid paired samples.", on_error)

    r = calc_pearson_r(obs_f, sim_f, on_error=on_error)
    if not np.isfinite(r):
        return np.nan

    sd_obs = np.std(obs_f, ddof=0)
    sd_sim = np.std(sim_f, ddof=0)
    if sd_obs < EPS:
        return _fail("KGE undefined when std(obs) is (near) zero.", on_error)
    alpha = sd_sim / (sd_obs + EPS)

    mu_obs = np.mean(obs_f)
    mu_sim = np.mean(sim_f)
    if abs(mu_obs) < EPS:
        return _fail("KGE undefined when mean(obs) is (near) zero.", on_error)
    beta = mu_sim / (mu_obs + EPS)

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def calc_pfab(
    obs: np.ndarray,
    sim: np.ndarray,
    *,
    exceedance_probability: float = 0.02,
    on_error: OnError = "nan",
) -> float:
    """Peak Flow Absolute Bias (PFAB)."""
    obs_f, sim_f, bad = _prep_pair(obs, sim, on_error)
    if bad is not None:
        return bad
    if obs_f.size < 1:
        return _fail("PFAB requires at least 1 valid paired sample.", on_error)
    if not (0.0 < exceedance_probability < 1.0):
        return _fail("exceedance_probability must be in (0,1).", on_error)

    L = max(1, int(obs_f.size * exceedance_probability))
    idx = np.argsort(obs_f)[::-1]  # sort by observed descending

    top_obs = obs_f[idx][:L]
    top_sim = sim_f[idx][:L]

    denom = np.sum(top_obs)
    if abs(denom) < EPS:
        return _fail("PFAB undefined when sum(top_obs) is (near) zero.", on_error)

    return float(100.0 * np.abs(np.sum(top_sim - top_obs) / (denom + EPS)))


# =========================
# optional utilities
# =========================
METRIC_REGISTRY: Dict[str, Callable[..., float]] = {
    "nse": calc_nse,
    "mnse": calc_mnse,
    "rmse": calc_rmse,
    "pbias": calc_pbias,
    "pearson_r": calc_pearson_r,
    "r2": calc_r2,
    "kge": calc_kge,
    "pfab": calc_pfab,
}


def calc_all_metrics(
    obs: np.ndarray,
    sim: np.ndarray,
    *,
    alpha_mnse: float = 1.0,
    exceedance_probability: float = 0.02,
    on_error: OnError = "nan",
) -> Dict[str, float]:
    """
    Convenience wrapper for a common metric bundle.
    """
    return {
        "NSE": calc_nse(obs, sim, on_error=on_error),
        "mNSE": calc_mnse(obs, sim, alpha=alpha_mnse, on_error=on_error),
        "RMSE": calc_rmse(obs, sim, on_error=on_error),
        "PBIAS": calc_pbias(obs, sim, on_error=on_error),
        "KGE": calc_kge(obs, sim, on_error=on_error),
        "R2": calc_r2(obs, sim, on_error=on_error),
        "PFAB": calc_pfab(obs, sim, exceedance_probability=exceedance_probability, on_error=on_error),
    }