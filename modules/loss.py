# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn.functional as F


# -------------------------
# helpers
# -------------------------
def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """[T] -> [1,T,1], [B,T] -> [B,T,1], [B,T,1] keep."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3 and x.size(-1) == 1:
        return x
    raise ValueError(f"Expected [T] or [B,T] or [B,T,1], got {tuple(x.shape)}")


def _apply_warmup(x: torch.Tensor, warmup_len: int) -> torch.Tensor:
    if warmup_len is None or warmup_len <= 0:
        return x
    T = x.size(1)
    if warmup_len >= T:
        # 不崩，返回空时间维
        return x[:, 0:0, :]
    return x[:, warmup_len:, :]


def _prepare(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    warmup_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize shapes, move device, apply warmup slicing, build mask."""
    yhat = _ensure_3d(yhat).to(dtype=torch.float32)
    y = _ensure_3d(y).to(dtype=torch.float32, device=yhat.device)

    yhat = _apply_warmup(yhat, warmup_len)
    y = _apply_warmup(y, warmup_len)

    if mask is None:
        mask3 = torch.ones_like(y, dtype=y.dtype, device=y.device)
    else:
        mask3 = _ensure_3d(mask).to(dtype=y.dtype, device=y.device)
        mask3 = _apply_warmup(mask3, warmup_len)

    return yhat, y, mask3


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x, mask: [B,T,1]
    return: scalar (mean over time per basin, then mean over basins)
    """
    if x.size(1) == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    denom = mask.sum(dim=1).clamp_min(eps)          # [B,1]
    xb = (x * mask).sum(dim=1) / denom              # [B,1]
    return xb.mean()


# -------------------------
# losses
# -------------------------
def nse_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    Loss = 1 - NSE (batch mean).
    Compatible with yhat,y: [B,T] or [B,T,1] or [T].
    """
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)

    if y.size(1) == 0:
        return torch.tensor(0.0, device=y.device, dtype=y.dtype)

    cnt = mask3.sum(dim=1, keepdim=True).clamp_min(1.0)      # [B,1,1]
    y_mean = (y * mask3).sum(dim=1, keepdim=True) / cnt      # [B,1,1]

    num = (((y - yhat) ** 2) * mask3).sum(dim=1)             # [B,1]
    den = (((y - y_mean) ** 2) * mask3).sum(dim=1)           # [B,1]

    nse_b = 1.0 - num / (den + eps)                          # [B,1]
    return (1.0 - nse_b).mean()


def mse_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """Masked MSE (batch mean)."""
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)
    se = (yhat - y) ** 2
    return _masked_mean(se, mask3, eps=eps)


def rmse_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """Masked RMSE (batch mean)."""
    return torch.sqrt(mse_loss(yhat, y, mask=mask, warmup_len=warmup_len, eps=eps) + eps)


def mae_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """Masked MAE (batch mean)."""
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)
    ae = torch.abs(yhat - y)
    return _masked_mean(ae, mask3, eps=eps)


def huber_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    delta: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """Masked Huber (SmoothL1) loss (batch mean)."""
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)
    l = F.huber_loss(yhat, y, delta=delta, reduction="none")  # [B,T,1]
    return _masked_mean(l, mask3, eps=eps)


def log_cosh_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """Masked log-cosh loss (batch mean)."""
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)
    r = yhat - y
    l = torch.log(torch.cosh(r) + eps)
    return _masked_mean(l, mask3, eps=eps)


def weighted_mse_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mode: str = "highflow",
    alpha: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    Masked weighted MSE (batch mean).

    mode:
      - "highflow": larger weight for higher observed flow
      - "lowflow" : larger weight for lower observed flow
      - "sqrt"    : milder emphasis using sqrt scaling
    """
    yhat, y, mask3 = _prepare(yhat, y, mask, warmup_len)
    if y.size(1) == 0:
        return torch.tensor(0.0, device=y.device, dtype=y.dtype)

    cnt = mask3.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1,1]
    mean_y = (y * mask3).sum(dim=1, keepdim=True) / cnt
    ratio = y / (mean_y + eps)

    if mode == "highflow":
        w = 1.0 + alpha * ratio
    elif mode == "lowflow":
        w = 1.0 + alpha * (1.0 - ratio)
        w = torch.clamp(w, min=0.0)
    elif mode == "sqrt":
        w = 1.0 + alpha * torch.sqrt(torch.clamp(ratio, min=0.0))
    else:
        raise ValueError(f"Unknown mode={mode}")

    se = (yhat - y) ** 2
    l = se * w
    return _masked_mean(l, mask3, eps=eps)


def composite_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    warmup_len: int = 0,
    terms: Optional[Dict[str, float]] = None,
    **kwargs
) -> torch.Tensor:
    """
    Weighted sum of multiple losses.

    Example:
      terms = {"nse": 1.0, "mae": 0.1, "huber": 0.05}

    Supported keys:
      "nse", "mse", "rmse", "mae", "huber", "logcosh", "wmse"
    """
    if terms is None:
        terms = {"nse": 1.0}

    total = None
    for name, w in terms.items():
        if w == 0:
            continue

        if name == "nse":
            l = nse_loss(yhat, y, mask=mask, warmup_len=warmup_len)
        elif name == "mse":
            l = mse_loss(yhat, y, mask=mask, warmup_len=warmup_len)
        elif name == "rmse":
            l = rmse_loss(yhat, y, mask=mask, warmup_len=warmup_len)
        elif name == "mae":
            l = mae_loss(yhat, y, mask=mask, warmup_len=warmup_len)
        elif name == "huber":
            l = huber_loss(yhat, y, delta=1.0, mask=mask, warmup_len=warmup_len)
        elif name == "logcosh":
            l = log_cosh_loss(yhat, y, mask=mask, warmup_len=warmup_len)
        elif name == "wmse":
            l = weighted_mse_loss(yhat, y, mode="highflow", alpha=1.0, mask=mask, warmup_len=warmup_len)
        else:
            raise ValueError(f"Unknown loss term: {name}")

        total = l * float(w) if total is None else total + l * float(w)

    if total is None:
        return torch.tensor(0.0, device=_ensure_3d(yhat).device, dtype=torch.float32)
    return total