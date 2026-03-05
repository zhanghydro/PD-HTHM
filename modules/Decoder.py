# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SimhydParamIndex:
    """
    Parameter ordering (must match encoder output):
      [SMSC, K, SUB, CRAK, SQ, COEFF, INSC]
    """
    SMSC: int = 0
    K: int = 1
    SUB: int = 2
    CRAK: int = 3
    SQ: int = 4
    COEFF: int = 5
    INSC: int = 6


def _check_2d_shape(x: torch.Tensor, last_dim: int, name: str) -> None:
    if x.dim() != 2 or x.size(-1) != last_dim:
        raise ValueError(f"{name} must be [B,{last_dim}], got {tuple(x.shape)}")


def _split_forcings(step_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # step_in: [B,3] -> (Temp, Prec, Evap) each [B,1]
    return step_in[:, 0:1], step_in[:, 1:2], step_in[:, 2:3]


def _split_states(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # states: [B,2] -> (GWt1, SMSt1) each [B,1]
    return states[:, 0:1], states[:, 1:2]


def _split_params(params: torch.Tensor, idx: SimhydParamIndex) -> Tuple[torch.Tensor, ...]:
    # params: [B,7] -> each [B,1]
    SMSC = params[:, idx.SMSC:idx.SMSC + 1]
    K = params[:, idx.K:idx.K + 1]
    SUB = params[:, idx.SUB:idx.SUB + 1]
    CRAK = params[:, idx.CRAK:idx.CRAK + 1]
    SQ = params[:, idx.SQ:idx.SQ + 1]
    COEFF = params[:, idx.COEFF:idx.COEFF + 1]
    INSC = params[:, idx.INSC:idx.INSC + 1]
    return SMSC, K, SUB, CRAK, SQ, COEFF, INSC


class SIMHYDPhysicalDecoder(nn.Module):
    """
    One-step SIMHYD update (process equations + state transition).

    Inputs
    ------
    step_in : Tensor [B, 3]
        [Temp, Prec, Evap] at current time step.
    states : Tensor [B, 2]
        [GWt1, SMSt1] (groundwater storage, soil moisture storage).
    params : Tensor [B, 7]
        [SMSC, K, SUB, CRAK, SQ, COEFF, INSC] already scaled to physical ranges.

    Outputs
    -------
    U : Tensor [B, 1]
        Total runoff at current time step.
    next_states : Tensor [B, 2]
        Updated [GW, SMS].
    """
    def __init__(self, eps: float = 1e-6, param_index: SimhydParamIndex = SimhydParamIndex()):
        super().__init__()
        self.eps = float(eps)
        self.idx = param_index

    def forward(self, step_in: torch.Tensor, states: torch.Tensor, params: torch.Tensor):
        _check_2d_shape(step_in, 3, "step_in")
        _check_2d_shape(states, 2, "states")
        _check_2d_shape(params, 7, "params")

        # unpack forcings/states/params (all [B,1])
        _temp, Prec, Evap = _split_forcings(step_in)
        GWt1, SMSt1 = _split_states(states)
        SMSC, K, SUB, CRAK, SQ, COEFF, INSC = _split_params(params, self.idx)

        # denominator guard
        denom = SMSC + self.eps

        # interception
        IMAX = torch.min(INSC, Evap)
        INT = torch.min(IMAX, Prec)
        INR = Prec - INT

        # runoff generation / partitioning
        RMO = torch.min(COEFF * torch.exp(-SQ * SMSt1 / denom), INR)
        IRUN = INR - RMO
        SRUN = SUB * SMSt1 / denom * RMO
        REC = CRAK * SMSt1 / denom * (RMO - SRUN)
        SMF = RMO - SRUN - REC

        # evapotranspiration
        POT = Evap - INT
        ET = torch.min(10 * SMSt1 / denom, POT)

        # soil moisture update with capacity constraint
        SMS = SMSt1 + SMF - ET
        REC = torch.where(SMS > SMSC, REC + (SMS - SMSC), REC)
        SMS = torch.min(SMS, SMSC)

        # groundwater and baseflow
        BAS = K * GWt1
        GW = GWt1 + REC - BAS

        # total runoff
        U = IRUN + SRUN + BAS

        next_states = torch.cat([GW, SMS], dim=-1)  # [B,2]
        return U, next_states