# -*- coding: utf-8 -*-
"""
  outputs: [B, T]
  params_sequence: [B, T, 7]
"""
from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn

from Encoder import HydrologicParameterEncoderTF, HydrologicParameterEncoderLSTM
from Decoder import SIMHYDPhysicalDecoder


# =========================================================
# helpers
# =========================================================
def _require_3d_inputs(inputs: torch.Tensor) -> Tuple[int, int, int]:
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be [B,T,F], got shape={tuple(inputs.shape)}")
    B, T, F = inputs.size()
    if T <= 0:
        raise ValueError("T must be positive")
    if F != 3:
        raise ValueError(f"Expected F=3 (Temp,Prec,Evap), got F={F}")
    return int(B), int(T), int(F)


def _left_pad_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Left-pad along time dimension to target_len if x is shorter.
    x: [B, <=L, F] -> [B, L, F]
    """
    B, T, F = x.shape
    if T == target_len:
        return x
    if T > target_len:
        return x[:, -target_len:, :]
    pad_len = target_len - T
    pad = torch.zeros((B, pad_len, F), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1)


def _window_inclusive(inputs: torch.Tensor, t: int, L: int) -> torch.Tensor:
    """
    Window ending at t (inclusive): max(t-L+1,0) .. t
    returns [B, <=L, 3]
    """
    start = t - L + 1
    if start < 0:
        start = 0
    end = t + 1
    return inputs[:, start:end, :]


def _stack_outputs(outputs_list: List[torch.Tensor]) -> torch.Tensor:
    """
    outputs_list: list of [B,1]
    -> outputs: [B,T]
    """
    out = torch.stack(outputs_list, dim=1)  # [B,T,1]
    if out.ndim == 3 and out.size(-1) == 1:
        out = out.squeeze(-1)
    return out


def _stack_params(params_list: List[torch.Tensor]) -> torch.Tensor:
    """
    params_list: list of [B,7]
    -> [B,T,7]
    """
    return torch.stack(params_list, dim=1)


# =========================================================
# main model wrapper
# =========================================================
class _SIMHYDRNNLayerBase(nn.Module):
    """
    Base wrapper:
    - sliding window parameter inference (encoder)
    - one-step SIMHYD update (decoder)
    """
    def __init__(self, device: Optional[torch.device] = None, seq_length: int = 30):
        super().__init__()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seq_length = int(seq_length)

        # decoder is device-agnostic; model.to(device) will move buffers/params if any
        self.rnn_cell = SIMHYDPhysicalDecoder()

    # -------------------------
    # utilities
    # -------------------------
    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("inputs must be a torch.Tensor")
        if x.device != self.device:
            return x.to(self.device)
        return x

    def _init_states(self, B: int) -> torch.Tensor:
        """
        states: [B,2] = [GWt1, SMSt1]
        SMSt1 initial uses a fixed reference capacity (consistent with prior implementation).
        """
        if B <= 0:
            raise ValueError(f"Batch size must be positive, got B={B}")

        SMSt0 = 0.5
        SMSC_ref = 300.0
        SMSt1_initial = float(SMSt0) * float(SMSC_ref)

        states = torch.zeros((B, 2), device=self.device, dtype=torch.float32)
        states[:, 1] = SMSt1_initial
        return states

    def _compute_params(self, step_seq: torch.Tensor) -> torch.Tensor:
        """
        step_seq: [B, <=L, 3] or [B, L, 3]
        returns:  [B, 7]
        """
        L = self.seq_length
        if step_seq.size(1) != L:
            step_seq = _left_pad_time(step_seq, L)
        return self.hyper_net(step_seq)

    def _run_one_step(
        self,
        x_t: torch.Tensor,
        states: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t:    [B,3]
        states: [B,2]
        params: [B,7]
        returns:
          out_t:  [B,1]
          states: [B,2]
        """
        return self.rnn_cell(x_t, states, params)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, inputs: torch.Tensor):
        inputs = self._to_device(inputs)
        B, T, _ = _require_3d_inputs(inputs)

        states = self._init_states(B)

        outputs_list: List[torch.Tensor] = []
        params_list: List[torch.Tensor] = []

        L = self.seq_length
        for t in range(T):
            step_seq = _window_inclusive(inputs, t, L)    # [B,<=L,3]
            params = self._compute_params(step_seq)       # [B,7]
            x_t = inputs[:, t, :]                         # [B,3]
            out_t, states = self._run_one_step(x_t, states, params)

            outputs_list.append(out_t)
            params_list.append(params)

        outputs = _stack_outputs(outputs_list)           # [B,T]
        params_sequence = _stack_params(params_list)     # [B,T,7]
        return outputs, params_sequence


class SIMHYDRNNLayerTF(_SIMHYDRNNLayerBase):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        seq_length: int = 30,
        hyper_kwargs: Optional[Dict] = None,
    ):
        super().__init__(device=device, seq_length=seq_length)

        defaults = dict(
            input_size=3,
            hidden_size=64,
            dynamic_param_size=2,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
        )
        if hyper_kwargs:
            defaults.update(hyper_kwargs)

        self.hyper_net = HydrologicParameterEncoderTF(**defaults).to(self.device)


class SIMHYDRNNLayerLSTM(_SIMHYDRNNLayerBase):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        seq_length: int = 30,
        hyper_kwargs: Optional[Dict] = None,
    ):
        super().__init__(device=device, seq_length=seq_length)

        defaults = dict(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
        )
        if hyper_kwargs:
            hk = dict(hyper_kwargs)
            if "no_of_layers" in hk and "num_layers" not in hk:
                hk["num_layers"] = hk.pop("no_of_layers")
            if "drop_out_rate" in hk and "dropout" not in hk:
                hk["dropout"] = hk.pop("drop_out_rate")
            if "dynamic_param_size" in hk:
                hk.pop("dynamic_param_size", None)
            defaults.update(hk)

        self.hyper_net = HydrologicParameterEncoderLSTM(**defaults).to(self.device)


def build_model(model_name: str = "simhyd_hyper_tf", input_size: int = 3, **kwargs) -> nn.Module:
    if int(input_size) != 3:
        raise ValueError(f"Expected input_size=3 (Temp, Prec, Evap), got {input_size}")

    name = (model_name or "simhyd_hyper_tf").lower()

    device = kwargs.pop("device", None)
    seq_length = int(kwargs.pop("transformer_seq_length", kwargs.pop("seq_length", 30)))
    hyper_kwargs = kwargs.pop("hyper_kwargs", None)

    if name in ["simhyd_hyper_tf", "simhyd_hyper", "paper_model", "simhyd_rnn"]:
        return SIMHYDRNNLayerTF(device=device, seq_length=seq_length, hyper_kwargs=hyper_kwargs)

    if name in ["simhyd_hyper_lstm", "paper_model_lstm", "simhyd_lstm_hyper"]:
        return SIMHYDRNNLayerLSTM(device=device, seq_length=seq_length, hyper_kwargs=hyper_kwargs)

    raise ValueError(f"Unknown model_name={model_name}")