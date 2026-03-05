# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from encoding import PositionalEncoding
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


@dataclass(frozen=True)
class ParamSpec:
    """
    Parameter ordering must match paper and downstream code.
    Output: [SMSC, K, SUB, CRAK, SQ, COEFF, INSC]
    """
    names: Tuple[str, ...] = ("SMSC", "K", "SUB", "CRAK", "SQ", "COEFF", "INSC")
    smsc_scale: float = 1000.0
    sq_scale: float = 10.0
    coeff_scale: float = 400.0
    insc_scale: float = 20.0


def _check_input_3d(X: torch.Tensor, feat_dim: int, name: str = "X") -> None:
    if X.ndim != 3:
        raise ValueError(f"{name} must be [B,T,F], got {tuple(X.shape)}")
    if X.size(-1) != feat_dim:
        raise ValueError(f"{name} feature dim mismatch: expected {feat_dim}, got {X.size(-1)}")


def _logit(p: float) -> float:
    p = float(p)
    return float(torch.log(torch.tensor(p / (1.0 - p))).item())


def _split_cls_seq(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H: [B, T+1, H]
    returns:
      H_cls: [B, H]
      H_seq: [B, T, H]
    """
    return H[:, 0, :], H[:, 1:, :]


class _BaseHyperNet(nn.Module):
    """
    Shared pieces:
    - 5 global learnable static parameters
    - output assembly + scaling
    """
    def __init__(self, spec: ParamSpec = ParamSpec()):
        super().__init__()
        self.spec = spec

        # static parameters (learnable)=:
        self.SUB = nn.Parameter(torch.tensor(0.5))
        self.CRAK = nn.Parameter(torch.tensor(0.5))
        self.SQ = nn.Parameter(torch.tensor(0.5))
        self.COEFF = nn.Parameter(torch.tensor(0.5))
        self.INSC = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def get_static_param_leaf_names() -> Tuple[str, ...]:
        return ("SUB", "CRAK", "SQ", "COEFF", "INSC")

    def _assemble_params(self, B: int, theta_dynamic_2: torch.Tensor) -> torch.Tensor:
        """
        theta_dynamic_2: [B,2] in (0,1) after sigmoid -> map to (SMSC, K) then append static params.
        """
        SMSC = theta_dynamic_2[:, 0:1] * self.spec.smsc_scale
        K = theta_dynamic_2[:, 1:2]

        SUB = self.SUB.repeat(B, 1)
        CRAK = self.CRAK.repeat(B, 1)
        SQ = (self.SQ * self.spec.sq_scale).repeat(B, 1)
        COEFF = (self.COEFF * self.spec.coeff_scale).repeat(B, 1)
        INSC = (self.INSC * self.spec.insc_scale).repeat(B, 1)

        return torch.cat([SMSC, K, SUB, CRAK, SQ, COEFF, INSC], dim=-1)

    # paper-friendly alias (no behavior change)
    def assemble_theta_t(self, B: int, theta_dynamic_2: torch.Tensor) -> torch.Tensor:
        return self._assemble_params(B, theta_dynamic_2)


class _TemporalAttentionPooling(nn.Module):
    """
    Temporal attention pooling with a learnable query vector.
    Input : H_seq [B,T,H]
    Output: pooled [B,H]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.query = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        nn.init.xavier_uniform_(self.query)

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        B, T, H = H_seq.shape
        q = self.query.expand(B, 1, H)                          # [B,1,H]
        scores = torch.bmm(H_seq, q.transpose(1, 2))            # [B,T,1]
        alpha = torch.softmax(scores, dim=1)                    # [B,T,1]
        pooled = torch.bmm(H_seq.transpose(1, 2), alpha).squeeze(2)  # [B,H]
        return pooled


class HydrologicParameterEncoderTF(_BaseHyperNet):
    """
    Transformer-based hydrologic parameter encoder.

    Input:  [B, T, 3]
    Output: [B, 7] = [SMSC, K, SUB, CRAK, SQ, COEFF, INSC]
    """
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        dynamic_param_size: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        spec: ParamSpec = ParamSpec(),
    ):
        super().__init__(spec=spec)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        if int(dynamic_param_size) != 2:
            raise ValueError("This setup expects dynamic_param_size=2 for (SMSC, K).")

        self.input_linear = nn.Linear(self.input_size, self.hidden_size)

        self.lstm_pre_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size, dropout=dropout)

        enc_layer = TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(self.hidden_size)
        )

        self.temporal_pool = _TemporalAttentionPooling(self.hidden_size)
        self.fc_dynamic_params = nn.Linear(self.hidden_size, 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.cls_token)

        # start dynamic head around 0.5 after sigmoid
        with torch.no_grad():
            b = _logit(0.5)  # == 0.0, kept explicit for clarity
            nn.init.constant_(self.fc_dynamic_params.bias, float(b))
            nn.init.constant_(self.fc_dynamic_params.weight, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _check_input_3d(X, self.input_size, name="X")

        H0 = self.input_linear(X)               # [B,T,H]
        HLSTM, _ = self.lstm_pre_encoder(H0)    # [B,T,H]

        B = HLSTM.size(0)
        H_cls = self.cls_token.expand(B, 1, self.hidden_size)     # [B,1,H]
        H_cat = torch.cat([H_cls, HLSTM], dim=1)                  # [B,T+1,H]

        H_pos = self.pos_encoder(H_cat)                           # [B,T+1,H]
        H_enc = self.transformer_encoder(H_pos)                   # [B,T+1,H]

        H_enc_cls, H_enc_seq = _split_cls_seq(H_enc)              # [B,H], [B,T,H]
        pooled = self.temporal_pool(H_enc_seq)                    # [B,H]
        H_out = H_enc_cls + pooled                                # [B,H]

        theta_dynamic_2 = torch.sigmoid(self.fc_dynamic_params(H_out))  # [B,2]
        return self._assemble_params(B, theta_dynamic_2)


class HydrologicParameterEncoderLSTM(_BaseHyperNet):
    """
    LSTM baseline hypernetwork.

    Input:  [B, T, 3]
    Output: [B, 7] = [SMSC, K, SUB, CRAK, SQ, COEFF, INSC]
    """
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        spec: ParamSpec = ParamSpec(),
    ):
        super().__init__(spec=spec)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.input_linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=(dropout if self.num_layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(dropout)
        self.fc_dynamic_params = nn.Linear(self.hidden_size, 2)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize forget gate bias (optional)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                H = self.hidden_size
                with torch.no_grad():
                    param[H:2 * H].fill_(3.0)

        with torch.no_grad():
            b = _logit(0.5)  # == 0.0
            nn.init.constant_(self.fc_dynamic_params.bias, float(b))
            nn.init.constant_(self.fc_dynamic_params.weight, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _check_input_3d(X, self.input_size, name="X")

        H0 = self.input_linear(X)              # [B,T,H]
        out, _ = self.lstm(H0)                 # [B,T,H]
        h_last = self.drop(out[:, -1, :])      # [B,H]

        theta_dynamic_2 = torch.sigmoid(self.fc_dynamic_params(h_last))  # [B,2]
        B = X.size(0)
        return self._assemble_params(B, theta_dynamic_2)