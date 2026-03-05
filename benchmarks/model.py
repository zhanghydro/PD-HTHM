# -*- coding: utf-8 -*-
"""
model.py
- LSTM / GRU / Transformer Encoder-Decoder
- Input: x [B, T, F], Output: y_hat [B, 1]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)  # dropout on the last hidden state (not RNN internal dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # [B, T, H]
        h = self.drop(out[:, -1, :])     # [B, H]
        y_hat = self.fc(h)               # [B, 1]
        return {"y_hat": y_hat}


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)  # dropout on the last hidden state
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)             # [B, T, H]
        h = self.drop(out[:, -1, :])     # [B, H]
        y_hat = self.fc(h)               # [B, 1]
        return {"y_hat": y_hat}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerED(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_proj = nn.Linear(input_size, d_model)
        self.tgt_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_src = PositionalEncoding(d_model, dropout=dropout, max_len=4096)
        self.pos_tgt = PositionalEncoding(d_model, dropout=dropout, max_len=16)

        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.size(0)
        src = self.pos_src(self.src_proj(x))                 # [B, T, D]
        tgt = self.pos_tgt(self.tgt_token.expand(B, 1, -1))  # [B, 1, D]
        out = self.tf(src=src, tgt=tgt)                      # [B, 1, D]
        y_hat = self.head(out[:, -1, :])                     # [B, 1]
        return {"y_hat": y_hat}


def build_model(model_name: str, input_size: int, **kwargs) -> nn.Module:
    m = model_name.lower()
    if m == "lstm":
        return LSTM(input_size=input_size, **kwargs)
    if m == "gru":
        return GRU(input_size=input_size, **kwargs)
    if m in ["transformer", "transformer_ed", "transformer-ed"]:
        return TransformerED(input_size=input_size, **kwargs)
    raise ValueError(f"Unknown model: {model_name} (choose lstm/gru/transformer_ed)")