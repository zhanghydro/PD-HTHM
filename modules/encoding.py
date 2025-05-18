import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention is All You Need".
    Adds periodic encoding to the input to represent positions.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding, inspired by BERT.
    Each position has an associated learnable embedding.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]


class ZeroPositionalEncoding(nn.Module):
    """
    No-op positional encoding (adds zero vector). Useful for ablation studies.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.register_buffer('pe', torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Time2VecEncoding(nn.Module):
    """
    Time2Vec encoding as in "Time2Vec: Learning a Vector Representation of Time".
    Combines linear and periodic encoding of time.
    """
    def __init__(self, d_model, time_feature_dim):
        super().__init__()
        self.linear = nn.Linear(time_feature_dim, 1)
        self.freq = nn.Parameter(torch.randn(time_feature_dim, d_model - 1))
        self.phase = nn.Parameter(torch.randn(time_feature_dim, d_model - 1))

    def forward(self, t):
        # t: [batch_size, seq_len, time_feature_dim]
        linear_term = self.linear(t)  # [batch, seq_len, 1]
        periodic = torch.sin(torch.matmul(t, self.freq) + self.phase)  # [batch, seq_len, d_model-1]
        return torch.cat([linear_term, periodic], dim=-1)  # [batch, seq_len, d_model]


class RotaryPositionalEmbedding(nn.Module):
    """
    Placeholder for rotary positional embedding.
    Not implemented but stub provided for structure completeness.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # This is a placeholder.
        return x


