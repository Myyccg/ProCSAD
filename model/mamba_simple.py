"""
Pure PyTorch implementation of BiMamba (Bidirectional Mamba)
No external mamba-ssm dependency required
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Simplified Selective State Space Model
    Uses a simpler, more stable implementation
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution (use padding='same' to maintain sequence length)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding='same',
            groups=self.d_inner
        )

        # SSM-like transformation (simplified)
        self.ssm_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.SiLU(),
            nn.Linear(self.d_inner, self.d_inner)
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Convolution for local context
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)  # (B, d_inner, T)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # SSM-like transformation
        y = self.ssm_proj(x_conv)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block
    Processes sequence in both forward and backward directions
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        residual = x
        x = self.norm(x)

        # Forward direction
        y_fwd = self.forward_ssm(x)

        # Backward direction
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Fuse bidirectional outputs
        y = self.fusion(torch.cat([y_fwd, y_bwd], dim=-1))
        y = self.dropout(y)

        return residual + y


class BiMambaEncoder(nn.Module):
    """
    BiMamba Encoder for time series
    """
    def __init__(self, input_dim, d_model=64, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, C) input time series
        returns: (B, T, D) encoded features
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
