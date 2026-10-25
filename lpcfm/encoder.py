import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: [B, T, C]
        gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [B, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """1D ConvNeXt V2 block."""

    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        x = self.dwconv(x).transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)  # [B, T, C']
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x).transpose(1, 2)  # [B, C, T]
        return residual + x


class MelEncoder(nn.Module):
    """Maps mel bins to the STFT frequency dimension: Conv1d (k=7) + one ConvNeXt V2 block."""

    def __init__(self, n_mel, n_freq):
        super().__init__()
        self.embed = nn.Conv1d(n_mel, n_freq, kernel_size=7, padding=3)
        self.convnext = ConvNeXtV2Block(n_freq, n_freq * 3)

    def forward(self, mel):
        # mel: [B, M, T] -> [B, F, T]
        return self.convnext(self.embed(mel))
