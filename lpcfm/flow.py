import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import UNetDecoder
from .encoder import MelEncoder


def project(x, a):
    """Per-sample projection onto the line direction a: P x = a (a^T x) / (a^T a).

    x: [B, F, T], a: broadcastable to x -> [B, F, T]
    """
    a = a.expand_as(x)
    coef = (a * x).sum(dim=(1, 2), keepdim=True) / (a * a).sum(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    return coef * a


class CFMVocoder(nn.Module):
    """Mel-to-waveform vocoder trained with OT-CFM or LP-CFM.

    The flow runs jointly on the log-magnitude and phase spectrograms, and each
    component can use either path ("ot" or "lp"). Line directions:
      - log-magnitude (amplitude scaling): a = 1
      - phase (time shift): a[k] = 2 * pi * k / N  (sign does not affect the projection)
    """

    def __init__(self, n_mel=80, n_fft=1024, channels=32, mag_path="lp", pha_path="lp", lam=1e-4):
        super().__init__()
        assert mag_path in ("ot", "lp") and pha_path in ("ot", "lp")
        self.n_freq = n_fft // 2 + 1
        self.mag_path = mag_path
        self.pha_path = pha_path
        self.lam = lam  # lambda for LP-CFM, sigma_min for OT-CFM

        self.mel_encoder = MelEncoder(n_mel, self.n_freq)
        self.decoder = UNetDecoder(channels)

        k = torch.arange(self.n_freq, dtype=torch.float32)
        self.register_buffer("mag_dir", torch.ones(1, 1, 1), persistent=False)
        self.register_buffer("pha_dir", (2 * math.pi * k / n_fft).view(1, -1, 1), persistent=False)

    def conditional_path(self, x1, t, path, a):
        """Samples x_t and the target velocity u_t.

        x1: [B, F, T] data, t: [B, 1, 1], a: line direction
        returns x_t, u_t: [B, F, T]
        """
        x0 = torch.randn_like(x1)
        if path == "ot":
            x_t = (1 - (1 - self.lam) * t) * x0 + t * x1
            u_t = x1 - (1 - self.lam) * x0
        else:
            # Target: N(b - Pb, M M^T) with M = lam * I + (1 - lam) * P and b = x1
            x1_lp = x1 - project(x1, a) + self.lam * x0 + (1 - self.lam) * project(x0, a)
            x_t = (1 - t) * x0 + t * x1_lp
            u_t = x1_lp - x0
        return x_t, u_t

    def compute_loss(self, log_mag, pha, mel):
        """
        log_mag, pha: [B, F, T] target spectrograms
        mel: [B, M, T] condition
        """
        t = torch.rand(log_mag.size(0), 1, 1, device=log_mag.device, dtype=log_mag.dtype)  # [B, 1, 1]
        xt_mag, u_mag = self.conditional_path(log_mag, t, self.mag_path, self.mag_dir)
        xt_pha, u_pha = self.conditional_path(pha, t, self.pha_path, self.pha_dir)

        cond = self.mel_encoder(mel)  # [B, F, T]
        v_mag, v_pha = self.decoder(xt_mag, xt_pha, cond, t.view(-1))

        loss_mag = F.mse_loss(v_mag, u_mag)
        loss_pha = F.mse_loss(v_pha, u_pha)
        return loss_mag + loss_pha, {"loss_mag": loss_mag.detach(), "loss_pha": loss_pha.detach()}

    @staticmethod
    def calibrate(v, a):
        """Vector Calibrated Sampling: remove the component parallel to the line, keep the norm.

        v: [B, F, T] -> [B, F, T]
        """
        v_perp = v - project(v, a)
        scale = v.norm(dim=(1, 2), keepdim=True) / v_perp.norm(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        return scale * v_perp

    @torch.inference_mode()
    def forward(self, mel, n_steps=6, vcs=False, temperature=1.0):
        """Generates (log_mag, pha) from mel with the Euler solver.

        mel: [B, M, T] -> log_mag, pha: [B, F, T]
        """
        b, _, n_frame = mel.shape
        x_mag = torch.randn(b, self.n_freq, n_frame, device=mel.device) * temperature
        x_pha = torch.randn(b, self.n_freq, n_frame, device=mel.device) * temperature
        cond = self.mel_encoder(mel)  # [B, F, T]
        return self.solve_euler(x_mag, x_pha, cond, n_steps, vcs)

    def solve_euler(self, x_mag, x_pha, cond, n_steps, vcs=False):
        t_span = torch.linspace(0, 1, n_steps + 1, device=cond.device)
        for i in range(n_steps):
            t, dt = t_span[i], t_span[i + 1] - t_span[i]
            v_mag, v_pha = self.decoder(x_mag, x_pha, cond, t)
            if vcs:
                v_mag = self.calibrate(v_mag, self.mag_dir)
                v_pha = self.calibrate(v_pha, self.pha_dir)
            x_mag = x_mag + dt * v_mag
            x_pha = x_pha + dt * v_pha
        return x_mag, x_pha
