import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class UNetDecoder(nn.Module):
    """Flow matching decoder: 2D UNet predicting magnitude and phase vectors.

    Three scales with one ResNet block each and no attention. Channels are
    (c, 2c, 4c) with c // 8 normalization groups (UNet-16 / 32 / 64).
    """

    num_downsamples = 3

    def __init__(self, channels=32):
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=3,  # x_t (log-magnitude), x_t (phase), encoded mel
            out_channels=channels,
            layers_per_block=1,
            block_out_channels=(channels, channels * 2, channels * 4),
            mid_block_type="UNetMidBlock2D",
            down_block_types=("ResnetDownsampleBlock2D",) * 3,
            up_block_types=("ResnetUpsampleBlock2D",) * 3,
            add_attention=False,
            norm_num_groups=channels // 8,
        )
        self.mag_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.pha_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x_mag, x_pha, cond, t):
        """
        x_mag, x_pha: [B, F, T] current flow states
        cond: [B, F, T] encoded mel
        t: [B] or scalar flow time in [0, 1]
        returns v_mag, v_pha: [B, F, T]
        """
        x = torch.stack((x_mag, x_pha, cond), dim=1)  # [B, 3, F, T]
        n_freq, n_frame = x.shape[-2:]

        # Pad F and T to multiples of the total UNet stride.
        stride = 2 ** self.num_downsamples
        f_pad = (stride - n_freq % stride) % stride
        t_pad = (stride - n_frame % stride) % stride
        x = F.pad(x, (0, t_pad, 0, f_pad), mode="reflect")  # [B, 3, F', T']

        h = self.unet(x, t).sample  # [B, C, F', T']
        v_mag = self.mag_head(h)[:, 0, :n_freq, :n_frame]  # [B, F, T]
        v_pha = self.pha_head(h)[:, 0, :n_freq, :n_frame]  # [B, F, T]
        return v_mag, v_pha
