import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


class AudioProcessor(nn.Module):
    """STFT analysis/synthesis and log-mel extraction.

    Targets are the log-magnitude and phase spectrograms; the condition is a log-mel spectrogram.
    """

    def __init__(self, sampling_rate, n_fft, hop_size, win_size, n_mel, fmin, fmax):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel, fmin=fmin, fmax=fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float(), persistent=False)  # [M, F]
        self.register_buffer("window", torch.hann_window(win_size), persistent=False)

    def stft(self, wav):
        # wav: [B, L] -> spec: [B, F, T] (complex), F = n_fft // 2 + 1
        return torch.stft(wav, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.window, center=True, pad_mode="reflect",
                          normalized=False, return_complex=True)

    def forward(self, wav):
        """Returns (log_mag, pha, log_mel) from a single STFT.

        wav: [B, L] -> log_mag: [B, F, T], pha: [B, F, T], log_mel: [B, M, T]
        """
        spec = self.stft(wav)
        log_mag = torch.log(spec.abs() + 1e-4)
        pha = torch.angle(spec)

        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-10).clamp(min=3e-5)
        log_mel = torch.log(torch.clamp(self.mel_basis @ mag, min=1e-5))
        return log_mag, pha, log_mel

    def mel(self, wav):
        # wav: [B, L] -> log_mel: [B, M, T]
        return self.forward(wav)[2]

    def istft(self, log_mag, pha, length=None):
        # log_mag, pha: [B, F, T] -> wav: [B, L]
        mag = torch.exp(log_mag)
        spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return torch.istft(spec, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                           window=self.window, center=True, length=length)


def peak_normalize(wav, peak=0.95):
    # wav: [..., L], normalized over the whole tensor
    return peak * wav / wav.abs().max()
