import json
import os
import random

import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import peak_normalize


class LJSpeechDataset(Dataset):
    """Loads peak-normalized waveforms listed in a json file of utterance ids.

    If segment_size is given, a random segment is cropped (or zero-padded) to that length.
    """

    def __init__(self, filelist, wav_dir, sampling_rate, segment_size=None):
        with open(filelist, "r") as f:
            self.utt_ids = json.load(f)
        self.wav_dir = wav_dir
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        path = os.path.join(self.wav_dir, self.utt_ids[index] + ".wav")
        wav, _ = librosa.load(path, sr=self.sampling_rate)
        wav = peak_normalize(torch.FloatTensor(wav))  # [L]

        if self.segment_size is not None:
            if wav.size(0) >= self.segment_size:
                start = random.randint(0, wav.size(0) - self.segment_size)
                wav = wav[start:start + self.segment_size]
            else:
                wav = F.pad(wav, (0, self.segment_size - wav.size(0)))
        return wav
