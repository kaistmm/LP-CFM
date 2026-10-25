import argparse
import glob
import logging
import os

import librosa
import soundfile as sf
import torch
import yaml

from lpcfm.audio import peak_normalize
from train import build_audio_processor, build_model

logger = logging.getLogger("lpcfm")


def main():
    parser = argparse.ArgumentParser(description="Copy-synthesis: wav -> mel -> generated wav")
    parser.add_argument("--exp_dir", required=True, help="training directory containing config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True, help="directory of .wav files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--vcs", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(os.path.join(args.exp_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    n_steps = args.n_steps if args.n_steps is not None else cfg["sampling"]["n_steps"]
    vcs = args.vcs if args.vcs is not None else cfg["sampling"]["vcs"]
    sr = cfg["audio"]["sampling_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    audio = build_audio_processor(cfg).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    logger.info(f"{len(paths)} files | n_steps={n_steps} | vcs={vcs}")

    for path in paths:
        wav, _ = librosa.load(path, sr=sr)
        wav = peak_normalize(torch.FloatTensor(wav)).unsqueeze(0).to(device)  # [1, L]
        mel = audio.mel(wav)  # [1, M, T]
        log_mag, pha = model(mel, n_steps, vcs=vcs)  # [1, F, T]
        wav_g = peak_normalize(audio.istft(log_mag, pha, length=wav.size(-1)))  # [1, L]
        sf.write(os.path.join(args.output_dir, os.path.basename(path)), wav_g[0].cpu().numpy(), sr)


if __name__ == "__main__":
    main()
