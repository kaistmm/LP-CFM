import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torchaudio
import yaml
from pesq import pesq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lpcfm.audio import AudioProcessor, peak_normalize
from lpcfm.dataset import LJSpeechDataset
from lpcfm.flow import CFMVocoder

logger = logging.getLogger("lpcfm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ljspeech.yaml")
    parser.add_argument("--exp_dir", required=True)
    # Optional overrides of the main experiment axes
    parser.add_argument("--channels", type=int, choices=[16, 32, 64])
    parser.add_argument("--mag", choices=["ot", "lp"])
    parser.add_argument("--pha", choices=["ot", "lp"])
    parser.add_argument("--vcs", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_config(args):
    # When resuming, the config saved in exp_dir takes precedence.
    saved = os.path.join(args.exp_dir, "config.yaml")
    if os.path.isfile(saved):
        with open(saved, "r") as f:
            return yaml.safe_load(f)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for section, key, value in [("model", "channels", args.channels), ("flow", "mag", args.mag),
                                ("flow", "pha", args.pha), ("sampling", "vcs", args.vcs)]:
        if value is not None:
            cfg[section][key] = value
    return cfg


def setup_logging(exp_dir):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(os.path.join(exp_dir, "train.log"))):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def build_model(cfg):
    return CFMVocoder(
        n_mel=cfg["audio"]["n_mel"],
        n_fft=cfg["audio"]["n_fft"],
        channels=cfg["model"]["channels"],
        mag_path=cfg["flow"]["mag"],
        pha_path=cfg["flow"]["pha"],
        lam=cfg["flow"]["lam"],
    )


def build_audio_processor(cfg):
    a = cfg["audio"]
    return AudioProcessor(a["sampling_rate"], a["n_fft"], a["hop_size"], a["win_size"],
                          a["n_mel"], a["fmin"], a["fmax"])


def spec_image(log_mag):
    # log_mag: [F, T] -> [1, F, T] in [0, 1], low frequencies at the bottom
    img = log_mag.flip(0)
    return ((img - img.min()) / (img.max() - img.min() + 1e-8)).unsqueeze(0)


@torch.no_grad()
def validate(model, audio, loader, cfg, writer, step, device):
    model.eval()
    sr = cfg["audio"]["sampling_rate"]
    pesq_scores = []
    for i, wav in enumerate(loader):
        wav = wav.to(device)  # [1, L]
        log_mag, _, mel = audio(wav)
        log_mag_g, pha_g = model(mel, cfg["sampling"]["n_steps"], vcs=cfg["sampling"]["vcs"])
        wav_g = audio.istft(log_mag_g, pha_g, length=wav.size(-1))  # [1, L]

        ref_16k = peak_normalize(torchaudio.functional.resample(wav, sr, 16000))
        deg_16k = peak_normalize(torchaudio.functional.resample(wav_g, sr, 16000))
        try:
            pesq_scores.append(pesq(16000, ref_16k[0].cpu().numpy(), deg_16k[0].cpu().numpy(), "wb"))
        except Exception as e:  # e.g. no utterance detected
            logger.warning(f"PESQ failed on sample {i}: {e}")

        if i < cfg["train"]["num_valid_samples"]:
            if step == cfg["train"]["valid_interval"]:
                writer.add_audio(f"gt/wav_{i}", peak_normalize(wav[0]).cpu(), step, sr)
                writer.add_image(f"gt/spec_{i}", spec_image(log_mag[0]), step)
            writer.add_audio(f"generated/wav_{i}", peak_normalize(wav_g[0]).cpu(), step, sr)
            writer.add_image(f"generated/spec_{i}", spec_image(log_mag_g[0]), step)

    val_pesq = float(np.mean(pesq_scores))
    writer.add_scalar("valid/pesq", val_pesq, step)
    model.train()
    return val_pesq


def main():
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(args.exp_dir, exist_ok=True)
    setup_logging(args.exp_dir)
    with open(os.path.join(args.exp_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    tcfg = cfg["train"]
    random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    audio = build_audio_processor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], betas=tuple(tcfg["betas"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tcfg["lr_decay"])
    logger.info(f"flow: mag={cfg['flow']['mag']}, pha={cfg['flow']['pha']} | UNet-{cfg['model']['channels']} | "
                f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Resume from the latest checkpoint in exp_dir
    step, start_epoch = 0, 0
    ckpts = sorted(glob.glob(os.path.join(args.exp_dir, "ckpt_*.pt")))
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch = state["step"] + 1, state["epoch"]
        logger.info(f"Resumed from {ckpts[-1]}")

    dcfg, sr = cfg["data"], cfg["audio"]["sampling_rate"]
    train_set = LJSpeechDataset(dcfg["train_filelist"], dcfg["wav_dir"], sr, tcfg["segment_size"])
    valid_set = LJSpeechDataset(dcfg["valid_filelist"], dcfg["wav_dir"], sr)
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=dcfg["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, num_workers=1, pin_memory=True)
    writer = SummaryWriter(os.path.join(args.exp_dir, "logs"))

    model.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        start = time.time()
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)  # [B, L]
            with torch.no_grad():
                log_mag, pha, mel = audio(wav)  # [B, F, T], [B, F, T], [B, M, T]

            loss, losses = model.compute_loss(log_mag, pha, mel)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()

            if torch.isnan(loss):
                raise ValueError(f"NaN loss at step {step}")

            if step % tcfg["log_interval"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"epoch {epoch + 1} | step {step} | loss {loss.item():.4f} "
                            f"(mag {losses['loss_mag'].item():.4f}, pha {losses['loss_pha'].item():.4f}) | "
                            f"grad_norm {grad_norm.item():.3f} | lr {lr:.2e}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/loss_mag", losses["loss_mag"].item(), step)
                writer.add_scalar("train/loss_pha", losses["loss_pha"].item(), step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                writer.add_scalar("train/lr", lr, step)

            if step % tcfg["checkpoint_interval"] == 0 and step != 0:
                path = os.path.join(args.exp_dir, f"ckpt_{step:08d}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "step": step, "epoch": epoch}, path)
                logger.info(f"Saved {path}")

            if step % tcfg["valid_interval"] == 0 and step != 0:
                val_pesq = validate(model, audio, valid_loader, cfg, writer, step, device)
                logger.info(f"[valid] step {step} | PESQ {val_pesq:.3f}")

            step += 1

        scheduler.step()
        logger.info(f"Epoch {epoch + 1} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
