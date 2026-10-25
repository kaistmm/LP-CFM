# LP-CFM: Perceptual Invariance-Aware Conditional Flow Matching for Speech Modeling

Official implementation of **LP-CFM** (ICASSP 2026).

Doyeop Kwak, Youngjoon Jang, Joon Son Chung

LP-CFM models each target as an elongated Gaussian along a line of perceptually
equivalent variants (amplitude scaling for log-magnitude, time shift for phase),
instead of an isotropic Gaussian around a single sample. Vector Calibrated
Sampling (VCS) keeps the sampling process aligned with this line-projection path.

This repository is a cleaned-up version of the code used for the neural vocoding
experiments in the paper (OT-CFM vs. LP-CFM on LJ Speech), kept minimal as a
reference implementation. The experiments are a controlled comparison between flow
matching formulations rather than an attempt at a state-of-the-art vocoder.

## Structure

```
configs/ljspeech.yaml   # paper settings
filelists/              # LJ Speech train (12,950) / validation (150) split
lpcfm/
  audio.py              # STFT, iSTFT, log-mel
  dataset.py            # LJ Speech dataset
  encoder.py            # mel encoder (Conv1d + ConvNeXt V2 block)
  decoder.py            # flow matching decoder (2D UNet)
  flow.py               # OT-CFM / LP-CFM paths, loss, Euler sampler with VCS
train.py
inference.py
```

## Setup

```bash
pip install -r requirements.txt
```

Download [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) and set `data.wav_dir`
in `configs/ljspeech.yaml` to the `LJSpeech-1.1/wavs` directory.

## Training

The main experiment axes can be set in the config or overridden from the command line:

| Option | Values | Description |
|---|---|---|
| `--channels` | `16`, `32`, `64` | UNet-16 / UNet-32 / UNet-64 |
| `--mag` | `lp`, `ot` | flow path for the log-magnitude spectrogram |
| `--pha` | `lp`, `ot` | flow path for the phase spectrogram |
| `--vcs` / `--no-vcs` | | VCS during validation sampling |

```bash
# LP-CFM, UNet-32
python train.py --exp_dir exp/lp_unet32 --channels 32 --mag lp --pha lp

# OT-CFM, UNet-32
python train.py --exp_dir exp/ot_unet32 --channels 32 --mag ot --pha ot --no-vcs

# Ablation: LP-CFM on magnitude only
python train.py --exp_dir exp/lpmag_unet32 --channels 32 --mag lp --pha ot --no-vcs
```

Training resumes automatically from the latest checkpoint in `--exp_dir`, using the
config saved there. Logs are written to `<exp_dir>/logs` for TensorBoard.

## Inference

Copy-synthesis from ground-truth mel-spectrograms (wav -> mel -> wav):

```bash
python inference.py --exp_dir exp/lp_unet32 --ckpt exp/lp_unet32/ckpt_XXXXXXXX.pt \
    --input_dir /path/to/wavs --output_dir samples/lp_unet32 --n_steps 6 --vcs
```

VCS is a sampling-time option, so the same checkpoint can be evaluated with and
without it. Since LP-CFM can produce outputs at different scales, outputs are
peak-normalized to 0.95 (as in the paper's evaluation).

## Acknowledgement

The training pipeline is based on [SEMamba](https://github.com/roychao19477/semamba).
We thank the authors for releasing their code.

This work was supported by IITP grant funded by the Korea government
(MSIT, RS-2024-00457882, National AI Research Lab Project).

## Citation

```bibtex
@inproceedings{kwak2026lp,
  title={LP-CFM: Perceptual Invariance-Aware Conditional Flow Matching for Speech Modeling},
  author={Kwak, Doyeop and Jang, Youngjoon and Chung, Joon Son},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17727--17731},
  year={2026},
  organization={IEEE}
}
```
