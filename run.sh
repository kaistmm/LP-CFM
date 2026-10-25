#!/bin/bash
# UNet-32 experiments (uncomment one line per GPU)

# OT-CFM
# CUDA_VISIBLE_DEVICES=0 python train.py --exp_dir exp/ot_unet32 --channels 32 --mag ot --pha ot --no-vcs

# LP-CFM on magnitude only
# CUDA_VISIBLE_DEVICES=1 python train.py --exp_dir exp/lpmag_unet32 --channels 32 --mag lp --pha ot --no-vcs

# LP-CFM on phase only
# CUDA_VISIBLE_DEVICES=2 python train.py --exp_dir exp/lppha_unet32 --channels 32 --mag ot --pha lp --no-vcs

# LP-CFM
# CUDA_VISIBLE_DEVICES=3 python train.py --exp_dir exp/lp_unet32 --channels 32 --mag lp --pha lp --vcs
