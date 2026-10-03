"""Evaluate G_last (and G_best) of every model on the same 32 seeds, without touching metrics.yaml."""
from pathlib import Path

import numpy as np
import torch

from src.features.descriptors import describe_volumes
from src.models.generate import generate_volumes, load_generator
from src.models.slicegan_wrapper import load_label_map
from src.utils import load_config

ML = "configs/data/microlib_000210.yaml"
ref = load_label_map(Path("data/processed/microlib_000210/train_2d/image.png"))
dev = torch.device("cuda")
print(f"{'model':13s} {'ckpt':5s} {'phi':>14s} {'|dphi|':>7s} {'S2 MAE':>7s} {'L MAE':>7s}")
for m in ("m1_cnn", "m1_extended", "m2_swin", "m3_swin_sam", "m4_ensemble", "m5_finetune"):
    cfg = load_config(f"configs/{m}.yaml", [ML])
    run_dir = Path(cfg["paths"]["models"]) / m
    for tag in ("best", "last"):
        netG = load_generator(cfg, run_dir, dev, checkpoint=f"G_{tag}.pt")
        vols = generate_volumes(netG, list(range(32)), cfg["z_channels"], dev)
        o = describe_volumes(vols, ref, 32)
        print(f"{m:13s} {tag:5s} {o['phi_mean']:.3f}±{o['phi_std']:.3f} {o['abs_dphi']:7.4f} {o['s2_mae']:7.4f} {o['L_mae']:7.4f}")
