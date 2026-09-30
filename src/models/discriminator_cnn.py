"""Original SliceGAN 2D CNN discriminator (M1), taken unchanged from upstream."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.slicegan_wrapper import upstream_classes


def build_cnn_discriminator(cfg: dict, run_dir: Path, training: bool = True) -> torch.nn.Module:
    """Instantiate the upstream 5-layer strided-conv critic (64x64 slice -> 1 logit)."""
    disc_cls, _ = upstream_classes(cfg["n_phases"], cfg["z_channels"], run_dir / "slicegan", training)
    return disc_cls()
