"""Swin-T 2D critic for SliceGAN (M2/M3): pretrained timm backbone, n-phase patch embedding, linear WGAN head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

log = get_logger(__name__)


def adapt_patch_embedding(conv_rgb: nn.Conv2d, n_phases: int) -> nn.Conv2d:
    """Build an n-phase patch embedding equivalent to the RGB one fed a centred grayscale image.

    The phases are mapped to gray levels k / (n_phases - 1) - 0.5 (for two phases: matrix -0.5,
    inclusion +0.5), which the pretrained filter sees replicated over R, G and B. Averaging the RGB
    weights instead would make one-hot inputs (channels summing to 1) indistinguishable.
    """
    if n_phases < 2:
        raise ValueError(f"n_phases must be >= 2, got {n_phases}")
    conv = nn.Conv2d(
        n_phases, conv_rgb.out_channels, conv_rgb.kernel_size, conv_rgb.stride,
        conv_rgb.padding, bias=conv_rgb.bias is not None,
    )
    with torch.no_grad():
        w_gray = conv_rgb.weight.sum(dim=1)  # response to a gray image replicated on RGB
        levels = torch.arange(n_phases, dtype=w_gray.dtype) / (n_phases - 1) - 0.5
        conv.weight.copy_(levels.view(1, -1, 1, 1) * w_gray.unsqueeze(1))
        if conv_rgb.bias is not None:
            conv.bias.copy_(conv_rgb.bias)
    return conv


class SwinCritic(nn.Module):
    """Swin-T backbone + global average pooling + linear layer -> one unbounded critic score per slice."""

    def __init__(self, backbone: nn.Module, n_phases: int, input_size: int, upsample: str = "bilinear"):
        """Wrap a timm Swin (created with num_classes=1) and swap its patch embedding to n phases."""
        super().__init__()
        backbone.patch_embed.proj = adapt_patch_embedding(backbone.patch_embed.proj, n_phases)
        self.backbone = backbone
        self.input_size = input_size
        self.upsample = upsample

    def freeze_stages(self, trainable_stages: list[int]) -> None:
        """Freeze patch embedding and the stages (1-based) not listed; final norm and head stay trainable."""
        self.backbone.patch_embed.requires_grad_(False)
        for idx, stage in enumerate(self.backbone.layers, start=1):
            stage.requires_grad_(idx in trainable_stages)
        self.backbone.norm.requires_grad_(True)
        self.backbone.head.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, n_phases, H, W) slices to (N, 1) critic scores."""
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode=self.upsample,
                              align_corners=False)
        return self.backbone(x)


def build_swin_discriminator(cfg: dict) -> SwinCritic:
    """Create the Swin-T critic from `model.swin` (pretrained weights from the Hugging Face Hub via timm)."""
    import timm

    scfg = cfg["model"]["swin"]
    try:
        backbone = timm.create_model(
            scfg["backbone"],
            pretrained=scfg["pretrained"],
            img_size=scfg["input_size"],
            num_classes=1,
            drop_rate=0.0,
            drop_path_rate=0.0,  # stochastic depth off: keeps the critic deterministic for the GP
        )
    except Exception as exc:  # network / hub / unknown model name
        raise RuntimeError(f"Could not create Swin backbone '{scfg['backbone']}': {exc}") from exc

    critic = SwinCritic(backbone, cfg["n_phases"], scfg["input_size"], scfg["upsample"])
    critic.freeze_stages(scfg["trainable_stages"])
    windows = [tuple(block.window_size) for stage in backbone.layers for block in stage.blocks[:1]]
    log.info(
        "Swin critic %s (pretrained=%s) | input %d | per-stage window %s | trainable stages %s",
        scfg["backbone"], scfg["pretrained"], scfg["input_size"], windows, scfg["trainable_stages"],
    )
    return critic
