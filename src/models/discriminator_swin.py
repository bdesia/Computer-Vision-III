"""Swin-T 2D critic for SliceGAN (M2/M3): pretrained timm backbone, n-phase patch embedding, linear WGAN head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

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


class ImprovedSpectralNorm(nn.Module):
    """ViTGAN's improved spectral normalization: W -> sigma(W_init) * W / sigma(W).

    Plain spectral normalization (W / sigma(W)) forces every layer to Lipschitz constant 1, which
    ViTGAN found to underfit; rescaling by the spectral norm at initialization keeps each layer at its
    starting (here: pretrained) scale while preventing it from growing during training
    (Lee et al., ICLR 2022, Eq. 7). sigma(W) is tracked with one power iteration per training forward.
    """

    def __init__(self, weight: torch.Tensor, init_iterations: int = 50, eps: float = 1e-12):
        """Estimate sigma(W_init) with power iteration and keep the singular vectors as buffers."""
        super().__init__()
        self.eps = eps
        w = weight.detach().flatten(1)
        u = F.normalize(torch.randn(w.shape[0], device=w.device, dtype=w.dtype), dim=0, eps=eps)
        for _ in range(init_iterations):
            v = F.normalize(w.t() @ u, dim=0, eps=eps)
            u = F.normalize(w @ v, dim=0, eps=eps)
        self.register_buffer("u", u)
        self.register_buffer("v", v)
        self.register_buffer("sigma_init", torch.dot(u, w @ v))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the rescaled weight; update the power-iteration vectors in training mode."""
        w = weight.flatten(1)
        if self.training:
            with torch.no_grad():
                v = F.normalize(w.t() @ self.u, dim=0, eps=self.eps)
                u = F.normalize(w @ v, dim=0, eps=self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        # Clones: several forwards (real, fake, GP interpolates) share one backward, so the graph
        # must not hold the buffers that the next training forward updates in place.
        u, v = self.u.clone(), self.v.clone()
        sigma = torch.dot(u, w @ v)
        return weight * (self.sigma_init / sigma)


def apply_improved_spectral_norm(module: nn.Module) -> int:
    """Register ISN on every trainable nn.Linear weight inside `module`; returns how many layers."""
    count = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear) and layer.weight.requires_grad:
            parametrize.register_parametrization(layer, "weight", ImprovedSpectralNorm(layer.weight))
            count += 1
    return count


class SwinCritic(nn.Module):
    """Swin-T backbone + global average pooling + linear layer -> one unbounded critic score per slice."""

    def __init__(self, backbone: nn.Module, n_phases: int, input_size: int, upsample: str = "bilinear",
                 head: str = "linear", feature_stages: tuple[int, ...] = (2, 3, 4), head_hidden: int = 256):
        """Wrap a timm Swin (created with num_classes=1) and swap its patch embedding to n phases.

        head="linear": timm's pooled last-stage features + linear layer (stages chosen by freeze_stages).
        head="multiscale": frozen backbone, pooled features of `feature_stages`, each LayerNorm-ed and
        concatenated into a small MLP (Vision-aided GAN style: pretrained features, trainable head only).
        """
        super().__init__()
        backbone.patch_embed.proj = adapt_patch_embedding(backbone.patch_embed.proj, n_phases)
        self.backbone = backbone
        self.input_size = input_size
        self.upsample = upsample
        self.head_type = head
        if head == "multiscale":
            dims = [backbone.feature_info[i - 1]["num_chs"] for i in feature_stages]
            self.feature_stages = tuple(feature_stages)
            self.stage_norms = nn.ModuleList(nn.LayerNorm(d) for d in dims)
            self.ms_head = nn.Sequential(nn.Linear(sum(dims), head_hidden), nn.LeakyReLU(0.2),
                                         nn.Linear(head_hidden, 1))
        elif head != "linear":
            raise ValueError(f"Unknown model.swin.head '{head}' (expected linear | multiscale)")

    def freeze_stages(self, trainable_stages: list[int]) -> None:
        """Freeze patch embedding and the stages (1-based) not listed; final norm and head stay trainable."""
        self.backbone.patch_embed.requires_grad_(False)
        for idx, stage in enumerate(self.backbone.layers, start=1):
            stage.requires_grad_(idx in trainable_stages)
        self.backbone.norm.requires_grad_(True)
        self.backbone.head.requires_grad_(True)

    def freeze_backbone(self) -> None:
        """Freeze every backbone weight (used with the multiscale head)."""
        self.backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, n_phases, H, W) slices to (N, 1) critic scores."""
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode=self.upsample,
                              align_corners=False)
        if self.head_type == "linear":
            return self.backbone(x)
        h = self.backbone.patch_embed(x)
        pooled = []
        for idx, stage in enumerate(self.backbone.layers, start=1):
            h = stage(h)  # NHWC
            if idx in self.feature_stages:
                pooled.append(self.stage_norms[len(pooled)](h.mean(dim=(1, 2))))
            if idx == max(self.feature_stages):
                break
        return self.ms_head(torch.cat(pooled, dim=1))


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

    head = scfg.get("head", "linear")
    critic = SwinCritic(backbone, cfg["n_phases"], scfg["input_size"], scfg["upsample"], head=head,
                        feature_stages=tuple(scfg.get("feature_stages", (2, 3, 4))),
                        head_hidden=scfg.get("head_hidden", 256))
    if head == "multiscale":
        critic.freeze_backbone()
    else:
        critic.freeze_stages(scfg["trainable_stages"])
    if scfg.get("isn"):
        n = apply_improved_spectral_norm(critic)
        log.info("Improved spectral normalization (ViTGAN) on %d trainable linear layers", n)
    windows = [tuple(block.window_size) for stage in backbone.layers for block in stage.blocks[:1]]
    log.info(
        "Swin critic %s (pretrained=%s) | input %d | per-stage window %s | head %s | trainable %s",
        scfg["backbone"], scfg["pretrained"], scfg["input_size"], windows, head,
        f"head on stages {scfg.get('feature_stages', [2, 3, 4])} (backbone frozen)" if head == "multiscale"
        else f"stages {scfg['trainable_stages']}",
    )
    return critic
