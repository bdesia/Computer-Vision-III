"""Thin wrapper around upstream SliceGAN (external/SliceGAN): networks, gradient penalty and slicing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SLICEGAN_ROOT = Path(__file__).resolve().parents[2] / "external" / "SliceGAN"
if not (_SLICEGAN_ROOT / "slicegan" / "networks.py").exists():
    raise ImportError(
        f"SliceGAN not found at {_SLICEGAN_ROOT}. Run `make vendor` (git submodule update --init)."
    )
if str(_SLICEGAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLICEGAN_ROOT))

from slicegan import networks as sg_networks  # noqa: E402
from slicegan import util as sg_util  # noqa: E402

# Architecture verbatim from upstream run_slicegan.py (G must stay identical across M1/M2/M3).
IMAGE_TYPE = "nphase"
LZ = 4  # spatial size of the latent cube; 4 -> 64^3 output with the rc generator
_LAYS_G, _LAYS_D = 5, 6
_DK, _GK = [4] * _LAYS_D, [4] * _LAYS_G
_DS, _GS = [2] * _LAYS_D, [2] * _LAYS_G
_DP, _GP = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

# Upstream permutations turning a (B, C, D, H, W) volume into 2D slices along each axis.
SLICE_PERMUTATIONS = ((2, 3, 4), (3, 2, 4), (4, 2, 3))


def architecture(n_phases: int, z_channels: int) -> dict:
    """Upstream layer lists (kernels, strides, filters, paddings) for D and G."""
    return {
        "dk": _DK, "ds": _DS, "df": [n_phases, 64, 128, 256, 512, 1], "dp": _DP,
        "gk": _GK, "gs": _GS, "gf": [z_channels, 1024, 512, 128, 32, n_phases], "gp": _GP,
    }


def upstream_classes(n_phases: int, z_channels: int, params_prefix: str | Path, training: bool):
    """Return upstream (Discriminator, Generator) classes; writes/reads `<prefix>_params.data`."""
    a = architecture(n_phases, z_channels)
    prefix = Path(params_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        return sg_networks.slicegan_rc_nets(
            str(prefix), training, IMAGE_TYPE,
            a["dk"], a["ds"], a["df"], a["dp"], a["gk"], a["gs"], a["gf"], a["gp"],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing SliceGAN params file {prefix}_params.data") from exc


def build_generator(cfg: dict, run_dir: Path, training: bool = True) -> torch.nn.Module:
    """Instantiate the unmodified upstream 3D generator."""
    _, gen_cls = upstream_classes(cfg["n_phases"], cfg["z_channels"], run_dir / "slicegan", training)
    return gen_cls()


def calc_gradient_penalty(netD, real, fake, batch_size, l, device, gp_lambda, nc) -> torch.Tensor:
    """Upstream WGAN-GP penalty (unchanged)."""
    return sg_util.calc_gradient_penalty(netD, real, fake, batch_size, l, device, gp_lambda, nc)


def sample_noise(n: int, z_channels: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Latent cube (n, z, LZ, LZ, LZ) as used upstream."""
    return torch.randn(n, z_channels, LZ, LZ, LZ, device=device, generator=generator)


def volume_to_slices(volume: torch.Tensor, perm: tuple[int, int, int]) -> torch.Tensor:
    """Turn a (B, C, L, L, L) volume batch into (B*L, C, L, L) slices along one axis (upstream permute)."""
    b, c, l = volume.shape[0], volume.shape[1], volume.shape[-1]
    d1, d2, d3 = perm
    return volume.permute(0, d1, 1, d2, d3).reshape(l * b, c, l, l)


def to_labels(volume: torch.Tensor) -> np.ndarray:
    """Convert generator softmax output (B, C, L, L, L) into uint8 label volumes (B, L, L, L)."""
    return volume.argmax(dim=1).to(torch.uint8).cpu().numpy()


def load_label_map(path: str | Path) -> np.ndarray:
    """Load a {0, 1} label map saved as PNG (0 / 255)."""
    path = Path(path)
    try:
        with Image.open(path) as im:
            return (np.asarray(im.convert("L")) > 127).astype(np.uint8)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training label map not found: {path}. Run `make data` first.") from exc


class RandomCropSampler:
    """Samples one-hot random crops from a 2D label map on the target device (replaces upstream batch())."""

    def __init__(self, labels: np.ndarray, crop: int, n_phases: int, device, seed: int, augment: bool = False):
        """Keep the label map on `device` and a dedicated RNG for crop positions (and augmentation)."""
        if min(labels.shape) < crop + 2:
            raise ValueError(f"Label map {labels.shape} too small for {crop}x{crop} crops")
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.crop = crop
        self.n_phases = n_phases
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def __call__(self, batch_size: int) -> torch.Tensor:
        """Return a (batch, n_phases, crop, crop) float one-hot batch."""
        h, w = self.labels.shape
        l = self.crop
        # Same bounds as upstream preprocessing.batch for 2D images
        xs = self.rng.integers(1, h - l - 1, size=batch_size)
        ys = self.rng.integers(1, w - l - 1, size=batch_size)
        crops = [self.labels[x : x + l, y : y + l] for x, y in zip(xs, ys)]
        if self.augment:  # rigid symmetries only: random 90-degree rotation + horizontal flip (D4 group)
            ks = self.rng.integers(0, 4, size=batch_size)
            flips = self.rng.random(batch_size) < 0.5
            crops = [torch.rot90(c, int(k), dims=(0, 1)) for c, k in zip(crops, ks)]
            crops = [torch.flip(c, dims=(1,)) if f else c for c, f in zip(crops, flips)]
        crops = torch.stack(crops)
        return F.one_hot(crops, self.n_phases).permute(0, 3, 1, 2).float()
