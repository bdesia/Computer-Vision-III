"""Differentiable augmentation (DiffAug, Zhao et al., NeurIPS 2020) for the 2D critic inputs.

The same random policy is applied to real crops and generated slices right before the critic, so the
critic cannot memorize the few real views of a single micrograph, while gradients still flow to G.
All ops act per sample on (N, C, H, W) one-hot phase maps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def translate(x: torch.Tensor, ratio: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Shift each sample by up to ratio*size pixels; borders are filled by reflection (stay one-hot)."""
    n, _, h, w = x.shape
    sy, sx = max(1, int(h * ratio)), max(1, int(w * ratio))
    padded = F.pad(x, (sx, sx, sy, sy), mode="reflect")
    dy = torch.randint(0, 2 * sy + 1, (n,), generator=gen, device="cpu").tolist()
    dx = torch.randint(0, 2 * sx + 1, (n,), generator=gen, device="cpu").tolist()
    return torch.stack([padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(n)])


def cutout(x: torch.Tensor, ratio: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Zero a random square of side ratio*size in each sample (as in DiffAug)."""
    n, _, h, w = x.shape
    ch, cw = max(1, int(h * ratio)), max(1, int(w * ratio))
    cy = torch.randint(0, h - ch + 1, (n,), generator=gen, device="cpu").tolist()
    cx = torch.randint(0, w - cw + 1, (n,), generator=gen, device="cpu").tolist()
    mask = torch.ones(n, 1, h, w, device=x.device, dtype=x.dtype)
    for i in range(n):
        mask[i, :, cy[i] : cy[i] + ch, cx[i] : cx[i] + cw] = 0
    return x * mask


def dihedral(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Random 90-degree rotation and horizontal flip per sample (D4 group; valid for isotropic media)."""
    n = x.shape[0]
    ks = torch.randint(0, 4, (n,), generator=gen, device="cpu").tolist()
    flips = (torch.rand(n, generator=gen, device="cpu") < 0.5).tolist()
    out = []
    for i in range(n):
        xi = torch.rot90(x[i], ks[i], dims=(1, 2))
        out.append(torch.flip(xi, dims=(2,)) if flips[i] else xi)
    return torch.stack(out)


class DiffAugment:
    """Callable DiffAug policy built from config: `train.diffaug: {policy: [...], translation, cutout}`."""

    def __init__(self, policy: list[str], translation: float = 0.125, cutout_ratio: float = 0.5,
                 seed: int = 0):
        """Validate the policy names and keep a dedicated CPU RNG for reproducibility."""
        unknown = set(policy) - {"translation", "cutout", "d4"}
        if unknown:
            raise ValueError(f"Unknown DiffAug ops {sorted(unknown)} (expected translation | cutout | d4)")
        self.policy, self.translation, self.cutout_ratio = list(policy), translation, cutout_ratio
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ops in the policy order."""
        for op in self.policy:
            if op == "translation":
                x = translate(x, self.translation, self.gen)
            elif op == "cutout":
                x = cutout(x, self.cutout_ratio, self.gen)
            elif op == "d4":
                x = dihedral(x, self.gen)
        return x


def build_diffaug(cfg: dict):
    """DiffAugment from `train.diffaug`, or identity when it is null/empty."""
    dcfg = cfg["train"].get("diffaug")
    if not dcfg or not dcfg.get("policy"):
        return lambda x: x
    return DiffAugment(dcfg["policy"], dcfg.get("translation", 0.125), dcfg.get("cutout", 0.5),
                       seed=cfg["seed"])
