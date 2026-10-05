"""M6: a 2D Diffusion Transformer (DiT) trained on micrograph crops, sampled in 3D by multi-plane denoising.

The denoiser is a small DiT (Peebles & Xie, ICCV 2023): the 64 x 64 slice is cut into patches, embedded as
tokens with fixed 2D sin-cos positions, processed by transformer blocks whose LayerNorms are modulated by
the diffusion timestep (adaLN-Zero), and unpatchified into a noise prediction. It is trained as a standard
DDPM (epsilon prediction, cosine schedule) on 64 x 64 crops of the one training micrograph, with the phase
map encoded as -1 (matrix) / +1 (inclusion).

3D volumes are generated from 3D Gaussian noise without any 3D data or adversarial training, with one of:

- "average" / "cycle": multi-plane DDIM as in Micro3Diff (Lee & Yun, 2024): at every step the noise is predicted
  slice by slice along all axes (averaged) or along one axis per step. With this model both drift to the
  majority phase: once one axis has been denoised, slices along the other axes look like stripes the 2D model
  never saw, and it falls back to predicting the matrix.
- "sdedit": a stack of independent 2D samples along z is made 3D-consistent by rounds of SDEdit
  (Meng et al., 2022): re-noise to t*, denoise along the next axis (y, x, z, ...), binarize. Each round makes
  another family of sections consistent; with `fixed_phi` the binarization keeps the training phase fraction,
  which otherwise drifts towards the matrix in the same way.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AXES = ("z", "y", "x")


# ----------------------------------------------------------------------------- DiT denoiser


def sincos_pos_embed_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding (grid*grid, dim), as in MAE / DiT."""
    if dim % 4:
        raise ValueError(f"Embedding dim {dim} must be divisible by 4")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim / 4))
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float64), torch.arange(grid, dtype=torch.float64),
                            indexing="ij")
    out = []
    for pos in (ys.reshape(-1), xs.reshape(-1)):
        ang = pos[:, None] * omega[None]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1).float()


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of (possibly fractional) timesteps -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning on the timestep."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        self.heads = heads
        # QK-norm (LayerNorm on queries and keys per head) bounds the attention logits; without it the first M6
        # run diverged after ~5k steps under bf16 autocast
        self.q_norm = nn.LayerNorm(dim // heads, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dim // heads, eps=1e-6) if qk_norm else nn.Identity()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)  # adaLN-Zero: every block starts as the identity
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        n, l, d = x.shape
        h = modulate(self.norm1(x), sh1, sc1)
        q, k, v = self.qkv(h).reshape(n, l, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v).transpose(1, 2).reshape(n, l, d)
        x = x + g1[:, None] * self.proj(h)
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), sh2, sc2))


class DiT(nn.Module):
    """Unconditional DiT denoiser for (N, C, S, S) images; returns the predicted noise, same shape."""

    def __init__(self, img_size: int = 64, patch: int = 4, in_ch: int = 1, dim: int = 384, depth: int = 8,
                 heads: int = 6, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        if img_size % patch:
            raise ValueError(f"img_size {img_size} is not a multiple of patch {patch}")
        self.img_size, self.patch, self.in_ch = img_size, patch, in_ch
        self.grid = img_size // patch
        self.embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.register_buffer("pos", sincos_pos_embed_2d(dim, self.grid)[None], persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio, qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * in_ch)
        for m in (self.ada[1], self.out):  # zero-init output path (DiT)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.t_mlp(timestep_embedding(t, 256))
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.ada(c).chunk(2, dim=1)
        h = self.out(modulate(self.norm(h), shift, scale))  # (N, grid*grid, p*p*C)
        p, g = self.patch, self.grid
        h = h.reshape(n, g, g, p, p, self.in_ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(n, self.in_ch, g * p, g * p)


def build_dit(cfg: dict) -> DiT:
    d = cfg["model"]["dit"]
    return DiT(cfg["img_size"], d["patch"], 1, d["dim"], d["depth"], d["heads"], d.get("mlp_ratio", 4.0),
               d.get("qk_norm", False))


# ----------------------------------------------------------------------------- noise schedule and DDIM


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal fraction alpha_bar_t, t = 0..T-1 (Nichol & Dhariwal, 2021), clipped like iDDPM."""
    f = lambda u: math.cos((u / timesteps + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
    betas = [min(1 - f(i + 1) / f(i), 0.999) for i in range(timesteps)]
    return torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), dim=0).float()


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Forward process x_t = sqrt(ab) x0 + sqrt(1 - ab) noise (any number of trailing dims)."""
    ab = alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def ddim_timesteps(timesteps: int, steps: int) -> list[int]:
    """Evenly spaced decreasing timesteps from T-1 to 0."""
    return sorted({int(round(v)) for v in np.linspace(0, timesteps - 1, steps)}, reverse=True)


def ddim_step(x: torch.Tensor, eps: torch.Tensor, ab_t: torch.Tensor, ab_prev: torch.Tensor, eta: float = 0.0,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """DDIM update with the x0 prediction clipped to the data range [-1, 1].

    eta = 0 is deterministic DDIM; eta = 1 re-injects DDPM-level noise, which lets the per-axis predictions of
    multi-plane sampling correct each other over the following steps.
    """
    x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()  # noise consistent with the clipped x0
    if eta == 0:
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
    z = torch.randn(x.shape, device=x.device, generator=generator)
    return ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * z


# ----------------------------------------------------------------------------- sampling


@torch.no_grad()
def predict_eps_slices(model: DiT, slices: torch.Tensor, t: int, chunk: int) -> torch.Tensor:
    """Noise prediction for (N, 1, S, S) slices in chunks (bf16 autocast on CUDA)."""
    out = torch.empty_like(slices)
    use_amp = slices.is_cuda
    for i in range(0, slices.shape[0], chunk):
        part = slices[i : i + chunk]
        tt = torch.full((part.shape[0],), t, device=slices.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out[i : i + chunk] = model(part, tt).float()
    return out


def predict_eps_volume(model: DiT, vol: torch.Tensor, t: int, axes: tuple[str, ...], chunk: int) -> torch.Tensor:
    """Average over `axes` of the slice-wise noise prediction of a (D, H, W) volume."""
    total = torch.zeros_like(vol)
    for axis in axes:
        dim = AXES.index(axis)
        slices = vol.movedim(dim, 0).unsqueeze(1)  # (L, 1, S, S): the sections orthogonal to `axis`
        eps = predict_eps_slices(model, slices, t, chunk).squeeze(1).movedim(0, dim)
        total += eps
    return total / len(axes)


@torch.no_grad()
def sample_volume(model: DiT, alpha_bar: torch.Tensor, size: int, seed: int, steps: int = 50, mode: str = "average",
                  chunk: int = 64, device=None, eta: float = 0.0) -> np.ndarray:
    """One {0, 1} volume of edge `size` by multi-plane DDIM sampling.

    mode "average": every step averages the predictions along z, y and x (3 model passes per step);
    mode "cycle": one axis per step, rotating z -> y -> x (1 pass per step, Micro3Diff-style alternation).
    """
    if mode not in ("average", "cycle"):
        raise ValueError(f"Unknown sampling mode '{mode}' (expected average | cycle)")
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(size, size, size, device=device, generator=gen)
    ab = alpha_bar.to(device)
    ts = ddim_timesteps(len(ab), steps)
    for i, t in enumerate(ts):
        axes = AXES if mode == "average" else (AXES[i % 3],)
        eps = predict_eps_volume(model, x, t, axes, chunk)
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x = ddim_step(x, eps, ab[t], ab_prev, eta, gen)
    return (x > 0).to(torch.uint8).cpu().numpy()


def binarize_signed(x: torch.Tensor, phi: float | None) -> torch.Tensor:
    """{-1, +1} field: sign of x, or thresholded at the (1 - phi) quantile so that a fraction phi is +1."""
    if phi is None:
        return torch.where(x > 0, 1.0, -1.0)
    flat = x.flatten()
    sample = flat[:: max(1, flat.numel() // 200_000)].float()  # quantile on a subsample (torch size limit)
    return torch.where(x > torch.quantile(sample, 1 - phi), 1.0, -1.0)


@torch.no_grad()
def denoise_from(model: DiT, x: torch.Tensor, alpha_bar: torch.Tensor, t_start: int, steps: int, axes_at, chunk: int,
                 eta: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """DDIM on a volume from timestep t_start down to 0 on the `steps`-point grid; axes_at(i) -> axes of step i."""
    ts = [t for t in ddim_timesteps(len(alpha_bar), steps) if t <= t_start]
    for i, t in enumerate(ts):
        eps = predict_eps_volume(model, x, t, axes_at(i), chunk)
        ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
        x = ddim_step(x, eps, alpha_bar[t], ab_prev, eta, generator)
    return x


@torch.no_grad()
def sample_volume_sdedit(model: DiT, alpha_bar: torch.Tensor, size: int, seed: int, steps: int = 20,
                         t_star: int = 700, rounds: int = 12, phi: float | None = None, chunk: int = 64,
                         device=None) -> np.ndarray:
    """Independent 2D samples along z, then `rounds` SDEdit passes along y, x, z, ... (see module docstring)."""
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    ab = alpha_bar.to(device)
    x = torch.randn(size, size, size, device=device, generator=gen)
    x = denoise_from(model, x, ab, len(ab) - 1, steps, lambda i: ("z",), chunk, generator=gen)
    for r in range(rounds):
        axis = ("y", "x", "z")[r % 3]
        x0 = binarize_signed(x, phi)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x = ab[t_star].sqrt() * x0 + (1 - ab[t_star]).sqrt() * noise
        x = denoise_from(model, x, ab, t_star, steps, lambda i, a=axis: (a,), chunk, generator=gen)
    return (binarize_signed(x, phi) > 0).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def sample_slices(model: DiT, alpha_bar: torch.Tensor, n: int, seed: int, steps: int = 50, chunk: int = 64,
                  device=None) -> np.ndarray:
    """n independent 2D samples (N, S, S) in {0, 1} (checks the 2D model on its own)."""
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    s = model.img_size
    x = torch.randn(n, 1, s, s, device=device, generator=gen)
    ab = alpha_bar.to(device)
    ts = ddim_timesteps(len(ab), steps)
    for i, t in enumerate(ts):
        eps = predict_eps_slices(model, x, t, chunk)
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x = ddim_step(x, eps, ab[t], ab_prev)
    return (x[:, 0] > 0).to(torch.uint8).cpu().numpy()


class DiffusionVolumeGenerator(nn.Module):
    """Wraps a trained DiT so the evaluation code can call `sample_volume(seed)` like a generator."""

    def __init__(self, model: DiT, alpha_bar: torch.Tensor, size: int, steps: int, mode: str, chunk: int,
                 eta: float = 0.0, t_star: int = 700, rounds: int = 12, phi: float | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.size, self.steps, self.mode, self.chunk, self.eta = size, steps, mode, chunk, eta
        self.t_star, self.rounds, self.phi = t_star, rounds, phi

    def sample_volume(self, seed: int) -> np.ndarray:
        if self.mode == "sdedit":
            return sample_volume_sdedit(self.model, self.alpha_bar, self.size, seed, self.steps, self.t_star,
                                        self.rounds, self.phi, self.chunk)
        return sample_volume(self.model, self.alpha_bar, self.size, seed, self.steps, self.mode, self.chunk,
                             eta=self.eta)


def build_volume_generator(cfg: dict, state_dict: dict, device, phi: float | None = None) -> DiffusionVolumeGenerator:
    """Rebuild the DiT from config, load weights and wrap it with the sampling settings of `model.dit`.

    `phi` is the training phase fraction, kept fixed by the sdedit sampler when `model.dit.fixed_phi` is set.
    """
    d = cfg["model"]["dit"]
    model = build_dit(cfg)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return DiffusionVolumeGenerator(model, cosine_alpha_bar(d["timesteps"]), cfg["volume_size"],
                                    d["sample_steps"], d["sample_mode"], d.get("sample_chunk", 64),
                                    d.get("sample_eta", 0.0), d.get("sdedit_t", 700), d.get("sdedit_rounds", 12),
                                    phi if d.get("fixed_phi") else None).to(device)
