"""Microstructure descriptors: phase fraction (phi) and radially averaged two-point correlation (S2)."""

from __future__ import annotations

import numpy as np

ORIENTATIONS = ("xy", "xz", "yz")


def indicator(labels: np.ndarray, phase: int = 1) -> np.ndarray:
    """Float64 indicator function of `phase` (1 where labels == phase, else 0)."""
    return (np.asarray(labels) == phase).astype(np.float64)


def phase_fraction(labels: np.ndarray, phase: int = 1) -> float:
    """Volume/area fraction of `phase` in a 2D image or 3D volume."""
    return float(indicator(labels, phase).mean())


def s2_map(images: np.ndarray, phase: int = 1, periodic: bool = False) -> np.ndarray:
    """Two-point correlation S2 over all lag vectors, centred at zero lag.

    `images` is (H, W) or (N, H, W). Returns (..., 2H-1, 2W-1) for non-periodic images, where each
    lag is normalized by its number of valid pixel pairs, or (..., H, W) for periodic images.
    """
    x = indicator(images, phase)
    h, w = x.shape[-2:]
    if periodic:
        f = np.fft.rfft2(x)
        auto = np.fft.irfft2(f * np.conj(f), s=(h, w)) / (h * w)
        return np.fft.fftshift(auto, axes=(-2, -1))

    shape = (2 * h, 2 * w)  # zero-pad to avoid wrap-around
    f = np.fft.rfft2(x, s=shape)
    auto = np.fft.irfft2(f * np.conj(f), s=shape)
    ones = np.fft.rfft2(np.ones((h, w)), s=shape)
    counts = np.rint(np.fft.irfft2(ones * np.conj(ones), s=shape))
    counts[counts == 0] = 1.0  # lag (+-h, +-w) has no pairs; it is cropped below anyway
    s2 = np.fft.fftshift(auto / counts, axes=(-2, -1))
    return s2[..., 1:, 1:]  # drop the empty +-h / +-w row and column


def _radial_bins(shape: tuple[int, int], rmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of lags with |r| <= rmax + 0.5 and their integer radius bin."""
    h, w = shape
    dy = np.arange(h) - h // 2
    dx = np.arange(w) - w // 2
    r = np.hypot(dy[:, None], dx[None, :])
    bins = np.rint(r).astype(int)
    mask = bins <= rmax
    return np.flatnonzero(mask), bins[mask]


def radial_average(maps: np.ndarray, rmax: int) -> np.ndarray:
    """Average centred S2 maps (..., H, W) over lag vectors with round(|r|) = 0..rmax."""
    h, w = maps.shape[-2:]
    if rmax > min(h, w) // 2:
        raise ValueError(f"rmax={rmax} exceeds the available lag range {min(h, w) // 2}")
    idx, bins = _radial_bins((h, w), rmax)
    onehot = np.zeros((idx.size, rmax + 1))
    onehot[np.arange(idx.size), bins] = 1.0
    flat = maps.reshape(*maps.shape[:-2], h * w)[..., idx]
    return (flat @ onehot) / onehot.sum(axis=0)


def s2_radial(images: np.ndarray, rmax: int, phase: int = 1, periodic: bool = False) -> np.ndarray:
    """Radially averaged S2(r), r = 0..rmax, for one image (rmax+1,) or a batch (N, rmax+1)."""
    images = np.asarray(images)
    if min(images.shape[-2:]) <= rmax:
        raise ValueError(f"Image {images.shape[-2:]} too small for rmax={rmax}")
    return radial_average(s2_map(images, phase, periodic), rmax)


def volume_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    """All 2D slices of a (Z, Y, X) volume, stacked per orientation as (N, H, W)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    return {
        "xy": volume,                         # fixed z
        "xz": np.moveaxis(volume, 1, 0),      # fixed y
        "yz": np.moveaxis(volume, 2, 0),      # fixed x
    }


def s2_volume(volume: np.ndarray, rmax: int, phase: int = 1) -> dict[str, np.ndarray]:
    """Mean S2(r) over the slices of each orientation, plus the mean over the three orientations."""
    curves = {
        name: s2_radial(slices, rmax, phase).mean(axis=0)
        for name, slices in volume_slices(volume).items()
    }
    curves["mean"] = np.mean([curves[o] for o in ORIENTATIONS], axis=0)
    return curves


def s2_mae(s2_a: np.ndarray, s2_b: np.ndarray) -> float:
    """Mean absolute error between two S2(r) curves over r = 0..rmax."""
    s2_a, s2_b = np.asarray(s2_a), np.asarray(s2_b)
    if s2_a.shape != s2_b.shape:
        raise ValueError(f"S2 curves differ in shape: {s2_a.shape} vs {s2_b.shape}")
    return float(np.abs(s2_a - s2_b).mean())


def describe_volumes(
    volumes: list[np.ndarray], image_2d: np.ndarray, rmax: int, phase: int = 1
) -> dict:
    """Compare generated volumes against the 2D training image: phi stats, |dphi| and S2 MAE."""
    if not volumes:
        raise ValueError("No volumes to describe")
    phi_ref = phase_fraction(image_2d, phase)
    s2_ref = s2_radial(image_2d, rmax, phase)

    phis = np.array([phase_fraction(v, phase) for v in volumes])
    s2_curves = np.stack([s2_volume(v, rmax, phase)["mean"] for v in volumes])
    s2_mean = s2_curves.mean(axis=0)
    per_volume_mae = np.array([s2_mae(c, s2_ref) for c in s2_curves])
    return {
        "n_volumes": len(volumes),
        "phi_train": phi_ref,
        "phi_mean": float(phis.mean()),
        "phi_std": float(phis.std()),
        "abs_dphi": float(abs(phis.mean() - phi_ref)),
        "s2_mae": s2_mae(s2_mean, s2_ref),
        "s2_mae_std": float(per_volume_mae.std()),
        "s2_train": s2_ref,
        "s2_generated": s2_mean,
    }
