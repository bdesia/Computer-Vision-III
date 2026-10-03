"""Microstructure descriptors: phase fraction (phi), two-point correlation S2(r) and lineal path L(r)."""

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
    """Mean absolute error between two descriptor curves (S2 or L) over r = 0..rmax."""
    s2_a, s2_b = np.asarray(s2_a), np.asarray(s2_b)
    if s2_a.shape != s2_b.shape:
        raise ValueError(f"Curves differ in shape: {s2_a.shape} vs {s2_b.shape}")
    return float(np.abs(s2_a - s2_b).mean())


def relative_error(generated: np.ndarray, reference: np.ndarray) -> float:
    """Micro3Diff-style error rate: mean|gen - ref| / mean|ref|."""
    denom = float(np.abs(np.asarray(reference)).mean())
    if denom == 0:
        raise ValueError("Reference curve is identically zero")
    return s2_mae(generated, reference) / denom


def lineal_path_axis(images: np.ndarray, rmax: int, axis: int, phase: int = 1) -> np.ndarray:
    """Lineal path L(r), r = 0..rmax, along one array axis (non-periodic).

    L(r) is the probability that a segment of r + 1 consecutive pixels along `axis` lies entirely in
    `phase`, averaged over all valid segment positions in the array (any number of leading dims).
    """
    x = np.moveaxis(indicator(images, phase), axis, -1).astype(np.int32)
    n = x.shape[-1]
    if n <= rmax:
        raise ValueError(f"Axis length {n} too small for rmax={rmax}")
    cs = np.concatenate([np.zeros((*x.shape[:-1], 1), dtype=np.int32), np.cumsum(x, axis=-1)], axis=-1)
    out = np.empty(rmax + 1)
    for r in range(rmax + 1):
        window = cs[..., r + 1 :] - cs[..., : n - r]  # phase pixels in each (r+1)-pixel segment
        out[r] = float((window == r + 1).mean())
    return out


def lineal_path(images: np.ndarray, rmax: int, phase: int = 1) -> np.ndarray:
    """In-plane lineal path of 2D image(s) (..., H, W): mean of the curves along H and W."""
    return 0.5 * (lineal_path_axis(images, rmax, -2, phase) + lineal_path_axis(images, rmax, -1, phase))


# In-plane array axes of each slice orientation for a (Z, Y, X) volume
_PLANE_AXES = {"xy": (1, 2), "xz": (0, 2), "yz": (0, 1)}


def lineal_path_volume(volume: np.ndarray, rmax: int, phase: int = 1) -> dict[str, np.ndarray]:
    """Lineal path per slice orientation (mean of its two in-plane axes) plus the mean over the three."""
    per_axis = [lineal_path_axis(volume, rmax, a, phase) for a in range(3)]
    curves = {name: 0.5 * (per_axis[a] + per_axis[b]) for name, (a, b) in _PLANE_AXES.items()}
    curves["mean"] = np.mean(per_axis, axis=0)
    return curves


def describe_volumes(
    volumes: list[np.ndarray],
    image_2d: np.ndarray,
    rmax: int,
    phase: int = 1,
    lineal: bool = True,
) -> dict:
    """Compare generated volumes against the 2D training image.

    Returns phi stats and |dphi|; S2 MAE and error rate; lineal-path MAE and error rate; and, as a 3D
    isotropy check, phi, S2 MAE and L MAE per slice orientation. Per-orientation phi averaged over all
    slices equals the volume phi by construction, so `phi_slice_std_<plane>` (spread of slice phi
    along the normal axis) is also returned to expose gradients along one direction.
    """
    if not volumes:
        raise ValueError("No volumes to describe")
    phi_ref = phase_fraction(image_2d, phase)
    s2_ref = s2_radial(image_2d, rmax, phase)

    phis = np.array([phase_fraction(v, phase) for v in volumes])
    s2_by_vol = [s2_volume(v, rmax, phase) for v in volumes]
    s2_mean = {k: np.mean([c[k] for c in s2_by_vol], axis=0) for k in ("xy", "xz", "yz", "mean")}
    per_volume_mae = np.array([s2_mae(c["mean"], s2_ref) for c in s2_by_vol])

    out = {
        "n_volumes": len(volumes),
        "phi_train": phi_ref,
        "phi_mean": float(phis.mean()),
        "phi_std": float(phis.std()),
        "abs_dphi": float(abs(phis.mean() - phi_ref)),
        "s2_mae": s2_mae(s2_mean["mean"], s2_ref),
        "s2_mae_std": float(per_volume_mae.std()),
        "s2_err": relative_error(s2_mean["mean"], s2_ref),
        "s2_train": s2_ref,
        "s2_generated": s2_mean["mean"],
        # per-volume values, for bootstrap confidence intervals over volumes
        "phi_per_volume": phis,
        "s2_per_volume": np.stack([c["mean"] for c in s2_by_vol]),
    }
    for plane, slices_axis in (("xy", 0), ("xz", 1), ("yz", 2)):
        slice_phi = np.stack([indicator(v, phase).mean(axis=tuple(a for a in range(3) if a != slices_axis))
                              for v in volumes])
        out[f"phi_{plane}"] = float(slice_phi.mean())
        out[f"phi_slice_std_{plane}"] = float(slice_phi.std(axis=1).mean())
        out[f"s2_mae_{plane}"] = s2_mae(s2_mean[plane], s2_ref)

    if lineal:
        l_ref = lineal_path(image_2d, rmax, phase)
        l_by_vol = [lineal_path_volume(v, rmax, phase) for v in volumes]
        l_mean = {k: np.mean([c[k] for c in l_by_vol], axis=0) for k in ("xy", "xz", "yz", "mean")}
        out.update({
            "L_mae": s2_mae(l_mean["mean"], l_ref),
            "L_err": relative_error(l_mean["mean"], l_ref),
            "L_train": l_ref,
            "L_generated": l_mean["mean"],
            "L_per_volume": np.stack([c["mean"] for c in l_by_vol]),
        })
        for plane in ("xy", "xz", "yz"):
            out[f"L_mae_{plane}"] = s2_mae(l_mean[plane], l_ref)
    return out


def bootstrap_metrics(
    phis: np.ndarray,
    s2_per_volume: np.ndarray,
    s2_ref: np.ndarray,
    l_per_volume: np.ndarray | None = None,
    l_ref: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Bootstrap distributions of |dphi|, S2 MAE and L MAE, resampling the generated volumes.

    Each replicate draws N volumes with replacement and recomputes the metrics exactly as reported:
    |mean phi - phi_ref| and the MAE of the mean curve vs the reference curve. phi_ref = S2_ref(0).
    """
    phis, s2_per_volume = np.asarray(phis), np.asarray(s2_per_volume)
    n = len(phis)
    if n < 2:
        raise ValueError("Need at least two volumes for a bootstrap")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {
        "abs_dphi": np.abs(phis[idx].mean(axis=1) - float(s2_ref[0])),
        "s2_mae": np.abs(s2_per_volume[idx].mean(axis=1) - s2_ref).mean(axis=-1),
    }
    if l_per_volume is not None and l_ref is not None:
        out["L_mae"] = np.abs(np.asarray(l_per_volume)[idx].mean(axis=1) - l_ref).mean(axis=-1)
    return out


def confidence_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution."""
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(np.asarray(samples), [tail, 100 - tail])
    return float(lo), float(hi)
