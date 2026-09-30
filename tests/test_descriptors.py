"""Tests for phi and S2 against images with known analytic correlation functions."""

import numpy as np
import pytest

from src.features.descriptors import (
    describe_volumes,
    phase_fraction,
    s2_mae,
    s2_map,
    s2_radial,
    s2_volume,
    volume_slices,
)


def _stripes(size=64, width=4):
    """Horizontal stripes of `width` px, period 2*width (phi = 0.5)."""
    rows = (np.arange(size) // width) % 2
    return np.repeat(rows[:, None], size, axis=1).astype(np.uint8)


def test_phase_fraction_2d_and_3d():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:3] = 1
    assert phase_fraction(img) == pytest.approx(0.3)
    assert phase_fraction(img, phase=0) == pytest.approx(0.7)
    assert phase_fraction(np.stack([img] * 4)) == pytest.approx(0.3)


def test_s2_zero_lag_equals_phi():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) < 0.3).astype(np.uint8)
    for periodic in (False, True):
        s2 = s2_radial(img, rmax=16, periodic=periodic)
        assert s2[0] == pytest.approx(phase_fraction(img))
    assert s2_radial(img, 16, phase=0)[0] == pytest.approx(1 - phase_fraction(img))


def test_s2_uncorrelated_image_decays_to_phi_squared():
    rng = np.random.default_rng(1)
    p = 0.3
    img = (rng.random((256, 256)) < p).astype(np.uint8)
    s2 = s2_radial(img, rmax=32)
    assert np.allclose(s2[1:], p**2, atol=0.01)


def test_s2_nonperiodic_normalization_on_full_image():
    # Every pair of pixels is phase 1, so S2 must be exactly 1 at every lag.
    s2 = s2_map(np.ones((20, 30), dtype=np.uint8))
    assert s2.shape == (39, 59)
    assert np.allclose(s2, 1.0)


def test_s2_stripes_match_analytic_profile():
    width = 4
    img = _stripes(64, width)
    # Periodic estimator is exact; the non-periodic one has an O(1/H) finite-size bias.
    for periodic, atol in ((True, 1e-9), (False, 0.01)):
        m = s2_map(img, periodic=periodic)
        cy, cx = m.shape[0] // 2, m.shape[1] // 2
        # Along the stripes (x): constant phi
        assert np.allclose(m[cy, cx : cx + 20], 0.5)
        # Across the stripes (y): triangle 0.5 * (1 - |dy| / width) for |dy| <= width
        dy = np.arange(width + 1)
        assert np.allclose(m[cy + dy, cx], 0.5 * (1 - dy / width), atol=atol)


def test_s2_batch_matches_single():
    rng = np.random.default_rng(2)
    batch = (rng.random((3, 48, 48)) < 0.4).astype(np.uint8)
    stacked = s2_radial(batch, rmax=10)
    assert stacked.shape == (3, 11)
    for i in range(3):
        assert np.allclose(stacked[i], s2_radial(batch[i], rmax=10))


def test_s2_rejects_too_large_rmax():
    with pytest.raises(ValueError):
        s2_radial(np.zeros((16, 16)), rmax=16)


def test_volume_slices_orientations():
    vol = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    s = volume_slices(vol)
    assert s["xy"].shape == (2, 3, 4)
    assert s["xz"].shape == (3, 2, 4)
    assert s["yz"].shape == (4, 2, 3)
    assert np.array_equal(s["xz"][1], vol[:, 1, :])
    assert np.array_equal(s["yz"][2], vol[:, :, 2])


def test_s2_volume_of_extruded_image():
    # Extruding a 2D image along z: xy slices reproduce the image's S2 exactly.
    rng = np.random.default_rng(3)
    img = (rng.random((32, 32)) < 0.3).astype(np.uint8)
    vol = np.repeat(img[None], 32, axis=0)
    curves = s2_volume(vol, rmax=8)
    assert set(curves) == {"xy", "xz", "yz", "mean"}
    assert np.allclose(curves["xy"], s2_radial(img, 8))


def test_s2_mae():
    a = np.array([0.3, 0.2, 0.1])
    assert s2_mae(a, a) == 0.0
    assert s2_mae(a, a + 0.05) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        s2_mae(a, a[:2])


def test_describe_volumes_on_isotropic_random_volumes():
    rng = np.random.default_rng(4)
    p = 0.25
    img = (rng.random((128, 128)) < p).astype(np.uint8)
    vols = [(rng.random((32, 32, 32)) < p).astype(np.uint8) for _ in range(4)]
    out = describe_volumes(vols, img, rmax=8)
    assert out["n_volumes"] == 4
    assert out["abs_dphi"] < 0.01
    assert out["s2_mae"] < 0.01
    assert out["s2_train"].shape == out["s2_generated"].shape == (9,)
    with pytest.raises(ValueError):
        describe_volumes([], img, rmax=8)
