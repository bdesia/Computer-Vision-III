"""Shape tests for the upstream SliceGAN networks, slicing helpers and the crop sampler."""

import numpy as np
import pytest
import torch

from src.models.discriminator_cnn import build_cnn_discriminator
from src.models.slicegan_wrapper import (
    SLICE_PERMUTATIONS,
    RandomCropSampler,
    build_generator,
    sample_noise,
    to_labels,
    volume_to_slices,
)

CFG = {"n_phases": 2, "z_channels": 32}


def test_generator_outputs_64_cube_softmax(tmp_path):
    netG = build_generator(CFG, tmp_path).eval()
    with torch.no_grad():
        out = netG(sample_noise(1, 32, "cpu"))
    assert out.shape == (1, 2, 64, 64, 64)
    assert torch.allclose(out.sum(dim=1), torch.ones(1, 64, 64, 64), atol=1e-5)
    labels = to_labels(out)
    assert labels.shape == (1, 64, 64, 64) and labels.dtype == np.uint8
    assert (tmp_path / "slicegan_params.data").exists()


def test_cnn_discriminator_gives_one_logit_per_slice(tmp_path):
    netD = build_cnn_discriminator(CFG, tmp_path)
    assert netD(torch.rand(5, 2, 64, 64)).view(-1).shape == (5,)


def test_volume_to_slices_matches_indexing():
    vol = torch.arange(2 * 1 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4, 4)
    for axis, perm in enumerate(SLICE_PERMUTATIONS):
        slices = volume_to_slices(vol, perm)
        assert slices.shape == (8, 1, 4, 4)
        # slice k of volume 0 along `axis`
        expected = vol[0, 0].select(axis, 1)
        assert torch.equal(slices[1, 0], expected)


def test_random_crop_sampler_one_hot():
    labels = np.zeros((80, 80), dtype=np.uint8)
    labels[:, 40:] = 1
    sampler = RandomCropSampler(labels, crop=64, n_phases=2, device="cpu", seed=0)
    batch = sampler(6)
    assert batch.shape == (6, 2, 64, 64)
    assert torch.equal(batch.sum(dim=1), torch.ones(6, 64, 64))
    with pytest.raises(ValueError):
        RandomCropSampler(labels[:60, :60], crop=64, n_phases=2, device="cpu", seed=0)


def test_random_crop_sampler_augmentation_preserves_phase_counts():
    rng = np.random.default_rng(1)
    labels = (rng.random((100, 100)) < 0.3).astype(np.uint8)
    plain = RandomCropSampler(labels, crop=64, n_phases=2, device="cpu", seed=5)
    aug = RandomCropSampler(labels, crop=64, n_phases=2, device="cpu", seed=5, augment=True)
    a, b = plain(16), aug(16)
    # Same crop positions; rigid rotations/flips keep each crop's phase fraction
    assert torch.allclose(a[:, 1].mean(dim=(1, 2)), b[:, 1].mean(dim=(1, 2)))
    assert not torch.equal(a, b)
