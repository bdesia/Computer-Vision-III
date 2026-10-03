"""Tests for DiffAug ops and the frozen-backbone multiscale Swin critic head."""

import pytest
import timm
import torch
import torch.nn.functional as F

from src.models.diffaug import DiffAugment, build_diffaug, cutout, dihedral, translate
from src.models.discriminator_swin import SwinCritic


def _onehot(n=6, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n, size, size, generator=g) < 0.3).long()
    return F.one_hot(labels, 2).permute(0, 3, 1, 2).float()


def test_translate_and_d4_keep_one_hot_and_shape():
    x = _onehot()
    for op in (lambda t: translate(t, 0.125), dihedral):
        y = op(x)
        assert y.shape == x.shape
        assert torch.equal(y.sum(dim=1), torch.ones(6, 64, 64))  # still one phase per pixel


def test_dihedral_preserves_phase_fraction_per_sample():
    x = _onehot()
    assert torch.allclose(dihedral(x)[:, 1].mean(dim=(1, 2)), x[:, 1].mean(dim=(1, 2)))


def test_cutout_zeros_a_square():
    y = cutout(torch.ones(4, 2, 64, 64), 0.5)
    zero_frac = (y.sum(dim=1) == 0).float().mean(dim=(1, 2))
    assert torch.allclose(zero_frac, torch.full((4,), 0.25))


def test_diffaug_is_differentiable_and_seeded():
    x = _onehot().requires_grad_(True)
    aug_a = DiffAugment(["translation", "cutout", "d4"], seed=1)
    aug_b = DiffAugment(["translation", "cutout", "d4"], seed=1)
    ya, yb = aug_a(x), aug_b(x)
    assert torch.equal(ya, yb)
    ya.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        DiffAugment(["color"])
    assert build_diffaug({"train": {"diffaug": None}, "seed": 0})(x) is x


def test_multiscale_head_trains_only_head_and_supports_gp():
    backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, img_size=64,
                                 num_classes=1, drop_path_rate=0.0)
    critic = SwinCritic(backbone, 2, 64, head="multiscale", feature_stages=(2, 3, 4), head_hidden=32)
    critic.freeze_backbone()
    trainable = {n for n, p in critic.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("ms_head", "stage_norms")) for n in trainable)
    x = _onehot(3).requires_grad_(True)
    out = critic(x)
    assert out.shape == (3, 1)
    grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean().backward()
    assert critic.ms_head[0].weight.grad is not None
    assert all(p.grad is None for p in critic.backbone.parameters())
