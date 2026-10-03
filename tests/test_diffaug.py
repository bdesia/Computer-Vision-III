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


def _swin_backbone():
    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, img_size=64,
                             num_classes=1, drop_path_rate=0.0)


def test_multiscale_patch_head_trains_only_heads_and_has_input_gradient():
    critic = SwinCritic(_swin_backbone(), 2, 64, head="multiscale_patch", feature_stages=(2, 3, 4))
    critic.freeze_backbone()
    trainable = {n: p.numel() for n, p in critic.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("patch_heads") for n in trainable)
    assert 170_000 <= sum(trainable.values()) <= 180_000          # ~0.17-0.18 M (probe A: 0.35 M)
    assert not any(p.requires_grad for p in critic.backbone.parameters())  # incl. norm and timm head
    maps = critic.score_maps(_onehot(2))
    assert [tuple(m.shape) for m in maps] == [(2, 1, 8, 8), (2, 1, 4, 4), (2, 1, 2, 2)]

    x = _onehot(3).requires_grad_(True)
    out = critic(x)
    assert out.shape == (3, 1)
    grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    assert grad.abs().sum() > 0
    ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean().backward()   # GP double backward
    assert critic.patch_heads[0].conv1.weight.grad is not None


def test_multiscale_patch_single_scale_is_mean_of_its_score_map():
    critic = SwinCritic(_swin_backbone(), 2, 64, head="multiscale_patch", feature_stages=(2,))
    x = _onehot(4)
    with torch.no_grad():
        score_map = critic.score_maps(x)[0]
        assert torch.allclose(critic(x), score_map.mean(dim=(2, 3)), atol=1e-6)


def test_linear_and_multiscale_heads_still_build_as_before():
    lin = SwinCritic(_swin_backbone(), 2, 64)
    lin.freeze_stages([3, 4])
    assert not hasattr(lin, "patch_heads") and not hasattr(lin, "ms_head")
    assert lin(_onehot(2)).shape == (2, 1)
    ms = SwinCritic(_swin_backbone(), 2, 64, head="multiscale", head_hidden=256)
    ms.freeze_backbone()
    n = sum(p.numel() for p in ms.parameters() if p.requires_grad)
    assert n == 2 * (192 + 384 + 768) + (1344 * 256 + 256) + (256 + 1)   # probe A head, unchanged
    assert ms(_onehot(2)).shape == (2, 1)
    with pytest.raises(ValueError):
        SwinCritic(_swin_backbone(), 2, 64, head="patch")


def test_forward_per_scale_matches_forward_and_head_sn_applies():
    from src.models.discriminator_swin import apply_head_spectral_norm

    critic = SwinCritic(_swin_backbone(), 2, 64, head="multiscale_patch", feature_stages=(2, 3, 4))
    critic.freeze_backbone()
    x = _onehot(3)
    with torch.no_grad():
        per_scale = critic.forward_per_scale(x)
        assert per_scale.shape == (3, 3)
        assert torch.allclose(per_scale.mean(dim=1, keepdim=True), critic(x), atol=1e-6)
    assert apply_head_spectral_norm(critic) == 6                   # 2 convs x 3 scales
    critic.train()
    w = critic.patch_heads[0].conv1.weight                         # parametrized weight, sigma -> 1
    for _ in range(30):
        with torch.no_grad():
            critic(x)
    sigma = torch.linalg.matrix_norm(w.detach().flatten(1), ord=2)
    assert abs(float(sigma) - 1.0) < 0.05
    # pooled multiscale head: 2 linears; linear head: none
    ms = SwinCritic(_swin_backbone(), 2, 64, head="multiscale")
    assert apply_head_spectral_norm(ms) == 2
    assert apply_head_spectral_norm(SwinCritic(_swin_backbone(), 2, 64)) == 0
