"""Tests for the Swin-T critic: n-phase patch embedding, output shape, gradient penalty and freezing."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminator_swin import SwinCritic, adapt_patch_embedding


def _critic(input_size=64):
    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=False, img_size=input_size, num_classes=1,
        drop_path_rate=0.0,
    )
    return SwinCritic(backbone, n_phases=2, input_size=input_size).eval()


def test_patch_embedding_matches_rgb_on_centred_gray():
    torch.manual_seed(0)
    rgb = nn.Conv2d(3, 8, kernel_size=4, stride=4)
    two_phase = adapt_patch_embedding(rgb, n_phases=2)
    inclusion = (torch.rand(3, 1, 16, 16) > 0.7).float()
    one_hot = torch.cat([1 - inclusion, inclusion], dim=1)
    gray_rgb = (inclusion - 0.5).expand(-1, 3, -1, -1)
    assert torch.allclose(two_phase(one_hot), rgb(gray_rgb), atol=1e-6)
    # Phases must be distinguishable (averaged RGB weights would give identical responses)
    assert not torch.allclose(two_phase(one_hot), two_phase(1 - one_hot))


def test_critic_scores_64px_slices_and_supports_gradient_penalty():
    critic = _critic()
    x = F.one_hot((torch.rand(4, 64, 64) > 0.5).long(), 2).permute(0, 3, 1, 2).float()
    x.requires_grad_(True)
    out = critic(x)
    assert out.shape == (4, 1)
    grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean().backward()  # WGAN-GP double backward
    assert critic.backbone.head.fc.weight.grad is not None


def test_critic_upsamples_when_input_size_differs():
    critic = _critic(input_size=96)
    assert critic(torch.rand(2, 2, 64, 64)).shape == (2, 1)


def test_freeze_stages_keeps_last_stages_and_head_trainable():
    critic = _critic()
    critic.freeze_stages([3, 4])
    b = critic.backbone
    assert not any(p.requires_grad for p in b.patch_embed.parameters())
    assert not any(p.requires_grad for p in b.layers[0].parameters())
    assert not any(p.requires_grad for p in b.layers[1].parameters())
    assert all(p.requires_grad for p in b.layers[2].parameters())
    assert all(p.requires_grad for p in b.layers[3].parameters())
    assert all(p.requires_grad for p in b.head.parameters())
