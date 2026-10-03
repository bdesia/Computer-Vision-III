"""Tests for critic branches: single critic, CNN + Swin ensemble (M4/M5) and generator warm start."""

import copy
from pathlib import Path

import pytest
import torch

from src.models.train import build_critics, train
from src.utils import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(name, tmp_path):
    cfg = copy.deepcopy(load_config(CONFIGS / f"{name}.yaml", [CONFIGS / "data" / "microlib_000210.yaml"]))
    cfg["model"]["swin"]["pretrained"] = False  # no download in tests
    cfg["device"] = "cpu"
    return cfg


def test_single_critic_branch_matches_config(tmp_path):
    (c,) = build_critics(_cfg("m1_cnn", tmp_path), tmp_path, "cpu")
    assert (c.name, c.loss, c.weight, c.diffaug, c.per_scale) == ("cnn", "wgan-gp", 1.0, False, False)
    assert c.opt.param_groups[0]["lr"] == 1e-4


def test_ensemble_has_m1_cnn_branch_and_hinge_swin_branch(tmp_path):
    cnn, swin = build_critics(_cfg("m4_ensemble", tmp_path), tmp_path, "cpu")
    assert (cnn.name, cnn.loss, cnn.weight, cnn.diffaug) == ("cnn", "wgan-gp", 1.0, False)
    x = torch.rand(2, 2, 64, 64)
    assert torch.equal(cnn.augment(x), x)                         # CNN branch sees exactly M1's inputs
    assert (swin.name, swin.loss, swin.weight, swin.diffaug, swin.per_scale) == ("swin", "hinge", 1.0, True, True)
    assert swin.opt.param_groups[0]["lr"] == 1e-4
    assert sum(p.numel() for p in swin.net.parameters() if p.requires_grad) < 200_000  # frozen backbone
    assert not any(p.requires_grad for p in swin.net.backbone.parameters())


def test_finetune_configs_share_init_and_steps():
    m5 = load_config(CONFIGS / "m5_finetune.yaml", [CONFIGS / "data" / "microlib_000210.yaml"])
    ext = load_config(CONFIGS / "m1_extended.yaml", [CONFIGS / "data" / "microlib_000210.yaml"])
    assert m5["train"]["init_generator"] == ext["train"]["init_generator"] == "models/microlib_000210/m1_cnn/G_best.pt"
    assert m5["epochs"] == ext["epochs"] == 20
    assert m5["model"]["discriminator"] == "cnn+swin" and ext["model"]["discriminator"] == "cnn"
    assert m5["run_name"] == "m5_finetune" and ext["run_name"] == "m1_extended"


def test_missing_init_generator_raises(tmp_path):
    cfg = _cfg("m1_extended", tmp_path)
    cfg["paths"].update(models=str(tmp_path / "models"), logs=str(tmp_path / "logs"))
    cfg["train"]["init_generator"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        train(cfg)
