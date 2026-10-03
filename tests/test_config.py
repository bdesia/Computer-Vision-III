"""Tests for config inheritance and seeding helpers."""

import random
from pathlib import Path

import numpy as np

from src.utils import deep_merge, load_config, set_seed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_deep_merge_overrides_nested_keys_only():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    merged = deep_merge(base, {"b": {"y": 3}})
    assert merged == {"a": 1, "b": {"x": 1, "y": 3}}
    assert base["b"]["y"] == 2  # base untouched


def test_model_configs_inherit_default():
    expected = {
        "m1_cnn": ("cnn", "raw"),
        "m2_swin": ("swin", "raw"),
        "m3_swin_sam": ("swin", "sam"),
    }
    for name, (disc, branch) in expected.items():
        cfg = load_config(CONFIGS / f"{name}.yaml")
        assert cfg["run_name"] == name
        assert cfg["model"]["discriminator"] == disc
        assert cfg["data"]["branch"] == branch
        # Inherited, not redefined
        assert cfg["volume_size"] == 64
        assert cfg["n_phases"] == 2
        assert cfg["model"]["swin"]["backbone"] == "swin_tiny_patch4_window7_224.ms_in1k"
        assert (cfg["train"]["m_d"], cfg["train"]["m_g"]) == (1, 2)  # m_G = 2 m_D, same for all models


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_dataset_overlay_and_data_name_paths():
    synthetic = load_config(CONFIGS / "m1_cnn.yaml")
    assert synthetic["data"]["name"] == "synthetic"
    assert synthetic["data"]["train_dirs"]["raw"] == "data/processed/synthetic/train_2d"
    assert synthetic["paths"]["models"] == "models/synthetic"

    real = load_config(CONFIGS / "m2_swin.yaml", [CONFIGS / "data" / "microlib_000210.yaml"])
    assert real["run_name"] == "m2_swin"
    assert real["model"]["discriminator"] == "swin"  # model settings kept
    assert real["data"]["source"] == "microlib"
    assert real["data"]["train_dirs"]["sam"] == "data/processed/microlib_000210/train_sam"
    assert real["paths"]["logs"] == "logs/microlib_000210"
    assert "{data_name}" not in str(real)


def test_only_swin_critics_use_lower_lr_and_all_select_best():
    lrs = {name: load_config(CONFIGS / f"{name}.yaml")["train"] for name in ("m1_cnn", "m2_swin", "m3_swin_sam")}
    assert lrs["m1_cnn"]["lr_d"] == 1e-4
    assert lrs["m2_swin"]["lr_d"] == lrs["m3_swin_sam"]["lr_d"] == 2e-5
    assert {t["lr_g"] for t in lrs.values()} == {1e-4}  # generator schedule identical
    for name in lrs:
        cfg = load_config(CONFIGS / f"{name}.yaml")
        assert cfg["generate"]["checkpoint"] == "auto"
        assert not set(cfg["train"]["select_seeds"]) & set(cfg["generate"]["seeds"])  # held-out seeds


def test_final_setup_has_vitgan_stabilizers_off():
    cfgs = {n: load_config(CONFIGS / f"{n}.yaml") for n in ("m1_cnn", "m2_swin", "m3_swin_sam")}
    for c in cfgs.values():
        assert c["model"]["swin"]["isn"] is False
        assert c["train"]["betas_d"] is None and c["train"]["ema_decay"] is None
        assert tuple(c["train"]["betas"]) == (0.9, 0.99)  # same optimizer settings everywhere
    assert cfgs["m2_swin"]["model"]["discriminator"] == cfgs["m3_swin_sam"]["model"]["discriminator"] == "swin"


def test_probe_c_config_matches_probe_a_except_head_and_name():
    c = load_config(CONFIGS / "m2_swin_patchheads.yaml", [CONFIGS / "data" / "microlib_000210.yaml"])
    assert c["run_name"] == "m2_swin_patchheads"                      # must not overwrite m2_swin
    assert c["model"]["discriminator"] == "swin" and c["model"]["swin"]["head"] == "multiscale_patch"
    assert c["model"]["swin"]["trainable_stages"] == []
    assert c["train"]["lr_d"] == 1e-4 and c["epochs"] == 12
    assert c["train"]["diffaug"]["policy"] == ["translation", "cutout", "d4"]
    assert c["train"]["select_seeds"] == list(range(1000, 1016)) and c["data"]["name"] == "microlib_000210"
    assert c["data"]["branch"] == "raw" and c["train"]["ema_decay"] is None


def test_default_loss_is_wgan_gp_everywhere():
    for n in ("m1_cnn", "m2_swin", "m3_swin_sam", "m2_swin_patchheads"):
        c = load_config(CONFIGS / f"{n}.yaml")
        assert c["train"]["loss"] == "wgan-gp" and c["model"]["swin"]["head_sn"] is False
