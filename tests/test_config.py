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
        assert cfg["train"]["fake_slices"] == 64  # same critic budget for all models


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
