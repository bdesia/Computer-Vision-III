"""Tests for synthetic RSA generation, thresholding, cropping and the full dataset build."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.make_dataset import (
    binarize,
    build_dataset,
    degrade,
    extract_crops,
    generate_rsa,
    load_grayscale,
    render_grayscale,
)
from src.utils import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _rsa(seed=0, canvas=128, phi=0.25):
    return generate_rsa(canvas, phi, 4, 8, 1, 50_000, np.random.default_rng(seed))


def test_rsa_reaches_target_fraction():
    labels, n = _rsa()
    assert labels.dtype == np.uint8 and set(np.unique(labels)) == {0, 1}
    assert n > 0
    assert 0.25 <= labels.mean() < 0.25 + 0.02  # overshoot at most ~one disc


def test_rsa_is_reproducible():
    a, _ = _rsa(seed=7)
    b, _ = _rsa(seed=7)
    c, _ = _rsa(seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rsa_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        generate_rsa(64, 0.8, 4, 8, 1, 100, np.random.default_rng(0))


def test_binarize_recovers_noisy_rendering():
    rng = np.random.default_rng(0)
    labels, _ = _rsa()
    gray = render_grayscale(labels, 0.35, 0.7, 1.5, 0.05, rng)
    pred, _ = binarize(gray)
    iou = np.logical_and(pred, labels).sum() / np.logical_or(pred, labels).sum()
    assert iou > 0.85


def _iou(pred, labels):
    return np.logical_and(pred, labels).sum() / np.logical_or(pred, labels).sum()


def test_ramp_and_overlapping_noise_break_a_global_threshold():
    """synthetic_sam degradations: a single Otsu cut loses most of its IoU (no SAM involved)."""
    labels, _ = _rsa(seed=3)
    syn = {"blur_sigma": 1.5, "ramp_amplitude": 0.5, "noise_std_matrix": 0.08, "noise_std_inclusion": 0.08,
           "rim_depth": 0.25, "rim_width": 1}
    clean = render_grayscale(labels, 0.35, 0.7, 1.5, 0.05, np.random.default_rng(0))
    rendered = render_grayscale(labels, 0.35, 0.7, 1.5, 0.0, np.random.default_rng(0))
    degraded = degrade(rendered, labels, syn, np.random.default_rng(0))
    iou_clean = _iou(binarize(clean)[0], labels)
    iou_degraded = _iou(binarize(degraded)[0], labels)
    assert iou_clean > 0.85 and iou_degraded < 0.6
    # the instance edge stays locally visible: within a column band, inclusions are still brighter than matrix
    band = np.s_[:, 60:68]
    assert degraded[band][labels[band] == 1].mean() > degraded[band][labels[band] == 0].mean() + 0.2


def test_degrade_is_a_no_op_without_degradation_keys():
    labels, _ = _rsa(seed=1)
    gray = render_grayscale(labels, 0.35, 0.7, 1.5, 0.05, np.random.default_rng(0))
    assert np.array_equal(degrade(gray, labels, {"blur_sigma": 1.5}, np.random.default_rng(0)), gray)


def test_binarize_labels_minority_as_inclusion():
    gray = np.zeros((32, 32), dtype=np.float32)
    gray[:8] = 1.0  # bright minority
    assert binarize(1.0 - gray)[0].mean() == pytest.approx(0.25)  # dark minority is still label 1
    assert binarize(gray)[0].mean() == pytest.approx(0.25)


def test_extract_crops_shape():
    crops = extract_crops(np.zeros((128, 96), dtype=np.uint8), size=64, stride=32)
    assert crops.shape == (3 * 2, 64, 64)
    with pytest.raises(ValueError):
        extract_crops(np.zeros((32, 32)), size=64, stride=32)


def test_build_dataset_synthetic(tmp_path):
    cfg = copy.deepcopy(load_config(CONFIGS / "default.yaml"))
    cfg["paths"].update(interim=str(tmp_path / "interim"), logs=str(tmp_path / "logs"))
    cfg["data"]["raw_path"] = str(tmp_path / "raw" / "micro_2d.png")
    cfg["data"]["train_dirs"]["raw"] = str(tmp_path / "processed" / "train_2d")
    cfg["data"]["synthetic"]["canvas"] = 128

    meta = build_dataset(cfg)

    out = tmp_path / "processed" / "train_2d"
    assert (out / "image.png").exists() and (out / "meta.yaml").exists()
    assert np.load(out / "crops.npy").shape == (9, 64, 64)
    assert yaml.safe_load((out / "meta.yaml").read_text())["n_crops"] == 9
    assert abs(meta["phi_train"] - meta["phi_true"]) < 0.03
    assert meta["otsu_iou_vs_gt"] > 0.85
    assert load_grayscale(cfg["data"]["raw_path"]).shape == (128, 128)


def test_download_microlib_crops_scale_bar(tmp_path, monkeypatch):
    import io
    import json

    from PIL import Image

    from src.data import make_dataset

    img = np.full((50, 40), 200, dtype=np.uint8)
    img[45:] = 255  # "scale bar" rows
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(make_dataset.urllib.request, "urlopen", lambda url, timeout: _Resp(buf.getvalue()))
    ml = {"id": "000123", "url": "https://example.org/{id}.jpg", "keep_rows": [0, 45], "keep_cols": [0, 40]}
    raw = tmp_path / "raw" / "micro_2d.png"
    make_dataset.download_microlib(ml, raw)

    assert np.asarray(Image.open(raw)).shape == (45, 40)
    assert (raw.parent / "original_000123.jpg").exists()
    assert json.loads(raw.with_suffix(".json").read_text())["url"] == "https://example.org/000123.jpg"
