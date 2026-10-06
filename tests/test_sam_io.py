"""Tests for SAM front-end logic (no model download): label-map I/O, tiling, mask merge and intensity rule."""

import numpy as np
import pytest

from src.features.sam_segment import (
    weighted_otsu,
    Mask,
    classify_groups,
    classify_groups_local,
    generate_masks,
    gt_crop_origins,
    iou_dice,
    load_label_map,
    mask_iou,
    merge_masks,
    save_label_map,
    tile_origins,
)


def _full(shape, box):
    m = np.zeros(shape, dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_label_map_roundtrip(tmp_path):
    labels = (np.random.default_rng(0).random((40, 30)) > 0.7).astype(np.uint8)
    save_label_map(tmp_path / "sub" / "labels.png", labels)
    loaded = load_label_map(tmp_path / "sub" / "labels.png")
    assert loaded.dtype == np.uint8 and np.array_equal(loaded, labels)
    with pytest.raises(FileNotFoundError):
        load_label_map(tmp_path / "missing.png")


def test_mask_bbox_and_iou():
    a = Mask.from_full(_full((20, 20), (2, 6, 2, 6)))
    b = Mask.from_full(_full((20, 20), (4, 8, 4, 8)))
    assert (a.y0, a.x0, a.y1, a.x1, a.area) == (2, 2, 6, 6, 16)
    assert mask_iou(a, b) == pytest.approx(4 / 28)
    assert mask_iou(a, Mask.from_full(_full((20, 20), (10, 12, 10, 12)))) == 0.0
    assert Mask.from_full(np.zeros((5, 5), dtype=bool)) is None


def test_tile_origins_cover_image():
    assert tile_origins(100, 256, 32) == [0]
    starts = tile_origins(437, 256, 32)
    assert starts[0] == 0 and starts[-1] + 256 == 437
    assert tile_origins(800, 256, 32) == [0, 224, 448, 544]


def test_merge_masks_groups_overlapping_duplicates():
    shape = (30, 30)
    masks = [Mask.from_full(_full(shape, b)) for b in [(0, 10, 0, 10), (1, 10, 0, 10), (20, 25, 20, 25)]]
    groups = sorted(sorted(g) for g in merge_masks(masks, iou_thr=0.3))
    assert groups == [[0, 1], [2]]


def test_classify_groups_intensity_rule():
    gray = np.full((40, 40), 0.8)          # bright matrix (majority)
    gray[5:12, 5:12] = 0.1                 # dark inclusion A
    gray[25:30, 20:28] = 0.15              # dark inclusion B (not covered by any mask)
    masks = [
        Mask.from_full(_full(gray.shape, (5, 12, 5, 12))),    # inclusion A
        Mask.from_full(_full(gray.shape, (15, 40, 0, 15))),   # matrix region
        Mask.from_full(_full(gray.shape, (0, 5, 20, 40))),    # matrix region
    ]
    labels, info = classify_groups(gray, masks, [[0], [1], [2]])
    assert info["n_inclusion_groups"] == 1
    assert labels[5:12, 5:12].all()
    assert labels.sum() == 49              # uncovered dark region B stays matrix (background rule)


def test_classify_groups_dark_matrix_bright_inclusions():
    gray = np.full((30, 30), 0.2)
    gray[:10, :10] = 0.9
    masks = [Mask.from_full(_full(gray.shape, (0, 10, 0, 10))), Mask.from_full(_full(gray.shape, (12, 30, 0, 30)))]
    labels, _ = classify_groups(gray, masks, [[0], [1]])
    assert labels[:10, :10].all() and labels.sum() == 100


def test_local_contrast_labels_both_ends_of_an_illumination_ramp():
    """Bright discs on a ramp: the dark-side disc is darker than the bright-side matrix, so the global split
    misses it, while each disc is brighter than its own surroundings (local rule)."""
    shape = (40, 120)
    ramp = np.linspace(0.0, 0.6, shape[1])[None, :].repeat(shape[0], axis=0)
    gray = 0.2 + ramp
    boxes = [(10, 20, 5, 15), (10, 20, 50, 60), (10, 20, 100, 110)]  # left, middle, right inclusions
    for y0, y1, x0, x1 in boxes:
        gray[y0:y1, x0:x1] += 0.3
    masks = [Mask.from_full(_full(shape, b)) for b in boxes]
    masks.append(Mask.from_full(_full(shape, (25, 40, 60, 120))))  # a matrix region on the bright side
    groups = [[0], [1], [2], [3]]
    global_labels, _ = classify_groups(gray, masks, groups)
    local_labels, info = classify_groups_local(gray, masks, groups, ring_width=3)
    assert not global_labels[10:20, 5:15].any()  # the global rule misses the dark-side inclusion
    assert info["n_inclusion_groups"] == 3
    for y0, y1, x0, x1 in boxes:
        assert local_labels[y0:y1, x0:x1].all()
    assert not local_labels[25:40, 60:120].any()  # matrix region stays matrix


def test_generate_masks_with_fake_generator_maps_tile_offsets():
    calls = []

    def fake_generator(crop, **kwargs):
        calls.append((crop.size, kwargs["points_per_crop"]))
        m = np.zeros((crop.size[1], crop.size[0]), dtype=bool)
        m[:4, :4] = True
        return {"masks": [m]}

    cfg = {"tile_size": 16, "tile_overlap": 4, "points_per_side": 8, "points_per_batch": 4,
           "pred_iou_thresh": 0.8, "stability_score_thresh": 0.85}
    masks = generate_masks(np.random.default_rng(0).random((20, 28)), fake_generator, cfg)
    assert len(calls) == len(masks) == 2 * 2   # tiles at y in {0, 4}, x in {0, 12}
    assert {(m.y0, m.x0) for m in masks} == {(y, x) for y in tile_origins(20, 16, 4) for x in tile_origins(28, 16, 4)}


def test_iou_dice_and_gt_crops():
    a = np.zeros((10, 10), dtype=np.uint8); a[:5] = 1
    b = np.zeros((10, 10), dtype=np.uint8); b[:5, :5] = 1
    assert iou_dice(a, b) == pytest.approx((0.5, 2 * 25 / 75))
    assert iou_dice(np.zeros((3, 3)), np.zeros((3, 3))) == (1.0, 1.0)
    origins = gt_crop_origins((437, 800), n=5, size=64, seed=42)
    assert origins == gt_crop_origins((437, 800), n=5, size=64, seed=42)
    assert len(origins) == 5 and all(0 <= y <= 437 - 64 and 0 <= x <= 800 - 64 for y, x in origins)


def test_weighted_otsu_splits_between_clusters():
    values = np.array([0.1, 0.12, 0.8, 0.82, 0.85])
    t = weighted_otsu(values, np.array([5, 5, 100, 100, 100]))
    assert 0.12 < t < 0.8


def test_annotation_reference_uses_midpoint_and_minority(tmp_path):
    from PIL import Image

    from src.features.sam_segment import annotation_reference

    gray = np.full((20, 20), 200, dtype=np.uint8)
    gray[:5] = 10                                    # dark minority phase (25 %)
    Image.fromarray(gray).save(tmp_path / "raw.png")
    cfg = {"data": {"raw_path": str(tmp_path / "raw.png"), "microlib": {"phases_gray": [8, 133]}}}
    ref = annotation_reference(cfg)
    assert ref.dtype == np.uint8 and ref.mean() == pytest.approx(0.25) and ref[:5].all()
    assert annotation_reference({"data": {"raw_path": "x", "microlib": None}}) is None
