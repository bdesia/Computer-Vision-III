"""Zero-shot SAM phase segmentation (M3 front-end): tiled automatic masks, IoU merge and intensity rule."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from skimage.segmentation import find_boundaries

from src.data.make_dataset import extract_crops, load_grayscale, save_png
from src.utils import get_logger, load_config, set_seed, setup_logging

log = get_logger(__name__)


@dataclass
class Mask:
    """A binary mask stored inside its bounding box [y0:y1, x0:x1] of the full image."""

    y0: int
    x0: int
    m: np.ndarray  # bool, bounding-box sized

    @property
    def y1(self) -> int:
        """Exclusive bottom row."""
        return self.y0 + self.m.shape[0]

    @property
    def x1(self) -> int:
        """Exclusive right column."""
        return self.x0 + self.m.shape[1]

    @property
    def area(self) -> int:
        """Number of pixels in the mask."""
        return int(self.m.sum())

    @classmethod
    def from_full(cls, full: np.ndarray, y_off: int = 0, x_off: int = 0) -> "Mask | None":
        """Crop a full-size (tile) boolean mask to its bounding box; None if empty."""
        rows, cols = np.flatnonzero(full.any(axis=1)), np.flatnonzero(full.any(axis=0))
        if rows.size == 0:
            return None
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return cls(int(y0 + y_off), int(x0 + x_off), full[y0:y1, x0:x1].copy())

    def paint(self, canvas: np.ndarray) -> None:
        """OR the mask into a full-size boolean canvas."""
        canvas[self.y0 : self.y1, self.x0 : self.x1] |= self.m


# --------------------------------------------------------------------------- mask generation


def tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of overlapping tiles covering [0, length)."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def generate_masks(gray: np.ndarray, generator, scfg: dict) -> list[Mask]:
    """Run SAM automatic mask generation on overlapping tiles of a [0, 1] grayscale image."""
    h, w = gray.shape
    tile, overlap = scfg["tile_size"], scfg["tile_overlap"]
    rgb = Image.fromarray((gray * 255).round().astype(np.uint8)).convert("RGB")
    masks: list[Mask] = []
    ys, xs = tile_origins(h, tile, overlap), tile_origins(w, tile, overlap)
    for y in ys:
        for x in xs:
            crop = rgb.crop((x, y, min(x + tile, w), min(y + tile, h)))
            out = generator(
                crop,
                points_per_crop=scfg["points_per_side"],
                points_per_batch=scfg["points_per_batch"],
                pred_iou_thresh=scfg["pred_iou_thresh"],
                stability_score_thresh=scfg["stability_score_thresh"],
            )
            for m in out["masks"]:
                mask = Mask.from_full(np.asarray(m, dtype=bool), y, x)
                if mask is not None:
                    masks.append(mask)
    log.info("SAM: %d masks from %d tiles of %d px (grid %d/side)", len(masks), len(ys) * len(xs),
             tile, scfg["points_per_side"])
    return masks


# --------------------------------------------------------------------------- merge + classify


def mask_iou(a: Mask, b: Mask) -> float:
    """IoU of two bounding-box masks (0 if the boxes do not overlap)."""
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    inter = int(np.logical_and(
        a.m[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0],
        b.m[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0],
    ).sum())
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def merge_masks(masks: list[Mask], iou_thr: float) -> list[list[int]]:
    """Group masks whose pairwise IoU exceeds `iou_thr` (transitively, union-find)."""
    n = len(masks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        boxes = np.array([[m.y0, m.y1, m.x0, m.x1] for m in masks])
        overlap = (
            (boxes[:, None, 0] < boxes[None, :, 1]) & (boxes[None, :, 0] < boxes[:, None, 1])
            & (boxes[:, None, 2] < boxes[None, :, 3]) & (boxes[None, :, 2] < boxes[:, None, 3])
        )
        for i, j in zip(*np.nonzero(np.triu(overlap, k=1))):
            if find(i) != find(j) and mask_iou(masks[i], masks[j]) > iou_thr:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def weighted_otsu(values: np.ndarray, weights: np.ndarray) -> float:
    """Exact weighted Otsu split of 1D values: midpoint between the two sorted values that
    maximize the between-class variance."""
    order = np.argsort(values)
    v, w = values[order].astype(float), weights[order].astype(float)
    w0 = np.cumsum(w)[:-1]
    w1 = w.sum() - w0
    m0 = np.cumsum(w * v)[:-1] / w0
    m1 = ((w * v).sum() - np.cumsum(w * v)[:-1]) / w1
    between = w0 * w1 * (m0 - m1) ** 2
    between[np.diff(v) == 0] = -1.0  # cannot split between equal values
    k = int(np.argmax(between))
    return float(0.5 * (v[k] + v[k + 1]))


def classify_groups(gray: np.ndarray, masks: list[Mask], groups: list[list[int]]) -> tuple[np.ndarray, dict]:
    """Label map from merged masks: groups whose mean gray departs from the matrix are inclusions.

    The matrix gray level is the image median (matrix = majority phase). Group means are split into
    two classes by an exact area-weighted Otsu split; the class on the far side from the matrix is the
    inclusion phase. Pixels outside every inclusion group (incl. uncovered pixels) are matrix.
    """
    shape = gray.shape
    group_masks, means, areas = [], [], []
    for g in groups:
        canvas = np.zeros(shape, dtype=bool)
        for i in g:
            masks[i].paint(canvas)
        group_masks.append(canvas)
        means.append(float(gray[canvas].mean()))
        areas.append(int(canvas.sum()))
    means, areas = np.array(means), np.array(areas)

    matrix_gray = float(np.median(gray))
    info = {"n_groups": len(groups), "matrix_gray": matrix_gray}
    labels = np.zeros(shape, dtype=np.uint8)
    if len(groups) < 2 or np.ptp(means) < 1e-6:
        log.warning("SAM produced fewer than two distinct mask groups; label map is all matrix")
        info.update(threshold=None, n_inclusion_groups=0)
        return labels, info

    threshold = weighted_otsu(means, areas)
    inclusion = (means > threshold) if matrix_gray <= threshold else (means < threshold)
    for is_inc, canvas in zip(inclusion, group_masks):
        if is_inc:
            labels[canvas] = 1
    info.update(threshold=threshold, n_inclusion_groups=int(inclusion.sum()))
    return labels, info


# --------------------------------------------------------------------------- evaluation + I/O


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """IoU and Dice of two binary masks (1.0 if both are empty)."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union, total = (p & g).sum(), (p | g).sum(), p.sum() + g.sum()
    if union == 0:
        return 1.0, 1.0
    return float(inter / union), float(2 * inter / total)


def save_label_map(path: str | Path, labels: np.ndarray) -> None:
    """Save a {0, 1} label map as a 0/255 PNG."""
    save_png(Path(path), labels, binary=True)


def load_label_map(path: str | Path) -> np.ndarray:
    """Load a 0/255 PNG as a {0, 1} uint8 label map."""
    path = Path(path)
    try:
        with Image.open(path) as im:
            return (np.asarray(im.convert("L")) > 127).astype(np.uint8)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Label map not found: {path}") from exc


def gt_crop_origins(shape: tuple[int, int], n: int, size: int, seed: int) -> list[tuple[int, int]]:
    """Deterministic, non-overlapping top-left corners of `n` evaluation crops."""
    rng = np.random.default_rng(seed)
    origins: list[tuple[int, int]] = []
    for _ in range(10_000):
        y, x = int(rng.integers(0, shape[0] - size + 1)), int(rng.integers(0, shape[1] - size + 1))
        if all(abs(y - oy) >= size or abs(x - ox) >= size for oy, ox in origins):
            origins.append((y, x))
            if len(origins) == n:
                return origins
    raise ValueError(f"Could not place {n} non-overlapping {size}px crops in {shape}")


def overlay(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """RGB overlay: grayscale image, inclusions tinted, SAM boundaries in red."""
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    inc = labels.astype(bool)
    rgb[inc] = 0.6 * rgb[inc] + 0.4 * np.array([0, 160, 255])
    rgb[find_boundaries(labels, mode="inner")] = [255, 40, 40]
    return rgb.round().astype(np.uint8)


def evaluate_gt_crops(cfg: dict, gray: np.ndarray, sam: np.ndarray, otsu: np.ndarray) -> dict:
    """Write the evaluation crops and compute IoU/Dice of SAM (and Otsu) wherever a GT crop exists.

    Synthetic data: GT crops are taken from the clean ground-truth mask. Real data: GT crops
    (`crop_XX_gt.png`) must be corrected by hand, starting from the provided `crop_XX_sam.png`.
    """
    scfg, size = cfg["sam"], cfg["img_size"]
    gt_dir = Path(scfg["gt_dir"])
    gt_dir.mkdir(parents=True, exist_ok=True)
    origins = gt_crop_origins(gray.shape, scfg["n_gt_crops"], size, cfg["seed"])

    full_gt = None
    if cfg["data"]["source"] == "synthetic":
        full_gt = load_label_map(Path(cfg["data"]["raw_path"]).with_name("micro_2d_gt.png"))

    rows = []
    for k, (y, x) in enumerate(origins):
        sl = (slice(y, y + size), slice(x, x + size))
        save_png(gt_dir / f"crop_{k:02d}_gray.png", gray[sl])
        save_label_map(gt_dir / f"crop_{k:02d}_sam.png", sam[sl])
        gt_path = gt_dir / f"crop_{k:02d}_gt.png"
        if full_gt is not None:
            save_label_map(gt_path, full_gt[sl])
        if not gt_path.exists():
            continue
        gt = load_label_map(gt_path)
        iou, dice = iou_dice(sam[sl], gt)
        iou_o, dice_o = iou_dice(otsu[sl], gt)
        rows.append({"crop": k, "y": y, "x": x, "sam_iou": iou, "sam_dice": dice,
                     "otsu_iou": iou_o, "otsu_dice": dice_o})

    result = {"crops": [{"crop": k, "y": y, "x": x} for k, (y, x) in enumerate(origins)],
              "gt_source": "synthetic ground truth" if full_gt is not None else "manual correction"}
    if rows:
        result["per_crop"] = rows
        for key in ("sam_iou", "sam_dice", "otsu_iou", "otsu_dice"):
            result[f"mean_{key}"] = float(np.mean([r[key] for r in rows]))
        log.info("GT crops (%d, %s): SAM IoU=%.3f Dice=%.3f | Otsu IoU=%.3f Dice=%.3f", len(rows),
                 result["gt_source"], result["mean_sam_iou"], result["mean_sam_dice"],
                 result["mean_otsu_iou"], result["mean_otsu_dice"])
    else:
        log.warning("No GT crops in %s yet: correct crop_XX_sam.png by hand and save as crop_XX_gt.png, "
                    "then rerun `make sam`.", gt_dir)
    with (gt_dir / "metrics.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result, fh, sort_keys=False)
    return result


# --------------------------------------------------------------------------- pipeline


def annotation_reference(cfg: dict) -> np.ndarray | None:
    """Label map from the dataset's curated phase gray levels (MicroLib `phases_gray`), if available.

    The threshold is the midpoint of the two annotated gray levels, applied to the raw 8-bit micrograph;
    the minority phase is labelled 1. This is a curated reference, not a pixel-accurate ground truth.
    """
    levels = (cfg["data"].get("microlib") or {}).get("phases_gray")
    if not levels:
        return None
    with Image.open(cfg["data"]["raw_path"]) as im:
        gray = np.asarray(im.convert("L")).astype(np.float32)
    labels = (gray > float(np.mean(levels))).astype(np.uint8)
    return labels if labels.mean() <= 0.5 else 1 - labels


def reference_scores(cfg: dict, sam: np.ndarray, otsu: np.ndarray) -> dict:
    """IoU/Dice of SAM and Otsu against the synthetic ground truth and/or the annotation reference."""
    out = {}
    if cfg["data"]["source"] == "synthetic":
        gt = load_label_map(Path(cfg["data"]["raw_path"]).with_name("micro_2d_gt.png"))
        out["synthetic_ground_truth"] = {"sam_iou_dice": list(iou_dice(sam, gt)),
                                         "otsu_iou_dice": list(iou_dice(otsu, gt)), "phi_ref": float(gt.mean())}
    ref = annotation_reference(cfg)
    if ref is not None:
        out["annotation_threshold"] = {"sam_iou_dice": list(iou_dice(sam, ref)),
                                       "otsu_iou_dice": list(iou_dice(otsu, ref)), "phi_ref": float(ref.mean()),
                                       "threshold_gray": float(np.mean(cfg["data"]["microlib"]["phases_gray"]))}
    for name, r in out.items():
        log.info("vs %s (phi %.4f): SAM IoU/Dice %.3f/%.3f | Otsu IoU/Dice %.3f/%.3f", name, r["phi_ref"],
                 *r["sam_iou_dice"], *r["otsu_iou_dice"])
    return out


def update_reference_scores(cfg: dict) -> dict:
    """Recompute reference scores from the saved SAM and Otsu label maps (no SAM inference)."""
    out_dir = Path(cfg["data"]["train_dirs"]["sam"])
    sam = load_label_map(out_dir / "image.png")
    otsu = load_label_map(Path(cfg["data"]["train_dirs"]["raw"]) / "image.png")
    scores = reference_scores(cfg, sam, otsu)
    meta_path = out_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["reference_scores"] = scores
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return scores


def build_generator(model_id: str, device: str):
    """Hugging Face mask-generation pipeline for SAM (no fine-tuning)."""
    import torch
    from transformers import pipeline

    dev = 0 if device.startswith("cuda") and torch.cuda.is_available() else -1
    try:
        return pipeline("mask-generation", model=model_id, device=dev)
    except Exception as exc:  # hub / network / weights
        raise RuntimeError(f"Could not load SAM model '{model_id}': {exc}") from exc


def run(cfg: dict, generator=None) -> dict:
    """Segment the raw micrograph with SAM and write the M3 training label map + diagnostics."""
    scfg = cfg["sam"]
    gray = load_grayscale(cfg["data"]["raw_path"])
    otsu = load_label_map(Path(cfg["data"]["train_dirs"]["raw"]) / "image.png")

    generator = generator or build_generator(scfg["model_id"], cfg["device"])
    masks = generate_masks(gray, generator, scfg)
    groups = merge_masks(masks, scfg["merge_iou"])
    labels, info = classify_groups(gray, masks, groups)

    out_dir = Path(cfg["data"]["train_dirs"]["sam"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_label_map(out_dir / "image.png", labels)
    crops = extract_crops(labels, cfg["img_size"], cfg["data"]["crop_stride"])
    np.save(out_dir / "crops.npy", crops)

    interim = Path(cfg["paths"]["interim"])
    save_label_map(interim / "sam_labels.png", labels)
    Image.fromarray(overlay(gray, labels)).save(interim / "sam_overlay.png")

    iou_otsu, dice_otsu = iou_dice(labels, otsu)
    meta = {
        "source": "sam",
        "model_id": scfg["model_id"],
        "params": {k: scfg[k] for k in ("tile_size", "tile_overlap", "points_per_side",
                                        "pred_iou_thresh", "stability_score_thresh", "merge_iou")},
        "n_masks": len(masks),
        **info,
        "phi_train": float(labels.mean()),
        "phi_otsu": float(otsu.mean()),
        "iou_vs_otsu": iou_otsu,
        "dice_vs_otsu": dice_otsu,
        "n_crops": int(len(crops)),
        "label_convention": {0: "matrix", 1: "inclusion/pore"},
    }
    if cfg["data"]["source"] == "synthetic":
        gt = load_label_map(Path(cfg["data"]["raw_path"]).with_name("micro_2d_gt.png"))
        meta["iou_vs_gt"], meta["dice_vs_gt"] = iou_dice(labels, gt)
        meta["otsu_iou_vs_gt"] = iou_dice(otsu, gt)[0]
    meta["gt_crops"] = {k: v for k, v in evaluate_gt_crops(cfg, gray, labels, otsu).items()
                        if k.startswith("mean_") or k == "gt_source"}
    meta["reference_scores"] = reference_scores(cfg, labels, otsu)

    with (out_dir / "meta.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    log.info("SAM label map: phi=%.4f (Otsu %.4f), IoU vs Otsu=%.3f, %d inclusion groups -> %s",
             meta["phi_train"], meta["phi_otsu"], iou_otsu, info.get("n_inclusion_groups", 0), out_dir)
    return meta


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_swin_sam.yaml")
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml")
    parser.add_argument("--reference-only", action="store_true",
                        help="only recompute IoU/Dice vs the references from the saved label maps")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    setup_logging(cfg["paths"]["logs"], "sam_segment", cfg["logging"]["level"])
    set_seed(cfg["seed"])
    try:
        update_reference_scores(cfg) if args.reference_only else run(cfg)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        log.exception("SAM segmentation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
