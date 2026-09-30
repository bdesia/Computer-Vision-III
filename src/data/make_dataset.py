"""Build the 2D training data: synthetic RSA micrograph or a real image, thresholded and cropped to 64x64."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from src.utils import get_logger, load_config, set_seed, setup_logging

log = get_logger(__name__)

SOURCES = ("synthetic", "slicegan", "microlib")
TIFF_SUFFIXES = (".tif", ".tiff")


# --------------------------------------------------------------------------- synthetic


def generate_rsa(
    canvas: int,
    phi_target: float,
    radius_min: float,
    radius_max: float,
    min_gap: float,
    max_attempts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Random sequential adsorption of non-overlapping discs until the pixel fraction reaches `phi_target`.

    Returns the binary label map (1 = inclusion) and the number of placed discs.
    """
    if not 0.0 < phi_target < 0.5:
        raise ValueError(f"phi_target must be in (0, 0.5) for disc RSA, got {phi_target}")

    labels = np.zeros((canvas, canvas), dtype=np.uint8)
    centers = np.empty((0, 2))
    radii = np.empty(0)
    target_px = phi_target * canvas * canvas
    filled_px = 0

    for _ in range(max_attempts):
        r = rng.uniform(radius_min, radius_max)
        c = rng.uniform(0, canvas, size=2)
        if radii.size and np.any(np.hypot(*(centers - c).T) < radii + r + min_gap):
            continue

        # Rasterize only the disc bounding box; discs are pixel-disjoint thanks to min_gap
        y0, y1 = max(int(c[0] - r), 0), min(int(c[0] + r) + 2, canvas)
        x0, x1 = max(int(c[1] - r), 0), min(int(c[1] + r) + 2, canvas)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r
        labels[y0:y1, x0:x1][disc] = 1
        filled_px += int(disc.sum())

        centers = np.vstack([centers, c])
        radii = np.append(radii, r)
        if filled_px >= target_px:
            break
    else:
        log.warning(
            "RSA hit max_attempts=%d at phi=%.4f (target %.4f)",
            max_attempts, filled_px / canvas**2, phi_target,
        )
    return labels, int(radii.size)


def render_grayscale(
    labels: np.ndarray,
    intensity_matrix: float,
    intensity_inclusion: float,
    blur_sigma: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a label map as a blurred, noisy grayscale micrograph in [0, 1]."""
    gray = np.where(labels == 1, intensity_inclusion, intensity_matrix).astype(np.float32)
    if blur_sigma > 0:
        gray = gaussian_filter(gray, sigma=blur_sigma)
    if noise_std > 0:
        gray = gray + rng.normal(0.0, noise_std, size=gray.shape).astype(np.float32)
    return np.clip(gray, 0.0, 1.0)


# --------------------------------------------------------------------------- image I/O


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load a 2D image as float32 grayscale in [0, 1] (first page/channel if multi-page or RGB)."""
    path = Path(path)
    try:
        if path.suffix.lower() in TIFF_SUFFIXES:
            arr = tifffile.imread(path)
        else:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("L"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw image not found: {path.resolve()}") from exc
    except Exception as exc:  # corrupt / unsupported file
        raise OSError(f"Could not read image {path}: {exc}") from exc

    arr = np.asarray(arr)
    while arr.ndim > 2:  # multi-page stack or channel axis -> take the first 2D plane
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        raise ValueError(f"Image {path} is constant; cannot segment it")
    return (arr - lo) / (hi - lo)


def save_png(path: Path, img: np.ndarray, binary: bool = False) -> None:
    """Save a [0, 1] float image or a {0, 1} label map as an 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (img * 255).round().astype(np.uint8) if not binary else (img > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------- processing


def binarize(gray: np.ndarray, inclusion_is_minority: bool = True) -> tuple[np.ndarray, float]:
    """Otsu-threshold a grayscale image; label 1 is the minority phase if requested."""
    threshold = float(threshold_otsu(gray))
    labels = (gray > threshold).astype(np.uint8)
    if inclusion_is_minority and labels.mean() > 0.5:
        labels = 1 - labels
    return labels, threshold


def extract_crops(labels: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Sliding-window crops of shape (N, size, size)."""
    if min(labels.shape) < size:
        raise ValueError(f"Image {labels.shape} smaller than crop size {size}")
    windows = np.lib.stride_tricks.sliding_window_view(labels, (size, size))[::stride, ::stride]
    return np.ascontiguousarray(windows.reshape(-1, size, size))


def download_microlib(ml: dict, raw_path: Path, timeout: float = 60.0) -> None:
    """Download a MicroLib/DoITPoMS micrograph, crop away the scale bar and save it as grayscale PNG."""
    url = ml["url"].format(id=ml["id"])
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OSError(
            f"Could not download {url}: {exc}. Download it manually and save the cropped grayscale "
            f"image at {raw_path}."
        ) from exc

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    (raw_path.parent / f"original_{ml['id']}.jpg").write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as im:
        gray = np.asarray(im.convert("L"))
    (r0, r1), (c0, c1) = ml["keep_rows"], ml["keep_cols"]
    cropped = gray[r0:r1, c0:c1]
    Image.fromarray(cropped).save(raw_path)
    with raw_path.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump({**ml, "source": "microlib", "url": url, "original_shape": list(gray.shape),
                   "cropped_shape": list(cropped.shape)}, fh, indent=2)
    log.info("Downloaded %s %s -> cropped %s", url, gray.shape, cropped.shape)


def resolve_raw_image(cfg: dict, rng: np.random.Generator) -> tuple[Path, dict]:
    """Make sure the raw grayscale micrograph exists at `data.raw_path`; return it plus source metadata."""
    data_cfg = cfg["data"]
    source = data_cfg["source"]
    raw_path = Path(data_cfg["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        syn = data_cfg["synthetic"]
        labels, n_discs = generate_rsa(
            canvas=syn["canvas"],
            phi_target=syn["phi_target"],
            radius_min=syn["radius_min"],
            radius_max=syn["radius_max"],
            min_gap=syn["min_gap"],
            max_attempts=syn["max_attempts"],
            rng=rng,
        )
        gray = render_grayscale(
            labels,
            syn["intensity_matrix"],
            syn["intensity_inclusion"],
            syn["blur_sigma"],
            syn["noise_std"],
            rng,
        )
        save_png(raw_path, gray)
        gt_path = raw_path.with_name(f"{raw_path.stem}_gt.png")
        save_png(gt_path, labels, binary=True)
        info = {
            "phi_true": float(labels.mean()),
            "phi_target": syn["phi_target"],
            "n_discs": n_discs,
            "seed": cfg["seed"],
            "params": syn,
            "gt_path": gt_path.as_posix(),
        }
        with raw_path.with_suffix(".json").open("w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        log.info("Synthetic RSA: %d discs, phi_true=%.4f -> %s", n_discs, info["phi_true"], raw_path)
        return raw_path, info

    if source == "slicegan":
        example = data_cfg.get("slicegan_example")
        if not example:
            raise ValueError("data.slicegan_example must name a file in external/SliceGAN/Examples/")
        src = Path(cfg["paths"]["slicegan"]) / "Examples" / example
        if not src.exists():
            raise FileNotFoundError(f"{src} not found. Run `make vendor` to fetch SliceGAN.")
        raw_path = raw_path.with_name(src.name)
        shutil.copy2(src, raw_path)
        log.info("Copied SliceGAN example %s -> %s", src, raw_path)
        return raw_path, {"slicegan_example": example}

    if source == "microlib":
        ml = data_cfg.get("microlib")
        if not ml:
            raise ValueError("data.source == microlib needs a data.microlib block (use a configs/data/ overlay)")
        if not raw_path.exists():
            download_microlib(ml, raw_path)
        info = {"microlib_id": ml["id"], "url": ml["url"].format(id=ml["id"]),
                "keep_rows": ml["keep_rows"], "keep_cols": ml["keep_cols"],
                "pixel_size_um": ml.get("pixel_size_um")}
        log.info("MicroLib %s -> %s", ml["id"], raw_path)
        return raw_path, info

    raise ValueError(f"Unknown data.source '{source}'; expected one of {SOURCES}")


def build_dataset(cfg: dict) -> dict:
    """Run the full raw -> interim -> processed pipeline and return the metadata written to meta.yaml."""
    rng = np.random.default_rng(cfg["seed"])
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    raw_path, source_info = resolve_raw_image(cfg, rng)

    # interim: normalized grayscale + thresholded label map
    gray = load_grayscale(raw_path)
    labels, threshold = binarize(gray, data_cfg["inclusion_is_minority"])
    interim = Path(paths["interim"])
    save_png(interim / "micro_2d_gray.png", gray)
    save_png(interim / "micro_2d_otsu.png", labels, binary=True)

    # processed: full label map (SliceGAN samples its own random crops) + fixed crops for inspection/SAM GT
    out_dir = Path(data_cfg["train_dirs"]["raw"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(out_dir / "image.png", labels, binary=True)
    crops = extract_crops(labels, cfg["img_size"], data_cfg["crop_stride"])
    np.save(out_dir / "crops.npy", crops)

    crop_phi = crops.reshape(len(crops), -1).mean(axis=1)
    meta = {
        "source": data_cfg["source"],
        "raw_path": raw_path.as_posix(),
        "shape": list(labels.shape),
        "n_phases": cfg["n_phases"],
        "label_convention": {0: "matrix", 1: "inclusion/pore"},
        "segmentation": "otsu",
        "otsu_threshold": threshold,
        "phi_train": float(labels.mean()),
        "crop_size": cfg["img_size"],
        "crop_stride": data_cfg["crop_stride"],
        "n_crops": int(len(crops)),
        "phi_crops_mean": float(crop_phi.mean()),
        "phi_crops_std": float(crop_phi.std()),
        "seed": cfg["seed"],
    }
    if "phi_true" in source_info:
        meta["phi_true"] = source_info["phi_true"]
        gt = (np.asarray(Image.open(source_info["gt_path"])) > 0).astype(np.uint8)
        inter, union = np.logical_and(gt, labels).sum(), np.logical_or(gt, labels).sum()
        meta["otsu_iou_vs_gt"] = float(inter / union)
    else:
        meta.update(source_info)

    with (out_dir / "meta.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    log.info(
        "Processed: phi_train=%.4f, %d crops of %dx%d -> %s",
        meta["phi_train"], meta["n_crops"], cfg["img_size"], cfg["img_size"], out_dir,
    )
    if "otsu_iou_vs_gt" in meta:
        log.info("Otsu vs synthetic GT: IoU=%.4f", meta["otsu_iou_vs_gt"])
    return meta


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml (repeatable)")
    parser.add_argument("--source", choices=SOURCES, help="override data.source")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    if args.source:
        cfg["data"]["source"] = args.source
    setup_logging(cfg["paths"]["logs"], "make_dataset", cfg["logging"]["level"])
    set_seed(cfg["seed"])
    try:
        build_dataset(cfg)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
