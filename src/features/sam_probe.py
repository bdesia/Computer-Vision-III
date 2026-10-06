"""SAM label probe on `synthetic_sam`: Otsu vs SAM (global and local-contrast labeling) against the clean mask.

The image has an illumination ramp larger than the phase contrast and overlapping per-phase noise, so a single
global threshold fails while every disc stays a closed object (configs/data/synthetic_sam.yaml). SAM masks are
generated once and labeled with both rules of `sam_segment`; scores are IoU, Dice and phi error on the whole
image and on the dark (left) and bright (right) halves. Writes reports/sam_probe.json and
reports/figures/sam_probe.png. This is a label-quality check: nothing is trained on the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.make_dataset import load_grayscale  # noqa: E402
from src.features.sam_segment import (  # noqa: E402
    build_generator,
    classify_groups,
    classify_groups_local,
    generate_masks,
    iou_dice,
    load_label_map,
    merge_masks,
)
from src.utils import get_logger, load_config, set_seed, setup_logging  # noqa: E402
from src.visualization.visualize import INK, INK_MUTED  # noqa: E402

log = get_logger(__name__)


def scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """IoU / Dice / phi error on the whole image and on the left and right halves (ramp direction: x)."""
    half = gt.shape[1] // 2
    out = {}
    for name, sl in (("whole", np.s_[:, :]), ("left", np.s_[:, :half]), ("right", np.s_[:, half:])):
        iou, dice = iou_dice(pred[sl], gt[sl])
        out[name] = {"iou": iou, "dice": dice, "phi": float(pred[sl].mean()),
                     "phi_error": float(pred[sl].mean() - gt[sl].mean())}
    return out


def probe(cfg: dict, out_json: Path, out_fig: Path, generator=None) -> dict:
    """Run SAM once, label its masks with both rules and score Otsu / SAM-global / SAM-local vs the clean mask."""
    if cfg["data"]["source"] != "synthetic":
        raise ValueError("The probe needs a synthetic dataset with a clean mask (use configs/data/synthetic_sam.yaml)")
    scfg = cfg["sam"]
    raw = Path(cfg["data"]["raw_path"])
    gray = load_grayscale(raw)
    gt = load_label_map(raw.with_name("micro_2d_gt.png"))
    otsu = load_label_map(Path(cfg["data"]["train_dirs"]["raw"]) / "image.png")

    generator = generator or build_generator(scfg["model_id"], cfg["device"])
    masks = generate_masks(gray, generator, scfg)
    groups = merge_masks(masks, scfg["merge_iou"])
    sam_global, info_g = classify_groups(gray, masks, groups)
    sam_local, info_l = classify_groups_local(gray, masks, groups, scfg.get("ring_width", 3))

    maps = {"otsu": otsu, "sam_global": sam_global, "sam_local": sam_local}
    result = {
        "dataset": cfg["data"]["name"], "phi_true": float(gt.mean()), "n_masks": len(masks), "n_groups": len(groups),
        "n_inclusion_groups": {"sam_global": info_g["n_inclusion_groups"], "sam_local": info_l["n_inclusion_groups"]},
        "ring_width": scfg.get("ring_width", 3),
        "scores": {k: scores(v, gt) for k, v in maps.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, s in result["scores"].items():
        log.info("%-10s IoU %.3f (left %.3f / right %.3f) phi %.3f (true %.3f)", k, s["whole"]["iou"],
                 s["left"]["iou"], s["right"]["iou"], s["whole"]["phi"], result["phi_true"])
    plot_probe(gray, gt, maps, result, out_fig)
    return result


def plot_probe(gray: np.ndarray, gt: np.ndarray, maps: dict, result: dict, path: Path) -> None:
    """Image, clean mask and the three label maps; errors coloured (missed inclusion / false inclusion)."""
    titles = {"otsu": "Otsu (global threshold)", "sam_global": "SAM, global labeling rule",
              "sam_local": "SAM, local-contrast labeling"}
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("synthetic_sam: ramp + noise + rims", fontsize=9.5, color=INK)
    axes[1].imshow(gt, cmap="gray_r", interpolation="nearest")
    axes[1].set_title(f"clean mask (phi {result['phi_true']:.3f})", fontsize=9.5, color=INK)
    for ax, (key, pred) in zip(axes[2:], maps.items()):
        rgb = np.full(gt.shape + (3,), 255, np.uint8)
        rgb[(pred == 1) & (gt == 1)] = (40, 40, 40)          # correct inclusion
        rgb[(pred == 0) & (gt == 1)] = (42, 120, 214)        # missed inclusion
        rgb[(pred == 1) & (gt == 0)] = (235, 104, 52)        # false inclusion
        ax.imshow(rgb, interpolation="nearest")
        s = result["scores"][key]
        ax.set_title(f"{titles[key]}\nIoU {s['whole']['iou']:.3f} (L {s['left']['iou']:.2f} / R {s['right']['iou']:.2f}),"
                     f" phi {s['whole']['phi']:.3f}", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "dark grey: correct inclusion · blue: missed inclusion · orange: false inclusion · "
             "illumination increases from left (L) to right (R)", ha="center", fontsize=9, color=INK_MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_swin_sam.yaml")
    parser.add_argument("--data", action="append", default=["configs/data/synthetic_sam.yaml"])
    parser.add_argument("--out", default="reports/sam_probe.json")
    parser.add_argument("--fig", default="reports/figures/sam_probe.png")
    args = parser.parse_args()
    cfg = load_config(args.config, args.data)
    setup_logging(cfg["paths"]["logs"], "sam_probe", cfg["logging"]["level"])
    set_seed(cfg["seed"])
    probe(cfg, Path(args.out), Path(args.fig))


if __name__ == "__main__":
    main()
