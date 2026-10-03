"""Figures and tables: volume slices, descriptor curves, qualitative panel, training curves, metrics.csv, pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

# Categorical slots 1-3 of the validated reference palette (blue, orange, aqua) + ink for the reference.
MODEL_STYLE = {
    "m1_cnn": {"label": "M1 CNN", "color": "#2a78d6", "ls": "-", "marker": "o"},
    "m2_swin": {"label": "M2 Swin", "color": "#eb6834", "ls": "--", "marker": "s"},
    "m3_swin_sam": {"label": "M3 Swin+SAM", "color": "#1baf7a", "ls": "-.", "marker": "^"},
    "m4_ensemble": {"label": "M4 CNN+Swin", "color": "#eda100", "ls": "-", "marker": "D"},
    "m5_finetune": {"label": "M5 M1+Swin fine-tune", "color": "#e87ba4", "ls": "--", "marker": "v"},
    "m1_extended": {"label": "M1 extended", "color": "#008300", "ls": ":", "marker": "P"},
}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dc"

METRIC_COLUMNS = [
    "dataset", "model", "reference", "n_volumes", "phi_ref", "phi_mean", "phi_std", "dphi",
    "s2_mae", "s2_mae_std", "s2_err", "L_mae", "L_err",
    "phi_xy", "phi_xz", "phi_yz", "phi_slice_std_xy", "phi_slice_std_xz", "phi_slice_std_yz",
    "s2_mae_xy", "s2_mae_xz", "s2_mae_yz", "L_mae_xy", "L_mae_xz", "L_mae_yz",
    "dphi_lo", "dphi_hi", "s2_mae_lo", "s2_mae_hi", "L_mae_lo", "L_mae_hi",  # 95 % bootstrap CIs
    # selection-free "typical quality": held-out S2 MAE / phi of every epoch in the late part of training
    "late_epochs", "late_s2_median", "late_s2_q25", "late_s2_q75", "late_phi_median", "late_collapsed",
]


def late_training_summary(run_dir: Path, fraction: float = 0.5) -> dict:
    """Median / IQR of the per-epoch held-out scores over the last `fraction` of epochs (no selection).

    Each epoch's held-out score is unbiased on its own; only taking the best epoch is optimistic
    (winner's curse), so this summarizes a model's typical quality and its stability.
    `late_collapsed` counts epochs whose held-out phi is below 0.05 (near-empty volumes).
    """
    path = run_dir / "selection.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    n_late = max(1, int(round(len(rows) * fraction)))
    late = rows[-n_late:]
    s2 = np.array([float(r["val_s2_mae"]) for r in late])
    phi = np.array([float(r["val_phi"]) for r in late])
    return {"late_epochs": f"{late[0]['epoch']}-{late[-1]['epoch']}", "late_s2_median": float(np.median(s2)),
            "late_s2_q25": float(np.percentile(s2, 25)), "late_s2_q75": float(np.percentile(s2, 75)),
            "late_phi_median": float(np.median(phi)), "late_collapsed": int((phi < 0.05).sum())}

COMPARISON_COLUMNS = ["dataset", "reference", "model", "baseline", "metric", "model_value", "baseline_value",
                      "diff", "diff_lo", "diff_hi", "verdict"]


def bootstrap_from_curves(run_dir: Path, reference: str, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Bootstrap distributions for one run/reference from curves.npz (None if per-volume data is missing)."""
    from src.features.descriptors import bootstrap_metrics

    curves = np.load(run_dir / "curves.npz")
    key = f"{reference}_phi_per_volume"
    if key not in curves:
        return None
    l_vol = curves.get(f"{reference}_L_per_volume")
    l_ref = curves.get(f"{reference}_L_train")
    return bootstrap_metrics(curves[key], curves[f"{reference}_s2_per_volume"], curves[f"{reference}_s2_train"],
                             l_vol, l_ref, n_boot=n_boot, seed=seed)


def _style_axes(ax) -> None:
    """Recessive axes: light grid, no top/right spines, muted tick labels."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_volume_slices(volume: np.ndarray, path: str | Path, title: str | None = None) -> None:
    """Save the central xy / xz / yz slices of a (Z, Y, X) label volume side by side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z, y, x = (s // 2 for s in volume.shape)
    panels = {"xy": volume[z], "xz": volume[:, y, :], "yz": volume[:, :, x]}

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (name, img) in zip(axes, panels.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- metrics table


def collect_metrics(run_dirs: list[Path]) -> list[dict]:
    """Flatten each run's metrics.yaml into one row per (model, reference)."""
    rows = []
    for run_dir in run_dirs:
        try:
            summary = yaml.safe_load((run_dir / "metrics.yaml").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{run_dir / 'metrics.yaml'} missing; run generate first.") from exc
        for ref, m in summary["references"].items():
            row = {"dataset": summary["dataset"], "model": summary["model"], "reference": ref,
                   "phi_ref": m["phi_train"], "dphi": m["abs_dphi"]}
            row.update({k: m.get(k) for k in METRIC_COLUMNS if k not in row and k in m})
            from src.features.descriptors import confidence_interval

            boot = bootstrap_from_curves(run_dir, ref)
            if boot is not None:
                for metric, col in (("abs_dphi", "dphi"), ("s2_mae", "s2_mae"), ("L_mae", "L_mae")):
                    if metric in boot:
                        row[f"{col}_lo"], row[f"{col}_hi"] = confidence_interval(boot[metric])
            row.update(late_training_summary(run_dir))
            rows.append(row)
    return rows


def write_metrics_csv(rows: list[dict], path: str | Path) -> None:
    """Upsert rows into metrics.csv: rows of the same dataset are replaced, others kept."""
    path = Path(path)
    datasets = {r["dataset"] for r in rows}
    kept = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r["dataset"] not in datasets]
    order = {m: i for i, m in enumerate(MODEL_STYLE)}
    all_rows = sorted(kept + rows, key=lambda r: (r["dataset"], r["reference"], order.get(r["model"], 99)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_comparison(run_dirs: list[Path], path: str | Path, baseline: str = "m1_cnn",
                     reference: str = "common") -> list[dict]:
    """Bootstrap difference (model - baseline) of |dphi|, S2 MAE and L MAE with 95 % CIs.

    Volumes of different models are independent samples, so each model is resampled independently;
    a difference whose CI excludes 0 is reported as better / worse, otherwise as no clear difference.
    """
    from src.features.descriptors import confidence_interval

    base_dir = next((d for d in run_dirs if d.name == baseline), None)
    if base_dir is None:
        return []
    base = bootstrap_from_curves(base_dir, reference, seed=1)
    if base is None:
        return []
    dataset = yaml.safe_load((base_dir / "metrics.yaml").read_text(encoding="utf-8"))["dataset"]
    rows = []
    for i, run_dir in enumerate(run_dirs):
        if run_dir.name == baseline:
            continue
        boot = bootstrap_from_curves(run_dir, reference, seed=100 + i)
        if boot is None:
            continue
        for metric in ("abs_dphi", "s2_mae", "L_mae"):
            if metric not in boot or metric not in base:
                continue
            diff = boot[metric] - base[metric]
            lo, hi = confidence_interval(diff)
            verdict = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
            rows.append({"dataset": dataset, "reference": reference, "model": run_dir.name,
                         "baseline": baseline, "metric": metric,
                         "model_value": float(np.median(boot[metric])),
                         "baseline_value": float(np.median(base[metric])),
                         "diff": float(np.median(diff)), "diff_lo": lo, "diff_hi": hi, "verdict": verdict})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, COMPARISON_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


# --------------------------------------------------------------------------- descriptor curves


def plot_descriptor_curves(run_dirs: list[Path], dataset: str, path: str | Path, reference: str = "common") -> None:
    """S2(r) and L(r): 2D reference vs the mean over generated volumes, one colour per model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ref_drawn = False
    for run_dir in run_dirs:
        curves = np.load(run_dir / "curves.npz")
        style = MODEL_STYLE.get(run_dir.name, {"label": run_dir.name, "color": INK_MUTED, "ls": ":", "marker": "x"})
        r = curves["r"]
        for ax, key in zip(axes, ("s2", "L")):
            if f"{reference}_{key}_generated" not in curves:
                continue
            if not ref_drawn:
                ax.plot(r, curves[f"{reference}_{key}_train"], color=INK, lw=2.4, label="2D reference")
            ax.plot(r, curves[f"{reference}_{key}_generated"], color=style["color"], ls=style["ls"], lw=2,
                    marker=style["marker"], markevery=4, ms=5, label=style["label"])
        ref_drawn = True
    ref_label = "ground-truth mask" if dataset == "synthetic" else "Otsu label map"
    for ax, name in zip(axes, ("Two-point correlation $S_2(r)$", "Lineal path $L(r)$")):
        _style_axes(ax)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("r [px]", color=INK_MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset}: generated 64³ volumes (mean over slices and seeds) vs 2D {ref_label}",
                 fontsize=11, color=INK)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(run_dirs: list[Path], dataset: str, path: str | Path) -> None:
    """Critic Wasserstein estimate per model as small multiples (CNN and Swin critics have different scales)."""
    fig, axes = plt.subplots(1, len(run_dirs), figsize=(4 * len(run_dirs), 3.2), squeeze=False)
    for ax, run_dir in zip(axes[0], run_dirs):
        style = MODEL_STYLE.get(run_dir.name, {"label": run_dir.name, "color": INK_MUTED})
        with (run_dir / "history.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        step = np.array([float(r["g_step"]) for r in rows])
        wass = np.array([float(r["wasserstein"]) for r in rows])
        k = max(1, len(wass) // 50)
        smooth = np.convolve(wass, np.ones(k) / k, mode="valid")
        ax.plot(step, wass, color=style["color"], alpha=0.25, lw=1)
        ax.plot(step[k - 1 :], smooth, color=style["color"], lw=2)
        _style_axes(ax)
        ax.set_title(style["label"], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("generator step", color=INK_MUTED)
    axes[0][0].set_ylabel("Wasserstein estimate", color=INK_MUTED)
    fig.suptitle(f"{dataset}: critic Wasserstein estimate during training", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- qualitative panel


def _isosurface(ax, volume: np.ndarray, color: str) -> None:
    """Draw the inclusion-phase isosurface of a label volume on a 3D axis."""
    from skimage.measure import marching_cubes

    ax.axis("off")
    ax.set_box_aspect((1, 1, 1))
    if volume.min() == volume.max():  # single-phase volume: no surface to draw
        ax.text2D(0.5, 0.5, "single phase", transform=ax.transAxes, ha="center", color=INK_MUTED)
        return
    padded = np.pad(volume.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, 0.5, step_size=2)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=color, linewidth=0, shade=True)
    n = volume.shape[0] + 2
    ax.set_xlim(0, n), ax.set_ylim(0, n), ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def plot_qualitative_panel(cfgs: list[dict], path: str | Path, crop: int = 64) -> None:
    """MicroLib Fig. 2 style: training input | SAM overlay (M3) | xy, xz, yz slices | 3D isosurface."""
    import tifffile

    n = len(cfgs)
    fig = plt.figure(figsize=(15, 2.8 * n))
    for i, cfg in enumerate(cfgs):
        run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
        style = MODEL_STYLE.get(cfg["run_name"], {"label": cfg["run_name"], "color": INK_MUTED})
        data = cfg["data"]
        train = np.asarray(Image.open(Path(data["train_dirs"][data["branch"]]) / "image.png"))[:crop, :crop]
        vol_path = sorted((run_dir / "volumes").glob("*.tif"))[0]
        vol = (tifffile.imread(vol_path) > 127).astype(np.uint8)
        z, y, x = (s // 2 for s in vol.shape)
        panels = [(f"training input ({crop}×{crop})", train)]
        overlay = Path(cfg["paths"]["interim"]) / "sam_overlay.png"
        panels.append(("SAM overlay", np.asarray(Image.open(overlay))[:crop, :crop])
                      if data["branch"] == "sam" and overlay.exists() else ("SAM overlay", None))
        panels += [("xy", vol[z]), ("xz", vol[:, y, :]), ("yz", vol[:, :, x])]
        for j, (title, img) in enumerate(panels):
            ax = fig.add_subplot(n, 6, i * 6 + j + 1)
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "n/a (no SAM)", ha="center", va="center", color=INK_MUTED, fontsize=9)
            else:
                ax.imshow(img, cmap=None if img.ndim == 3 else "gray", interpolation="nearest")
            if i == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if j == 0:
                ax.text(-0.08, 0.5, style["label"], transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=11, color=INK)
        ax3d = fig.add_subplot(n, 6, i * 6 + 6, projection="3d")
        _isosurface(ax3d, vol, style["color"])
        if i == 0:
            ax3d.set_title("64³ volume (inclusions)", fontsize=10, color=INK)
    fig.suptitle(f"{cfgs[0]['data']['name']}: 2D input and generated volumes", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- pipeline diagram


def plot_pipeline(path: str | Path) -> None:
    """Architecture diagram of the data -> SAM -> SliceGAN (G + slicer + CNN/Swin critic) -> metrics flow."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13), ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, face="#f4f3ee", edge=INK_MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)
        return (x, y, w, h)

    def arrow(p, q, text=None, color=INK_MUTED):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=8, color=INK_MUTED)

    blue, orange, aqua = (MODEL_STYLE[m]["color"] for m in ("m1_cnn", "m2_swin", "m3_swin_sam"))
    box(0.2, 2.3, 1.8, 0.9, "2D micrograph\n(MicroLib / synthetic)", bold=True)
    box(2.6, 3.4, 1.9, 0.8, "Otsu threshold\n→ label map", edge=blue)
    box(2.6, 1.3, 1.9, 0.8, "SAM ViT-B (zero-shot)\ntiles + merge + gray rule", edge=aqua)
    box(5.1, 2.3, 1.8, 0.9, "random 64×64 crops\n(rot90 / flip)")
    box(5.1, 4.1, 1.8, 0.8, "latent z\n32×4×4×4")
    box(7.5, 4.1, 2.0, 0.8, "SliceGAN 3D generator\n(unchanged, all models)", bold=True)
    box(7.5, 2.9, 2.0, 0.8, "slicer: all 64 slices\nper axis (x, y, z)")
    box(7.5, 0.95, 2.0, 1.5, "2D critic\nM1: CNN (WGAN-GP)\nM2/M3: Swin-T + DiffAug\nM4/M5: CNN + frozen Swin", bold=True, edge=orange)
    box(10.3, 4.1, 2.4, 0.8, "64³ volumes (N = 128 seeds)")
    box(10.3, 2.6, 2.4, 1.0, "metrics vs 2D reference\nφ, S₂(r), L(r), per plane\n+ 95 % bootstrap CIs")
    box(10.3, 1.1, 2.4, 0.9, "SAM IoU / Dice\n(vs GT / annotation)")

    arrow((2.0, 2.9), (2.6, 3.7))
    arrow((2.0, 2.6), (2.6, 1.7))
    arrow((4.5, 3.7), (5.1, 2.95), "M1, M2, M4, M5")
    arrow((4.5, 1.7), (5.1, 2.55), "M3")
    arrow((6.9, 4.5), (7.5, 4.5))
    arrow((8.5, 4.1), (8.5, 3.7))
    arrow((8.5, 2.9), (8.5, 2.3), "fake slices")
    arrow((6.9, 2.6), (7.5, 1.9))
    ax.text(7.05, 2.05, "real\ncrops", ha="right", fontsize=8, color=INK_MUTED)
    ax.add_patch(FancyArrowPatch((7.5, 1.4), (7.1, 4.3), connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                                 mutation_scale=12, color=orange, linewidth=1.2, linestyle="--"))
    ax.text(6.35, 3.35, "critic score\n→ G update", fontsize=8, color=orange, ha="center")
    arrow((9.5, 4.5), (10.3, 4.5))
    arrow((11.5, 4.1), (11.5, 3.6))
    # SAM evaluation path routed below the critic
    ax.plot([3.55, 3.55, 11.5], [1.3, 0.55, 0.55], color=aqua, linewidth=1.2)
    arrow((11.5, 0.55), (11.5, 1.1), None, aqua)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def main() -> None:
    """Aggregate metrics and draw the figures for one dataset from already-generated runs."""
    from src.utils import get_logger, load_config, setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml")
    args = parser.parse_args()

    cfgs = [load_config(c, args.data) for c in args.configs]
    setup_logging(cfgs[0]["paths"]["logs"], "visualize", cfgs[0]["logging"]["level"])
    log = get_logger(__name__)
    dataset = cfgs[0]["data"]["name"]
    run_dirs = [Path(c["paths"]["models"]) / c["run_name"] for c in cfgs]
    fig_dir = Path(cfgs[0]["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        write_metrics_csv(collect_metrics(run_dirs), Path(cfgs[0]["paths"]["reports"]) / "metrics.csv")
        write_comparison(run_dirs, Path(cfgs[0]["paths"]["reports"]) / f"comparison_vs_m1_{dataset}.csv")
        plot_descriptor_curves(run_dirs, dataset, fig_dir / f"{dataset}_descriptors.png")
        plot_training_curves(run_dirs, dataset, fig_dir / f"{dataset}_training.png")
        plot_qualitative_panel(cfgs, fig_dir / f"{dataset}_qualitative.png")
        plot_pipeline(fig_dir / "pipeline.png")
    except (FileNotFoundError, KeyError, IndexError) as exc:
        log.exception("Visualization failed: %s", exc)
        raise SystemExit(1) from exc
    log.info("Wrote reports/metrics.csv and figures for %s -> %s", dataset, fig_dir)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- stabilization probes


def plot_selection_curves(runs: list[dict], path: str | Path, phi_target: float, title: str) -> None:
    """Per-epoch held-out phi and S2 MAE (selection.csv) of several training runs, side by side.

    `runs`: dicts with keys label, csv, color, ls, marker. The phi panel marks the training-image phi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for run in runs:
        with Path(run["csv"]).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ep = [int(r["epoch"]) for r in rows]
        for ax, key in zip(axes, ("val_phi", "val_s2_mae")):
            ax.plot(ep, [float(r[key]) for r in rows], color=run["color"], ls=run["ls"], lw=2,
                    marker=run["marker"], ms=5, label=run["label"])
    axes[0].axhline(phi_target, color=INK, lw=1.2, ls=":", label=f"training image φ = {phi_target:.3f}")
    axes[1].set_yscale("log")
    for ax, name in zip(axes, ("Held-out φ", "Held-out S₂ MAE (log)")):
        _style_axes(ax)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("epoch (100 generator steps)", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
