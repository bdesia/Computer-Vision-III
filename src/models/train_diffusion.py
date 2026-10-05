"""Train M6, the 2D diffusion transformer, on 64 x 64 crops of the training micrograph.

Called from src.models.train when `model.generator: dit`. Same data, logging, tracking and checkpoint
conventions as the GAN runs: history.csv (per log step), selection.csv / selection.yaml (per epoch, 3D
volumes from held-out seeds scored against the training image), G_best.pt / G_last.pt and half-precision
snapshots (EMA weights), so `src.models.generate` evaluates M6 with the same protocol as M1-M5.
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.features.descriptors import phase_fraction, s2_mae, s2_radial, s2_volume
from src.models.diffusion import build_dit, cosine_alpha_bar, q_sample, sample_slices, sample_volume
from src.models.slicegan_wrapper import RandomCropSampler, load_label_map
from src.models.train import SELECTION_FIELDS, update_ema
from src.tracking import Tracker
from src.utils import get_device, get_logger
from src.visualization.visualize import plot_volume_slices

log = get_logger(__name__)

HISTORY_FIELDS = ("epoch", "g_step", "g_loss", "lr", "sec_per_g_step")  # g_loss = denoising MSE (MLflow backfill)


def to_signed(onehot: torch.Tensor) -> torch.Tensor:
    """(N, 2, S, S) one-hot phase map -> (N, 1, S, S) in {-1 (matrix), +1 (inclusion)}."""
    return onehot[:, 1:2] * 2 - 1


def validation(model, alpha_bar, cfg: dict, s2_ref: np.ndarray) -> tuple[dict, list[np.ndarray]]:
    """3D volumes from held-out seeds (fast sampler settings) scored against the training image."""
    tc = cfg["train"]
    seeds = tc["select_seeds"][: tc["val_volumes"]]
    vols = [sample_volume(model, alpha_bar, cfg["volume_size"], s, tc["val_steps"], tc["val_mode"]) for s in seeds]
    rmax = cfg["metrics"]["s2_rmax"]
    s2 = np.mean([s2_volume(v, rmax)["mean"] for v in vols], axis=0)
    return {"val_phi": float(np.mean([phase_fraction(v) for v in vols])), "val_s2_mae": s2_mae(s2, s2_ref)}, vols


def save_slice_grid(slices: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 1.9))
    for ax, s in zip(np.atleast_1d(axes), slices):
        ax.imshow(s, cmap="gray_r", interpolation="nearest")
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(cfg: dict) -> Path:
    """Train the DiT (MLflow-tracked when `tracking.mlflow`); returns the run directory."""
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(cfg, run_dir)
    tracker.start()
    try:
        _train(cfg, run_dir, tracker)
    except BaseException:
        tracker.end("FAILED")
        raise
    tracker.artifacts([run_dir / n for n in ("config.yaml", "history.csv", "selection.csv", "selection.yaml")])
    tracker.artifacts([run_dir / "previews"])
    tracker.end()
    return run_dir


def _train(cfg: dict, run_dir: Path, tracker: Tracker) -> None:
    tc = cfg["train"]
    dcfg = cfg["model"]["dit"]
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    device = get_device(cfg["device"])
    torch.backends.cuda.matmul.allow_tf32 = True

    labels = load_label_map(Path(cfg["data"]["train_dirs"][cfg["data"]["branch"]]) / "image.png")
    sampler = RandomCropSampler(labels, cfg["img_size"], cfg["n_phases"], device, seed=cfg["seed"],
                                augment=cfg["data"]["augment"])
    s2_ref = s2_radial(labels, cfg["metrics"]["s2_rmax"])

    model = build_dit(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    alpha_bar = cosine_alpha_bar(dcfg["timesteps"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    warmup = tc.get("warmup_steps", 0)
    total = cfg["epochs"] * cfg["iters_per_epoch"]
    log.info("Run %s | DiT %.1fM params (patch %d, dim %d, depth %d, heads %d) | %d steps x batch %d | "
             "data %s phi=%.4f | sampler: %d DDIM steps, %s", cfg["run_name"],
             sum(p.numel() for p in model.parameters()) / 1e6, dcfg["patch"], dcfg["dim"], dcfg["depth"],
             dcfg["heads"], total, tc["batch"], cfg["data"]["name"], labels.mean(), dcfg["sample_steps"],
             dcfg["sample_mode"])

    history = run_dir / "history.csv"
    selection = run_dir / "selection.csv"
    for path, fields in ((history, HISTORY_FIELDS), (selection, SELECTION_FIELDS)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(fields)
    (run_dir / "previews").mkdir(exist_ok=True)
    best = {"val_s2_mae": float("inf"), "epoch": None}
    step, last_log, losses = 0, time.time(), []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for _ in range(cfg["iters_per_epoch"]):
            x0 = to_signed(sampler(tc["batch"]))
            t = torch.randint(0, dcfg["timesteps"], (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(q_sample(x0, t, noise, alpha_bar), t)
            loss = F.mse_loss(pred.float(), noise)
            if not torch.isfinite(loss):  # never let one bad batch poison the weights (and the EMA)
                log.warning("Non-finite loss at step %d; batch skipped", step)
                opt.zero_grad(set_to_none=True)
                continue
            lr = tc["lr"] * min(1.0, (step + 1) / warmup) if warmup else tc["lr"]
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            update_ema(ema, model, tc["ema_decay"], step)
            step += 1
            losses.append(loss.item())
            if step % tc["log_every"] == 0:
                now = time.time()
                row = {"epoch": epoch, "g_step": step, "g_loss": float(np.mean(losses)), "lr": lr,
                       "sec_per_g_step": (now - last_log) / tc["log_every"]}
                last_log, losses = now, []
                with history.open("a", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, HISTORY_FIELDS).writerow(row)
                tracker.metrics({"train_g_loss": row["g_loss"], "train_sec_per_g_step": row["sec_per_g_step"]},
                                step=step)
                log.info("ep %d/%d | step %d/%d | loss %.4f | lr %.2e | %.3f s/step | ETA %.1f min", epoch,
                         cfg["epochs"], step, total, row["g_loss"], lr, row["sec_per_g_step"],
                         row["sec_per_g_step"] * (total - step) / 60)

        # ---- end of epoch: 3D validation on held-out seeds with the EMA weights
        val, vols = validation(ema, alpha_bar, cfg, s2_ref)
        is_best = val["val_s2_mae"] < best["val_s2_mae"]
        if is_best:
            best = {**val, "epoch": epoch, "g_step": step}
            torch.save(ema.state_dict(), run_dir / "G_best.pt")
        with selection.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, SELECTION_FIELDS).writerow({"epoch": epoch, "g_step": step, **val, "best": int(is_best)})
        tracker.metrics(val, step=epoch)
        log.info("Selection ep %d: val phi=%.4f S2 MAE=%.4f%s (best: ep %s, %.4f)", epoch, val["val_phi"],
                 val["val_s2_mae"], " *" if is_best else "", best["epoch"], best["val_s2_mae"])
        plot_volume_slices(vols[0], run_dir / "previews" / f"epoch_{epoch:03d}.png",
                           f"{cfg['run_name']} epoch {epoch}: 3D sample (multi-plane DDIM)")
        save_slice_grid(sample_slices(ema, alpha_bar, 8, seed=epoch, steps=tc["val_steps"]),
                        run_dir / "previews" / f"epoch_{epoch:03d}_2d.png", f"epoch {epoch}: 2D samples")
        snap = tc.get("snapshot_every")
        if snap and (epoch % snap == 0 or epoch == cfg["epochs"]):
            (run_dir / "snapshots").mkdir(exist_ok=True)
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()},
                       run_dir / "snapshots" / f"G_epoch{epoch:03d}.pt")

    torch.save(ema.state_dict(), run_dir / "G_last.pt")
    with (run_dir / "selection.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"best_epoch": best["epoch"], "best_g_step": best.get("g_step"),
                        "val_phi": best.get("val_phi"), "val_s2_mae": best["val_s2_mae"],
                        "select_seeds": tc["select_seeds"][: tc["val_volumes"]]}, fh, sort_keys=False)
    log.info("Training finished -> %s (best epoch %s, val S2 MAE %.4f)", run_dir, best["epoch"], best["val_s2_mae"])
