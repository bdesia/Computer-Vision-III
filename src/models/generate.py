"""Generate N 64^3 volumes from a trained generator and score phi / S2 / L against 2D references."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml

from src.features.descriptors import describe_volumes
from src.models.slicegan_wrapper import build_generator, load_label_map, sample_noise, to_labels
from src.tracking import RUN_ID_FILE, Tracker, eval_metrics
from src.utils import get_device, get_logger, load_config, set_seed, setup_logging
from src.visualization.visualize import plot_volume_slices

log = get_logger(__name__)

CURVE_KEYS = ("s2_train", "s2_generated", "L_train", "L_generated",
              "phi_per_volume", "s2_per_volume", "L_per_volume")


def load_generator(cfg: dict, run_dir: Path, device, checkpoint: str = "G_last.pt") -> torch.nn.Module:
    """Rebuild the upstream generator from its params file and load trained weights.

    For M6 (`model.generator: dit`) the checkpoint holds DiT weights and the returned module samples
    volumes by multi-plane diffusion (`sample_volume(seed)`).
    """
    ckpt = run_dir / checkpoint
    if cfg["model"].get("generator", "slicegan") == "dit":
        from src.models.diffusion import build_volume_generator

        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No checkpoint {ckpt}; train the model first.") from exc
        train_map = load_label_map(Path(cfg["data"]["train_dirs"][cfg["data"]["branch"]]) / "image.png")
        return build_volume_generator(cfg, state, device, phi=float(train_map.mean()))
    netG = build_generator(cfg, run_dir, training=False)
    try:
        netG.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No checkpoint {ckpt}; train the model first.") from exc
    return netG.to(device).eval()


@torch.no_grad()
def generate_volumes(netG: torch.nn.Module, seeds: list[int], z_channels: int, device) -> list[np.ndarray]:
    """One uint8 {0, 1} volume per seed (seeded latent cube, generator in eval mode)."""
    if hasattr(netG, "sample_volume"):  # M6: multi-plane diffusion sampling, seeded 3D noise
        volumes = []
        for i, seed in enumerate(seeds):
            volumes.append(netG.sample_volume(seed))
            if (i + 1) % 16 == 0:
                log.info("sampled %d / %d volumes", i + 1, len(seeds))
        return volumes
    volumes = []
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        volumes.append(to_labels(netG(sample_noise(1, z_channels, device, generator=gen)))[0])
    return volumes


def candidate_checkpoints(run_dir: Path) -> list[str]:
    """Checkpoint files (relative to run_dir) a model can be evaluated from: best, last and snapshots."""
    names = [n for n in ("G_best.pt", "G_last.pt") if (run_dir / n).exists()]
    names += sorted(p.relative_to(run_dir).as_posix() for p in (run_dir / "snapshots").glob("G_epoch*.pt"))
    return names


def select_checkpoint(cfg: dict, run_dir: Path, device) -> tuple[str, dict[str, float]]:
    """Pick the candidate with the lowest S2 MAE vs the model's own training image on validation seeds.

    Validation seeds (`generate.validation_seeds`) are disjoint from the per-epoch selection seeds used
    during training and from the evaluation seeds, so the reported metrics stay unbiased.
    """
    from src.features.descriptors import s2_mae, s2_radial, s2_volume

    candidates = candidate_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No generator checkpoint in {run_dir}; train the model first.")
    if len(candidates) == 1:
        return candidates[0], {}
    seeds = cfg["generate"]["validation_seeds"]
    rmax = cfg["metrics"]["s2_rmax"]
    ref = references(cfg)["train"]
    s2_ref = s2_radial(ref, rmax)
    scores = {}
    for name in candidates:
        netG = load_generator(cfg, run_dir, device, checkpoint=name)
        vols = generate_volumes(netG, seeds, cfg["z_channels"], device)
        scores[name] = s2_mae(np.mean([s2_volume(v, rmax)["mean"] for v in vols], axis=0), s2_ref)
        log.info("validation (%d seeds) %s: S2 MAE %.4f", len(seeds), name, scores[name])
    best = min(scores, key=scores.get)
    log.info("Selected %s on validation seeds", best)
    return best, scores


def references(cfg: dict) -> dict[str, np.ndarray]:
    """2D label maps to compare against: the model's own training image, plus the common reference.

    `train`: the image this model was trained on (Otsu map for M1/M2, SAM map for M3).
    `common`: shared by all models of a dataset, so M2 vs M3 is comparable: the exact ground-truth
    mask for synthetic data, the Otsu map for real data (no ground truth exists).
    """
    data = cfg["data"]
    refs = {"train": load_label_map(Path(data["train_dirs"][data["branch"]]) / "image.png")}
    if data["source"] == "synthetic":
        refs["common"] = load_label_map(Path(data["raw_path"]).with_name("micro_2d_gt.png"))
    else:
        refs["common"] = load_label_map(Path(data["train_dirs"]["raw"]) / "image.png")
    return refs


def evaluate(cfg: dict) -> dict:
    """Generate the evaluation volumes of one trained model, save them and write metrics + curves."""
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    device = get_device(cfg["device"])
    seeds = cfg["generate"]["seeds"]
    rmax = cfg["metrics"]["s2_rmax"]
    if len(seeds) != cfg["metrics"]["n_volumes_eval"]:
        raise ValueError(f"generate.seeds has {len(seeds)} entries, metrics.n_volumes_eval="
                         f"{cfg['metrics']['n_volumes_eval']}")

    tag = cfg["generate"].get("checkpoint", "last")
    val_scores: dict[str, float] = {}
    if tag == "auto":
        ckpt, val_scores = select_checkpoint(cfg, run_dir, device)
    else:
        ckpt = f"G_{tag}.pt"
        if not (run_dir / ckpt).exists() and tag != "last":
            log.warning("No %s in %s (older run?); using G_last.pt", ckpt, run_dir)
            ckpt = "G_last.pt"
    tag = ckpt
    netG = load_generator(cfg, run_dir, device, checkpoint=ckpt)
    volumes = generate_volumes(netG, seeds, cfg["z_channels"], device)
    vol_dir = run_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)
    for seed, vol in zip(seeds, volumes):
        try:
            tifffile.imwrite(vol_dir / f"volume_{cfg['volume_size']}_seed{seed}.tif", vol * 255)
        except OSError as exc:
            log.error("Could not write volume for seed %s: %s", seed, exc)
    plot_volume_slices(volumes[0], run_dir / "volume_slices.png",
                       f"{cfg['run_name']} ({cfg['data']['name']}) seed {seeds[0]}")

    summary = {"dataset": cfg["data"]["name"], "model": cfg["run_name"], "seeds": list(seeds),
               "checkpoint": tag, "references": {}}
    if val_scores:
        summary["checkpoint_validation"] = {"seeds": len(cfg["generate"]["validation_seeds"]),
                                            "s2_mae_vs_train": {k: float(v) for k, v in val_scores.items()}}
    if (run_dir / "selection.yaml").exists():
        summary["selection"] = yaml.safe_load((run_dir / "selection.yaml").read_text(encoding="utf-8"))
    curves = {}
    for ref_name, ref in references(cfg).items():
        out = describe_volumes(volumes, ref, rmax, lineal=cfg["metrics"]["lineal_path"])
        for key in CURVE_KEYS:
            if key in out:
                curves[f"{ref_name}_{key}"] = out.pop(key)
        summary["references"][ref_name] = {k: float(v) if isinstance(v, (float, np.floating)) else v
                                           for k, v in out.items()}
        log.info("%s vs %s: phi=%.4f±%.4f (ref %.4f) |dphi|=%.4f S2 MAE=%.4f err=%.3f L MAE=%s",
                 cfg["run_name"], ref_name, out["phi_mean"], out["phi_std"], out["phi_train"],
                 out["abs_dphi"], out["s2_mae"], out["s2_err"],
                 f"{out['L_mae']:.4f}" if "L_mae" in out else "n/a")

    np.savez(run_dir / "curves.npz", r=np.arange(rmax + 1), **curves)
    with (run_dir / "metrics.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    log.info("Saved %d volumes, metrics.yaml and curves.npz -> %s", len(volumes), run_dir)
    if (run_dir / RUN_ID_FILE).exists():  # add evaluation to the training run's MLflow record
        tracker = Tracker(cfg, run_dir)
        tracker.start(resume=True)
        tracker.metrics(eval_metrics(summary))
        tracker.artifacts([run_dir / n for n in ("metrics.yaml", "curves.npz", "volume_slices.png")])
        tracker.end()
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml")
    parser.add_argument("--device", help="override device (cuda | cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    if args.device:
        cfg["device"] = args.device
    setup_logging(cfg["paths"]["logs"], f"generate_{cfg['run_name']}", cfg["logging"]["level"])
    set_seed(cfg["seed"])
    try:
        evaluate(cfg)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        log.exception("Generation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
