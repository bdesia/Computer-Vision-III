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
from src.utils import get_device, get_logger, load_config, set_seed, setup_logging
from src.visualization.visualize import plot_volume_slices

log = get_logger(__name__)

CURVE_KEYS = ("s2_train", "s2_generated", "L_train", "L_generated")


def load_generator(cfg: dict, run_dir: Path, device, checkpoint: str = "G_last.pt") -> torch.nn.Module:
    """Rebuild the upstream generator from its params file and load trained weights."""
    netG = build_generator(cfg, run_dir, training=False)
    ckpt = run_dir / checkpoint
    try:
        netG.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No checkpoint {ckpt}; train the model first.") from exc
    return netG.to(device).eval()


@torch.no_grad()
def generate_volumes(netG: torch.nn.Module, seeds: list[int], z_channels: int, device) -> list[np.ndarray]:
    """One uint8 {0, 1} volume per seed (seeded latent cube, generator in eval mode)."""
    volumes = []
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        volumes.append(to_labels(netG(sample_noise(1, z_channels, device, generator=gen)))[0])
    return volumes


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

    netG = load_generator(cfg, run_dir, device)
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
               "references": {}}
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
