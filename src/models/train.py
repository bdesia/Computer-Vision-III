"""Train SliceGAN (upstream G) with a CNN or Swin-T critic on a 2D label map; fork of slicegan/model.py."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from src.features.descriptors import phase_fraction, s2_mae, s2_radial, s2_volume
from src.models.discriminator_cnn import build_cnn_discriminator
from src.models.generate import generate_volumes
from src.models.slicegan_wrapper import (
    SLICE_PERMUTATIONS,
    RandomCropSampler,
    build_generator,
    calc_gradient_penalty,
    load_label_map,
    sample_noise,
    to_labels,
    volume_to_slices,
)
from src.utils import get_device, get_logger, load_config, set_seed, setup_logging
from src.visualization.visualize import plot_volume_slices

log = get_logger(__name__)

HISTORY_FIELDS = ("epoch", "g_step", "d_real", "d_fake", "wasserstein", "gp", "g_loss", "sec_per_g_step")
SELECTION_FIELDS = ("epoch", "g_step", "val_phi", "val_s2_mae", "best")


def build_discriminator(cfg: dict, run_dir: Path) -> torch.nn.Module:
    """Select the 2D critic from `model.discriminator`."""
    kind = cfg["model"]["discriminator"]
    if kind == "cnn":
        return build_cnn_discriminator(cfg, run_dir)
    if kind == "swin":
        from src.models.discriminator_swin import build_swin_discriminator

        return build_swin_discriminator(cfg)
    raise ValueError(f"Unknown model.discriminator '{kind}' (expected cnn | swin)")


def validation_score(netG, cfg: dict, s2_ref: np.ndarray, device) -> dict:
    """Score G on held-out latent seeds: mean phi and S2 MAE vs the model's own 2D training image.

    The seeds (`train.select_seeds`) differ from the evaluation seeds (`generate.seeds`), so the
    volumes used to pick the checkpoint are never the ones that are reported.
    """
    netG.eval()
    volumes = generate_volumes(netG, cfg["train"]["select_seeds"], cfg["z_channels"], device)
    netG.train()
    rmax = cfg["metrics"]["s2_rmax"]
    s2 = np.mean([s2_volume(v, rmax)["mean"] for v in volumes], axis=0)
    return {"val_phi": float(np.mean([phase_fraction(v) for v in volumes])), "val_s2_mae": s2_mae(s2, s2_ref)}


def _save_checkpoint(run_dir: Path, netG, netD, tag: str) -> None:
    """Save G and D state dicts under `run_dir` (tag: 'last' or 'epochXXX')."""
    try:
        torch.save(netG.state_dict(), run_dir / f"G_{tag}.pt")
        torch.save(netD.state_dict(), run_dir / f"D_{tag}.pt")
    except OSError as exc:
        log.error("Could not save checkpoint '%s' in %s: %s", tag, run_dir, exc)


def train(cfg: dict) -> Path:
    """Run WGAN-GP training as in SliceGAN Algorithm 1 (isotropic: one critic for the three axes).

    Each critic step generates m_d volumes and shows the critic all l slices per axis of each; each
    generator step uses m_g = 2 m_d volumes, again with all slices (Kench & Cooper, 2021, Sec. 3).
    """
    tcfg = cfg["train"]
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    device = get_device(cfg["device"])
    torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(tcfg.get("allow_tf32", False))
    l, nc, nz = cfg["img_size"], cfg["n_phases"], cfg["z_channels"]
    m_d, m_g = tcfg["m_d"], tcfg["m_g"]
    real_batch = tcfg["real_batch"]
    critic_iters = tcfg["critic_iters"]
    max_minutes = tcfg.get("max_minutes")
    if real_batch > l * m_d:
        raise ValueError(f"train.real_batch={real_batch} exceeds the {l * m_d} fake slices per axis")

    train_dir = Path(cfg["data"]["train_dirs"][cfg["data"]["branch"]])
    labels = load_label_map(train_dir / "image.png")
    sampler = RandomCropSampler(labels, l, nc, device, seed=cfg["seed"], augment=cfg["data"]["augment"])
    log.info("Dataset %s: %s %s, phi=%.4f", cfg["data"]["name"], train_dir / "image.png", labels.shape,
             labels.mean())

    netG = build_generator(cfg, run_dir).to(device)
    netD = build_discriminator(cfg, run_dir).to(device)
    optG = optim.Adam(netG.parameters(), lr=tcfg["lr_g"], betas=tuple(tcfg["betas"]))
    betas_d = tuple(tcfg.get("betas_d") or tcfg["betas"])  # critic-only override; default = shared betas
    optD = optim.Adam([p for p in netD.parameters() if p.requires_grad], lr=tcfg["lr_d"], betas=betas_d)
    n_g = sum(p.numel() for p in netG.parameters())
    n_d = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    log.info(
        "Run %s | D=%s | device=%s | G params=%.2fM | D trainable params=%.2fM",
        cfg["run_name"], cfg["model"]["discriminator"], device, n_g / 1e6, n_d / 1e6,
    )
    log.info(
        "epochs=%d x %d G steps | critic_iters=%d | m_D=%d, m_G=%d volumes (all %d slices/axis) | "
        "real batch=%d | augment=%s | lr_g=%g lr_d=%g betas_d=%s",
        cfg["epochs"], cfg["iters_per_epoch"], critic_iters, m_d, m_g, l, real_batch,
        cfg["data"]["augment"], tcfg["lr_g"], tcfg["lr_d"], betas_d,
    )

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(HISTORY_FIELDS)
    selection_path = run_dir / "selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(SELECTION_FIELDS)
    s2_ref = s2_radial(labels, cfg["metrics"]["s2_rmax"])
    best = {"val_s2_mae": float("inf"), "epoch": None}

    start = time.time()
    last_log = start
    g_step = 0
    stop = False
    for epoch in range(1, cfg["epochs"] + 1):
        for i in range(1, cfg["iters_per_epoch"] * critic_iters + 1):
            # ---- Critic: one update per axis, same critic for all axes (isotropic)
            fake = netG(sample_noise(m_d, nz, device)).detach()
            for perm in SLICE_PERMUTATIONS:
                netD.zero_grad(set_to_none=True)
                real = sampler(real_batch)
                out_real = netD(real).view(-1).mean()
                fake_slices = volume_to_slices(fake, perm)  # all l * m_d slices
                out_fake = netD(fake_slices).mean()
                # Upstream GP: real batch vs the first real_batch fake slices
                gp = calc_gradient_penalty(
                    netD, real, fake_slices[:real_batch], real_batch, l, device, tcfg["gp_lambda"], nc
                )
                (out_fake - out_real + gp).backward()
                optD.step()

            # ---- Generator: every critic_iters critic iterations
            if i % critic_iters != 0:
                continue
            netG.zero_grad(set_to_none=True)
            fake = netG(sample_noise(m_g, nz, device))
            err_g = 0.0
            for perm in SLICE_PERMUTATIONS:
                err_g = err_g - netD(volume_to_slices(fake, perm)).mean()
            err_g.backward()
            optG.step()
            g_step += 1

            if g_step % tcfg["log_every"] == 0:
                now = time.time()
                row = {
                    "epoch": epoch,
                    "g_step": g_step,
                    "d_real": out_real.item(),
                    "d_fake": out_fake.item(),
                    "wasserstein": out_real.item() - out_fake.item(),
                    "gp": gp.item(),
                    "g_loss": err_g.item(),
                    "sec_per_g_step": (now - last_log) / tcfg["log_every"],
                }
                last_log = now
                with history_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, HISTORY_FIELDS).writerow(row)
                total_steps = cfg["epochs"] * cfg["iters_per_epoch"]
                eta_min = row["sec_per_g_step"] * (total_steps - g_step) / 60
                log.info(
                    "ep %d/%d | G step %d/%d | W=%.4f gp=%.4f G=%.4f | %.2f s/step | ETA %.1f min",
                    epoch, cfg["epochs"], g_step, total_steps, row["wasserstein"], row["gp"],
                    row["g_loss"], row["sec_per_g_step"], eta_min,
                )

            if max_minutes and (time.time() - start) / 60 >= max_minutes:
                log.warning("Reached train.max_minutes=%s; stopping after epoch %d.", max_minutes, epoch)
                stop = True
                break

        if epoch % tcfg["ckpt_every"] == 0 or epoch == cfg["epochs"] or stop:
            _save_checkpoint(run_dir, netG, netD, "last")
            netG.eval()
            with torch.no_grad():
                vol = to_labels(netG(sample_noise(1, nz, device)))[0]
            netG.train()
            plot_volume_slices(vol, run_dir / "previews" / f"epoch{epoch:03d}.png",
                               f"{cfg['run_name']} epoch {epoch} (phi={vol.mean():.3f})")
            log.info("Checkpoint saved (epoch %d, preview phi=%.4f)", epoch, vol.mean())
        # ---- Checkpoint selection on held-out seeds (same rule for every model)
        if epoch % tcfg["select_every"] == 0 or epoch == cfg["epochs"] or stop:
            score = validation_score(netG, cfg, s2_ref, device)
            improved = score["val_s2_mae"] < best["val_s2_mae"]
            if improved:
                best = {**score, "epoch": epoch, "g_step": g_step}
                _save_checkpoint(run_dir, netG, netD, "best")
            with selection_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, SELECTION_FIELDS).writerow(
                    {"epoch": epoch, "g_step": g_step, **score, "best": int(improved)})
            log.info("Selection ep %d: val phi=%.4f S2 MAE=%.4f%s (best: ep %s, %.4f)", epoch,
                     score["val_phi"], score["val_s2_mae"], " *" if improved else "", best["epoch"],
                     best["val_s2_mae"])
        if stop:
            break

    with (run_dir / "selection.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"best_epoch": best["epoch"], "best_g_step": best.get("g_step"),
                        "val_phi": best.get("val_phi"), "val_s2_mae": best["val_s2_mae"],
                        "select_seeds": list(tcfg["select_seeds"]), "criterion": "S2 MAE vs own training image"},
                       fh, sort_keys=False)
    log.info("Training finished in %.1f min -> %s (best checkpoint: epoch %s, val S2 MAE %.4f)",
             (time.time() - start) / 60, run_dir, best["epoch"], best["val_s2_mae"])
    return run_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml, e.g. configs/data/microlib_000210.yaml")
    parser.add_argument("--epochs", type=int, help="override epochs (e.g. smoke tests)")
    parser.add_argument("--iters-per-epoch", type=int, help="override iters_per_epoch")
    parser.add_argument("--device", help="override device (cuda | cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    for key, value in (("epochs", args.epochs), ("iters_per_epoch", args.iters_per_epoch),
                       ("device", args.device)):
        if value is not None:
            cfg[key] = value
    setup_logging(cfg["paths"]["logs"], cfg["run_name"], cfg["logging"]["level"])
    set_seed(cfg["seed"])
    try:
        train(cfg)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        log.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
