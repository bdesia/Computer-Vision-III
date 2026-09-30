"""Train SliceGAN (upstream G) with a CNN or Swin-T critic on a 2D label map; fork of slicegan/model.py."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from src.models.discriminator_cnn import build_cnn_discriminator
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


def build_discriminator(cfg: dict, run_dir: Path) -> torch.nn.Module:
    """Select the 2D critic from `model.discriminator`."""
    kind = cfg["model"]["discriminator"]
    if kind == "cnn":
        return build_cnn_discriminator(cfg, run_dir)
    if kind == "swin":
        from src.models.discriminator_swin import build_swin_discriminator

        return build_swin_discriminator(cfg)
    raise ValueError(f"Unknown model.discriminator '{kind}' (expected cnn | swin)")


def _fake_slices(fake: torch.Tensor, perm, n_slices: int | None, gen: torch.Generator) -> torch.Tensor:
    """All slices along one axis (upstream), or a random subset of `n_slices` to bound D cost."""
    slices = volume_to_slices(fake, perm)
    if n_slices is None or n_slices >= slices.shape[0]:
        return slices
    idx = torch.randperm(slices.shape[0], generator=gen, device="cpu")[:n_slices].to(slices.device)
    return slices[idx]


def _save_checkpoint(run_dir: Path, netG, netD, tag: str) -> None:
    """Save G and D state dicts under `run_dir` (tag: 'last' or 'epochXXX')."""
    try:
        torch.save(netG.state_dict(), run_dir / f"G_{tag}.pt")
        torch.save(netD.state_dict(), run_dir / f"D_{tag}.pt")
    except OSError as exc:
        log.error("Could not save checkpoint '%s' in %s: %s", tag, run_dir, exc)


def train(cfg: dict) -> Path:
    """Run WGAN-GP training as in upstream SliceGAN (isotropic: one critic for the three axes)."""
    tcfg = cfg["train"]
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    device = get_device(cfg["device"])
    torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(tcfg.get("allow_tf32", False))
    l, nc, nz = cfg["img_size"], cfg["n_phases"], cfg["z_channels"]
    batch_size = cfg["batch_size"]
    d_batch_size = tcfg.get("d_batch_size", batch_size)
    critic_iters = tcfg["critic_iters"]
    n_fake_slices = tcfg.get("fake_slices")
    max_minutes = tcfg.get("max_minutes")

    train_dir = Path(cfg["data"]["train_dirs"][cfg["data"]["branch"]])
    labels = load_label_map(train_dir / "image.png")
    sampler = RandomCropSampler(labels, l, nc, device, seed=cfg["seed"])
    log.info("Training data: %s %s, phi=%.4f", train_dir / "image.png", labels.shape, labels.mean())

    netG = build_generator(cfg, run_dir).to(device)
    netD = build_discriminator(cfg, run_dir).to(device)
    optG = optim.Adam(netG.parameters(), lr=tcfg["lr_g"], betas=tuple(tcfg["betas"]))
    optD = optim.Adam(
        [p for p in netD.parameters() if p.requires_grad], lr=tcfg["lr_d"], betas=tuple(tcfg["betas"])
    )
    n_g = sum(p.numel() for p in netG.parameters())
    n_d = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    log.info(
        "Run %s | D=%s | device=%s | G params=%.2fM | D trainable params=%.2fM",
        cfg["run_name"], cfg["model"]["discriminator"], device, n_g / 1e6, n_d / 1e6,
    )
    log.info(
        "epochs=%d x %d G steps, critic_iters=%d, batch=%d, fake slices/axis=%s",
        cfg["epochs"], cfg["iters_per_epoch"], critic_iters, batch_size, n_fake_slices or "all",
    )

    slice_gen = torch.Generator().manual_seed(cfg["seed"])
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(HISTORY_FIELDS)

    start = time.time()
    last_log = start
    g_step = 0
    stop = False
    for epoch in range(1, cfg["epochs"] + 1):
        for i in range(1, cfg["iters_per_epoch"] * critic_iters + 1):
            # ---- Critic: one update per axis, same critic for all axes (isotropic)
            fake = netG(sample_noise(d_batch_size, nz, device)).detach()
            for perm in SLICE_PERMUTATIONS:
                netD.zero_grad(set_to_none=True)
                real = sampler(batch_size)
                out_real = netD(real).view(-1).mean()
                fake_slices = _fake_slices(fake, perm, n_fake_slices, slice_gen)
                out_fake = netD(fake_slices).mean()
                gp = calc_gradient_penalty(
                    netD, real, fake_slices[:batch_size], batch_size, l, device, tcfg["gp_lambda"], nc
                )
                (out_fake - out_real + gp).backward()
                optD.step()

            # ---- Generator: every critic_iters critic iterations
            if i % critic_iters != 0:
                continue
            netG.zero_grad(set_to_none=True)
            fake = netG(sample_noise(batch_size, nz, device))
            err_g = 0.0
            for perm in SLICE_PERMUTATIONS:
                err_g = err_g - netD(_fake_slices(fake, perm, n_fake_slices, slice_gen)).mean()
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
        if stop:
            break

    log.info("Training finished in %.1f min -> %s", (time.time() - start) / 60, run_dir)
    return run_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="override epochs (e.g. smoke tests)")
    parser.add_argument("--iters-per-epoch", type=int, help="override iters_per_epoch")
    parser.add_argument("--device", help="override device (cuda | cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
