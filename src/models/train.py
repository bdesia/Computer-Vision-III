"""Train SliceGAN (upstream G) with a CNN or Swin-T critic on a 2D label map; fork of slicegan/model.py."""

from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from src.features.descriptors import phase_fraction, s2_mae, s2_radial, s2_volume
from src.models.diffaug import build_diffaug
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
from src.tracking import Tracker
from src.utils import get_device, get_logger, load_config, set_seed, setup_logging
from src.visualization.visualize import plot_volume_slices

log = get_logger(__name__)

HISTORY_FIELDS = ("epoch", "g_step", "d_real", "d_fake", "wasserstein", "gp", "g_loss", "sec_per_g_step",
                  "aux_real", "aux_fake")  # aux_*: second critic of an ensemble (empty otherwise)
SELECTION_FIELDS = ("epoch", "g_step", "val_phi", "val_s2_mae", "best")


def build_discriminator(cfg: dict, run_dir: Path) -> torch.nn.Module:
    """Select the 2D critic from `model.discriminator`."""
    kind = cfg["model"]["discriminator"]
    if kind == "cnn":
        return build_cnn_discriminator(cfg, run_dir)
    if kind == "swin":
        from src.models.discriminator_swin import build_swin_discriminator

        return build_swin_discriminator(cfg)
    raise ValueError(f"Unknown model.discriminator '{kind}' (expected cnn | swin | cnn+swin)")


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float, step: int) -> None:
    """EMA of G's parameters with the usual warm-up min(decay, (1+t)/(10+t)); buffers are copied."""
    d = min(decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1.0 - d)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


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


LOSSES = ("wgan-gp", "hinge")


def critic_scores(netD, x: torch.Tensor, per_scale: bool) -> torch.Tensor:
    """Critic output (N, 1), or (N, S) per-scale scores for multiscale_patch heads under hinge loss."""
    return netD.forward_per_scale(x) if per_scale else netD(x)


@dataclass
class CriticBranch:
    """One 2D critic with its own loss, optimizer, input augmentation and weight in the generator loss."""

    name: str
    net: torch.nn.Module
    opt: optim.Optimizer
    loss: str
    augment: Callable[[torch.Tensor], torch.Tensor]
    weight: float = 1.0
    per_scale: bool = False
    diffaug: bool = False


def build_critics(cfg: dict, run_dir: Path, device) -> list[CriticBranch]:
    """Single critic (cnn | swin) or the CNN + Swin ensemble (cnn+swin, Vision-aided GAN style).

    Ensemble: the CNN branch is exactly M1's critic (WGAN-GP, lr_d, no DiffAug); the Swin branch uses
    `model.ensemble` (loss, lr, weight lambda in the generator loss) and `train.diffaug`.
    """
    tcfg = cfg["train"]
    betas_d = tuple(tcfg.get("betas_d") or tcfg["betas"])  # critic-only override; default = shared betas

    has_policy = bool((tcfg.get("diffaug") or {}).get("policy"))

    def make(name, net, loss, lr, augment, weight, diffaug):
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss '{loss}' for critic '{name}' (expected {' | '.join(LOSSES)})")
        net = net.to(device)
        opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, betas=betas_d)
        # Hinge on multiscale_patch heads: loss per scale, then averaged (Projected / Vision-aided GAN)
        per_scale = loss == "hinge" and hasattr(net, "forward_per_scale")
        return CriticBranch(name, net, opt, loss, augment, weight, per_scale, diffaug)

    kind = cfg["model"]["discriminator"]
    if kind != "cnn+swin":
        return [make(kind, build_discriminator(cfg, run_dir), tcfg.get("loss", "wgan-gp"), tcfg["lr_d"],
                     build_diffaug(cfg), 1.0, has_policy)]
    from src.models.discriminator_swin import build_swin_discriminator

    ens = cfg["model"]["ensemble"]
    identity = lambda x: x  # noqa: E731
    return [
        make("cnn", build_cnn_discriminator(cfg, run_dir), "wgan-gp", tcfg["lr_d"], identity, 1.0, False),
        make("swin", build_swin_discriminator(cfg), ens["swin_loss"], ens["swin_lr"], build_diffaug(cfg),
             float(ens["swin_weight"]), has_policy),
    ]


def _save_checkpoint(run_dir: Path, netG, critics: list[CriticBranch], tag: str) -> None:
    """Save G and the critic(s) under `run_dir` (D_<tag>.pt: a state dict, or {name: state dict})."""
    try:
        torch.save(netG.state_dict(), run_dir / f"G_{tag}.pt")
        d_state = (critics[0].net.state_dict() if len(critics) == 1
                   else {c.name: c.net.state_dict() for c in critics})
        torch.save(d_state, run_dir / f"D_{tag}.pt")
    except OSError as exc:
        log.error("Could not save checkpoint '%s' in %s: %s", tag, run_dir, exc)


def train(cfg: dict) -> Path:
    """Train one model; optionally tracked in MLflow (`tracking.mlflow`), marked FAILED on errors."""
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(cfg, run_dir)
    tracker.start()
    try:
        out = _train(cfg, tracker)
    except BaseException:
        tracker.end("FAILED")
        raise
    tracker.artifacts([run_dir / n for n in ("config.yaml", "history.csv", "selection.csv", "selection.yaml")])
    tracker.artifacts([run_dir / "previews"])
    tracker.end()
    return out


def _train(cfg: dict, tracker: Tracker) -> Path:
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
    init_g = tcfg.get("init_generator")
    if init_g:  # warm start (M5 / M1-extended): continue from a trained generator
        try:
            netG.load_state_dict(torch.load(init_g, map_location=device, weights_only=True))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"train.init_generator not found: {init_g}") from exc
        log.info("Generator initialized from %s", init_g)
    ema_decay = tcfg.get("ema_decay")
    # Generator used for previews, selection and saved checkpoints: an EMA copy when enabled (ViTGAN)
    netG_eval = copy.deepcopy(netG).requires_grad_(False) if ema_decay else netG
    critics = build_critics(cfg, run_dir, device)
    optG = optim.Adam(netG.parameters(), lr=tcfg["lr_g"], betas=tuple(tcfg["betas"]))
    n_g = sum(p.numel() for p in netG.parameters())
    log.info(
        "Run %s | D=%s | device=%s | G params=%.2fM | %s",
        cfg["run_name"], cfg["model"]["discriminator"], device, n_g / 1e6,
        " + ".join(f"{c.name} trainable params={sum(p.numel() for p in c.net.parameters() if p.requires_grad) / 1e6:.2f}M"
                   for c in critics),
    )
    log.info(
        "epochs=%d x %d G steps | critic_iters=%d | m_D=%d, m_G=%d volumes (all %d slices/axis) | "
        "real batch=%d | augment=%s | DiffAug=%s | lr_g=%g | G EMA=%s",
        cfg["epochs"], cfg["iters_per_epoch"], critic_iters, m_d, m_g, l, real_batch,
        cfg["data"]["augment"], (tcfg.get("diffaug") or {}).get("policy"), tcfg["lr_g"], ema_decay or "off",
    )
    for c in critics:
        log.info("critic %s: loss=%s%s | lr=%g betas=%s | weight in G loss=%g | DiffAug=%s", c.name, c.loss,
                 " (per scale)" if c.per_scale else "", c.opt.param_groups[0]["lr"],
                 c.opt.param_groups[0]["betas"], c.weight, c.diffaug)

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
                real_raw = sampler(real_batch)
                fake_raw = volume_to_slices(fake, perm)  # all l * m_d slices
                stats = []
                for c in critics:
                    netD = c.net
                    netD.zero_grad(set_to_none=True)
                    real = c.augment(real_raw)
                    fake_slices = c.augment(fake_raw)
                    if c.loss == "wgan-gp":
                        out_real = netD(real).view(-1).mean()
                        out_fake = netD(fake_slices).mean()
                        # Upstream GP: real batch vs the first real_batch fake slices
                        gp = calc_gradient_penalty(
                            netD, real, fake_slices[:real_batch], real_batch, l, device, tcfg["gp_lambda"], nc
                        )
                        (out_fake - out_real + gp).backward()
                    else:  # hinge, no gradient penalty
                        s_real = critic_scores(netD, real, c.per_scale)
                        s_fake = critic_scores(netD, fake_slices, c.per_scale)
                        out_real, out_fake = s_real.mean(), s_fake.mean()
                        gp = torch.zeros((), device=device)
                        (F.relu(1.0 - s_real).mean() + F.relu(1.0 + s_fake).mean()).backward()
                    c.opt.step()
                    stats.append((out_real, out_fake, gp))
                (out_real, out_fake, gp) = stats[0]

            # ---- Generator: every critic_iters critic iterations
            if i % critic_iters != 0:
                continue
            netG.zero_grad(set_to_none=True)
            fake = netG(sample_noise(m_g, nz, device))
            # One backward per slice orientation (same gradient as summing first): only one orientation's
            # critic activations are held at a time, which keeps large-input critics (Swin at 128 px) in memory.
            err_g = torch.zeros((), device=device)
            for k, perm in enumerate(SLICE_PERMUTATIONS):
                slices = volume_to_slices(fake, perm)
                err = sum(-c.weight * critic_scores(c.net, c.augment(slices), c.per_scale).mean() for c in critics)
                err.backward(retain_graph=k < len(SLICE_PERMUTATIONS) - 1)
                err_g = err_g + err.detach()
            optG.step()
            g_step += 1
            if ema_decay:
                update_ema(netG_eval, netG, ema_decay, g_step)

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
                    "aux_real": stats[1][0].item() if len(stats) > 1 else "",
                    "aux_fake": stats[1][1].item() if len(stats) > 1 else "",
                }
                last_log = now
                with history_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, HISTORY_FIELDS).writerow(row)
                tracker.metrics({f"train_{k}": v for k, v in row.items() if k not in ("epoch", "g_step")},
                                step=g_step)
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
            _save_checkpoint(run_dir, netG_eval, critics, "last")
            netG_eval.eval()
            with torch.no_grad():
                vol = to_labels(netG_eval(sample_noise(1, nz, device)))[0]
            netG.train()
            plot_volume_slices(vol, run_dir / "previews" / f"epoch{epoch:03d}.png",
                               f"{cfg['run_name']} epoch {epoch} (phi={vol.mean():.3f})")
            log.info("Checkpoint saved (epoch %d, preview phi=%.4f)", epoch, vol.mean())
        # ---- Checkpoint selection on held-out seeds (same rule for every model)
        if epoch % tcfg["select_every"] == 0 or epoch == cfg["epochs"] or stop:
            score = validation_score(netG_eval, cfg, s2_ref, device)
            improved = score["val_s2_mae"] < best["val_s2_mae"]
            if improved:
                best = {**score, "epoch": epoch, "g_step": g_step}
                _save_checkpoint(run_dir, netG_eval, critics, "best")
            with selection_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, SELECTION_FIELDS).writerow(
                    {"epoch": epoch, "g_step": g_step, **score, "best": int(improved)})
            tracker.metrics(score, step=epoch)
            log.info("Selection ep %d: val phi=%.4f S2 MAE=%.4f%s (best: ep %s, %.4f)", epoch,
                     score["val_phi"], score["val_s2_mae"], " *" if improved else "", best["epoch"],
                     best["val_s2_mae"])
        # ---- Half-precision generator snapshots: candidates for validation-seed selection (generate.py)
        snap_every = tcfg.get("snapshot_every")
        if snap_every and (epoch % snap_every == 0 or epoch == cfg["epochs"]):
            snap_dir = run_dir / "snapshots"
            snap_dir.mkdir(exist_ok=True)
            try:
                torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in netG_eval.state_dict().items()},
                           snap_dir / f"G_epoch{epoch:03d}.pt")
            except OSError as exc:
                log.error("Could not save snapshot for epoch %d: %s", epoch, exc)
        if stop:
            break

    with (run_dir / "selection.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"best_epoch": best["epoch"], "best_g_step": best.get("g_step"),
                        "val_phi": best.get("val_phi"), "val_s2_mae": best["val_s2_mae"],
                        "select_seeds": list(tcfg["select_seeds"]), "criterion": "S2 MAE vs own training image"},
                       fh, sort_keys=False)
    tracker.metrics({f"best_{k}": v for k, v in best.items() if k in ("epoch", "val_phi", "val_s2_mae")})
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
        if cfg["model"].get("generator", "slicegan") == "dit":  # M6: diffusion transformer, no critic
            from src.models.train_diffusion import train as train_dit

            train_dit(cfg)
        else:
            train(cfg)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        log.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
