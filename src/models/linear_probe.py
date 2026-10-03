"""Linear-probe diagnostic (Vision-aided GAN, Sec. 3.2): can frozen Swin features separate real from generated slices?

A logistic-regression probe is trained on pooled, frozen Swin-T features of real 64x64 crops vs 2D slices
of volumes from a trained generator, at several input resolutions. High held-out accuracy means the frozen
feature space carries a usable real-vs-fake signal for a frozen-backbone critic; ~50 % means it does not.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.discriminator_swin import adapt_patch_embedding
from src.models.generate import generate_volumes
from src.models.slicegan_wrapper import RandomCropSampler, build_generator, load_label_map, volume_to_slices
from src.utils import get_logger, load_config, set_seed, setup_logging

log = get_logger(__name__)
PERMS = ((2, 3, 4), (3, 2, 4), (4, 2, 3))


def swin_features(images: torch.Tensor, size: int, stages=(2, 3, 4), batch: int = 32) -> dict[str, np.ndarray]:
    """Mean-pooled features of each Swin stage (+ their concatenation) for one-hot images resized to `size`."""
    import timm

    model = timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True, img_size=size,
                              num_classes=0).eval()
    model.patch_embed.proj = adapt_patch_embedding(model.patch_embed.proj, images.shape[1])
    feats: dict[int, list] = {s: [] for s in stages}
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = images[i : i + batch]
            if size != x.shape[-1]:
                x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            h = model.patch_embed(x)
            for idx, stage in enumerate(model.layers, start=1):
                h = stage(h)
                if idx in feats:
                    feats[idx].append(h.mean(dim=(1, 2)).numpy())
                if idx == max(stages):
                    break
    out = {f"stage{s}": np.concatenate(v) for s, v in feats.items()}
    out["concat"] = np.concatenate([out[f"stage{s}"] for s in stages], axis=1)
    return out


def probe_accuracy(x_real: np.ndarray, x_fake: np.ndarray, seed: int, epochs: int = 300) -> tuple[float, float]:
    """Held-out (30 %) accuracy of an L2-regularized logistic regression; returns (train_acc, val_acc)."""
    x = np.concatenate([x_real, x_fake]).astype(np.float32)
    y = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_fake))]).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(0.7 * len(x))
    tr, va = idx[:cut], idx[cut:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xt, xv = torch.tensor((x[tr] - mu) / sd), torch.tensor((x[va] - mu) / sd)
    yt, yv = torch.tensor(y[tr]), torch.tensor(y[va])
    torch.manual_seed(seed)
    clf = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(clf(xt).squeeze(1), yt).backward()
        opt.step()
    with torch.no_grad():
        acc = lambda a, b: float(((clf(a).squeeze(1) > 0).float() == b).float().mean())  # noqa: E731
        return acc(xt, yt), acc(xv, yv)


def fake_slices(volumes: list[np.ndarray], n: int, seed: int, n_phases: int = 2) -> torch.Tensor:
    """n random one-hot 2D slices (all three orientations) from label volumes."""
    vols = torch.as_tensor(np.stack(volumes), dtype=torch.long)
    onehot = F.one_hot(vols, n_phases).permute(0, 4, 1, 2, 3).float()
    slices = torch.cat([volume_to_slices(onehot, p) for p in PERMS])
    return slices[torch.randperm(len(slices), generator=torch.Generator().manual_seed(seed))[:n]]


def main() -> None:
    """CLI: probe frozen Swin features (real vs generated / random) at several resolutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_cnn.yaml")
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--run-dir", required=True, help="folder with G_best.pt and slicegan_params.data")
    parser.add_argument("--n", type=int, default=600, help="samples per class")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 224])
    parser.add_argument("--out", default="reports/linear_probe.csv")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    setup_logging(cfg["paths"]["logs"], "linear_probe", cfg["logging"]["level"])
    set_seed(cfg["seed"])
    run_dir = Path(args.run_dir)

    labels = load_label_map(Path(cfg["data"]["train_dirs"]["raw"]) / "image.png")
    real = RandomCropSampler(labels, cfg["img_size"], cfg["n_phases"], "cpu", seed=cfg["seed"], augment=True)(args.n)

    netG = build_generator(cfg, run_dir, training=False)
    netG.load_state_dict(torch.load(run_dir / "G_best.pt", map_location="cpu", weights_only=True))
    netG.eval()
    gen_vols = generate_volumes(netG, list(range(2000, 2006)), cfg["z_channels"], "cpu")
    phi = float(labels.mean())
    rng = np.random.default_rng(cfg["seed"])
    noise_vols = [(rng.random((64, 64, 64)) < phi).astype(np.uint8) for _ in range(6)]
    fakes = {"generated (G_best)": fake_slices(gen_vols, args.n, 1), "random noise, same phi": fake_slices(noise_vols, args.n, 2)}
    log.info("Real crops: %d | generated phi=%.3f | noise phi=%.3f | train phi=%.3f", len(real),
             np.mean([v.mean() for v in gen_vols]), np.mean([v.mean() for v in noise_vols]), phi)

    rows = []
    for size in args.sizes:
        f_real = swin_features(real, size)
        for fake_name, fk in fakes.items():
            f_fake = swin_features(fk, size)
            for key in f_real:
                tr_acc, va_acc = probe_accuracy(f_real[key], f_fake[key], cfg["seed"])
                rows.append({"input_px": size, "fake": fake_name, "features": key,
                             "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4)})
                log.info("%3d px | %-22s | %-7s | train %.3f | held-out %.3f", size, fake_name, key, tr_acc, va_acc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
