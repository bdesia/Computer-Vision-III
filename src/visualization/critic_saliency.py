"""What does the critic look at? SmoothGrad input-gradient saliency of trained critics on real and generated slices.

For each critic (M1 CNN, M2 Swin-T, M4 frozen-Swin heads) the gradient of the critic score with respect to
the one-hot input slice is averaged over noisy copies of the input (SmoothGrad, Smilkov et al., 2017) and
summed over the two phase channels. The share of saliency on phase boundaries (pixels within 1 px of an
interface) measures whether a critic judges interfaces or bulk regions; critic scores on real vs generated
slices show how well it still separates them at the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from src.models.discriminator_cnn import build_cnn_discriminator  # noqa: E402
from src.models.discriminator_swin import build_swin_discriminator  # noqa: E402
from src.models.generate import generate_volumes, load_generator  # noqa: E402
from src.models.slicegan_wrapper import RandomCropSampler, load_label_map  # noqa: E402
from src.utils import get_logger, load_config, set_seed, setup_logging  # noqa: E402
from src.visualization.visualize import INK, INK_MUTED, MODEL_STYLE, _style_axes  # noqa: E402

log = get_logger(__name__)

CRITICS = (("m1_cnn", "cnn", "M1: SliceGAN CNN"), ("m2_swin", "swin", "M2: Swin-T (stages 3-4 trained)"),
           ("m4_ensemble", "swin", "M4: frozen Swin-T, per-position heads"))


def load_critic(cfg: dict, run_dir: Path, branch: str) -> torch.nn.Module:
    """Rebuild a trained critic (matching the evaluated generator checkpoint) and load its weights."""
    ckpt = yaml.safe_load((run_dir / "metrics.yaml").read_text(encoding="utf-8"))["checkpoint"]
    state = torch.load(run_dir / ckpt.replace("G_", "D_"), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and set(state) == {"cnn", "swin"}:  # ensemble checkpoint
        state = state[branch]
    if branch == "cnn":
        net = build_cnn_discriminator(cfg, run_dir, training=False)
    else:
        net = build_swin_discriminator(cfg)  # model.swin holds the head / spectral-norm settings of the run
    net.load_state_dict(state)
    return net.eval()


def smoothgrad(net: torch.nn.Module, x: torch.Tensor, n: int = 16, sigma: float = 0.15, seed: int = 0) -> torch.Tensor:
    """Mean |d score / d input| over n noisy copies, summed over channels -> (N, H, W)."""
    gen = torch.Generator().manual_seed(seed)
    total = torch.zeros(x.shape[0], *x.shape[2:])
    for _ in range(n):
        xi = (x + sigma * torch.randn(x.shape, generator=gen)).requires_grad_(True)
        score = net(xi).reshape(x.shape[0], -1).mean(dim=1)
        (grad,) = torch.autograd.grad(score.sum(), xi)
        total += grad.abs().sum(dim=1)
    return total / n


def boundary_share(saliency: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of saliency on pixels within 1 px of a phase interface (vs the area fraction of those pixels)."""
    edge = np.zeros_like(labels, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge = binary_dilation(edge)
    return float(saliency[edge].sum() / max(saliency.sum(), 1e-12))


def edge_area(labels: np.ndarray) -> float:
    """Area fraction of the boundary band used in boundary_share."""
    edge = np.zeros_like(labels, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    return float(binary_dilation(edge).mean())


def analyse(cfg_data: list[str], n_slices: int, out_fig: Path, out_json: Path) -> dict:
    """Saliency maps, boundary shares and real/fake critic scores for the three critics."""
    device = torch.device("cpu")
    results, panels = {}, []
    for run, branch, title in CRITICS:
        cfg = load_config(f"configs/{run}.yaml", cfg_data)
        run_dir = Path(cfg["paths"]["models"]) / run
        net = load_critic(cfg, run_dir, branch)
        labels = load_label_map(Path(cfg["data"]["train_dirs"][cfg["data"]["branch"]]) / "image.png")
        real = RandomCropSampler(labels, cfg["img_size"], cfg["n_phases"], device, seed=11)(n_slices)
        ckpt = yaml.safe_load((run_dir / "metrics.yaml").read_text(encoding="utf-8"))["checkpoint"]
        G = load_generator(cfg, run_dir, device, checkpoint=ckpt)
        vols = generate_volumes(G, list(range(500, 500 + n_slices)), cfg["z_channels"], device)
        fake_lab = np.stack([v[32] for v in vols])  # central xy slice of each volume
        fake = F.one_hot(torch.as_tensor(fake_lab, dtype=torch.long), 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            s_real = net(real).reshape(n_slices, -1).mean(1).numpy()
            s_fake = net(fake).reshape(n_slices, -1).mean(1).numpy()
        sal_real = smoothgrad(net, real).numpy()
        sal_fake = smoothgrad(net, fake).numpy()
        real_lab = real.argmax(1).numpy()
        bs_real = np.mean([boundary_share(s, l) for s, l in zip(sal_real, real_lab)])
        bs_fake = np.mean([boundary_share(s, l) for s, l in zip(sal_fake, fake_lab)])
        area = np.mean([edge_area(l) for l in real_lab])
        results[run] = {"title": title, "boundary_share_real": float(bs_real), "boundary_share_fake": float(bs_fake),
                        "boundary_area_fraction": float(area), "score_real_mean": float(s_real.mean()),
                        "score_fake_mean": float(s_fake.mean()),
                        "real_above_fake": float((s_real[:, None] > s_fake[None, :]).mean())}
        log.info("%s: boundary share real %.2f / fake %.2f (boundary area %.2f) | P(score real > fake) %.2f",
                 run, bs_real, bs_fake, area, results[run]["real_above_fake"])
        panels.append((run, title, real_lab[0], sal_real[0], fake_lab[0], sal_fake[0], s_real, s_fake))

    fig, axes = plt.subplots(len(panels), 5, figsize=(15, 3.1 * len(panels)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})
    for row, (run, title, rl, rs, fl, fs, sr, sf) in zip(axes, panels):
        for ax, img, cmap, t in ((row[0], rl, "gray", "real crop"), (row[1], rs, "magma", "saliency (real)"),
                                 (row[2], fl, "gray", "generated slice"), (row[3], fs, "magma", "saliency (generated)")):
            ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_title(t, fontsize=9, color=INK_MUTED)
            ax.axis("off")
        row[0].text(-0.12, 0.5, title, transform=row[0].transAxes, rotation=90, ha="right", va="center",
                    fontsize=9.5, color=INK)
        color = MODEL_STYLE[run]["color"]
        bins = np.linspace(min(sr.min(), sf.min()), max(sr.max(), sf.max()), 20)
        row[4].hist(sr, bins=bins, color=color, alpha=0.85, label="real")
        row[4].hist(sf, bins=bins, color=INK_MUTED, alpha=0.5, label="generated")
        _style_axes(row[4])
        row[4].set_title(f"critic scores (boundary share {results[run]['boundary_share_real']:.2f})",
                         fontsize=9, color=INK)
        row[4].legend(frameon=False, fontsize=8)
    fig.suptitle("What do the critics look at? SmoothGrad saliency on real crops and generated slices "
                 "(MicroLib 000210)", fontsize=11, color=INK)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", default=["configs/data/microlib_000210.yaml"])
    parser.add_argument("--n", type=int, default=32, help="real crops and generated slices per critic")
    parser.add_argument("--out", default="reports/figures/critic_saliency.png")
    args = parser.parse_args()
    setup_logging("logs", "critic_saliency", "INFO")
    set_seed(0)
    analyse(args.data, args.n, Path(args.out), Path("reports/critic_saliency.json"))


if __name__ == "__main__":
    main()
