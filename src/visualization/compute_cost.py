"""Model size and computational cost: parameters, FLOPs per forward pass, training cost per generator step,
inference time per volume, and the SAM front-end. Writes reports/compute_cost.json (report Section 5.8).

FLOPs are counted with torch.utils.flop_counter (2 FLOPs per multiply-add) on one input. The training cost per
generator step is an estimate from SliceGAN's schedule (Algorithm 1, m_D = 1, m_G = 2, 5 critic iterations):
each G step evaluates the critic on 5 x 3 x (64 + 8) + 3 x 128 = 1464 slices and the generator on 7 volumes,
forward plus backward (about 3x the forward FLOPs); the gradient penalty and DiffAug are not included.
Wall-clock training times come from the run logs.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.discriminator_cnn import build_cnn_discriminator
from src.models.discriminator_swin import build_swin_discriminator
from src.models.generate import load_generator
from src.models.slicegan_wrapper import build_generator
from src.utils import get_logger, load_config, setup_logging

log = get_logger(__name__)

ML = "configs/data/microlib_000210.yaml"
TRAIN_RUNS = ("m1_cnn", "m1_cnn_diffaug", "m1_extended", "m2_swin", "m3_swin_sam", "m4_ensemble", "m5_finetune",
              "m4_ensemble_swin128")


def params(module: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


def flops(module: torch.nn.Module, *inputs) -> float:
    """Forward GFLOPs on the given inputs."""
    module.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(*inputs)
    return fc.get_total_flops() / 1e9


def time_forward(fn, n: int, device: torch.device) -> float:
    """Mean seconds per call after 2 warm-up calls."""
    for _ in range(2):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def training_times(logs: Path = Path("logs")) -> dict:
    """Wall-clock training minutes of the final runs from logs/chain_train_<dataset>_<run>.stdout."""
    out = {}
    for f in sorted(logs.glob("chain_train_*.stdout")):
        m = re.search(r"Training finished in ([0-9.]+) min", f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            out[f.stem.replace("chain_train_", "")] = float(m.group(1))
    return out


def measure(with_sam: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg1 = load_config("configs/m1_cnn.yaml", [ML])
    res: dict = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "components": {}}

    with tempfile.TemporaryDirectory() as tmp:
        G = build_generator(cfg1, Path(tmp), training=True)
        D = build_cnn_discriminator(cfg1, Path(tmp), training=True)
    z64 = torch.randn(1, cfg1["z_channels"], 4, 4, 4)
    z128 = torch.randn(1, cfg1["z_channels"], 6, 6, 6)
    slice64 = torch.randn(1, 2, 64, 64)
    swin_m2 = build_swin_discriminator(load_config("configs/m2_swin.yaml", [ML]))
    swin_m4 = build_swin_discriminator(load_config("configs/m4_ensemble.yaml", [ML]))
    swin_128 = build_swin_discriminator(load_config("configs/m4_ensemble_swin128.yaml", [ML]))
    comp = res["components"]
    comp["generator"] = {**params(G), "gflops_64": flops(G, z64), "gflops_128": flops(G, z128)}
    comp["cnn_critic"] = {**params(D), "gflops_per_slice": flops(D, slice64)}
    comp["swin_critic_m2"] = {**params(swin_m2), "gflops_per_slice": flops(swin_m2, slice64)}
    comp["swin_critic_m4"] = {**params(swin_m4), "gflops_per_slice": flops(swin_m4, slice64)}
    comp["swin_critic_m4_128px"] = {**params(swin_128), "gflops_per_slice": flops(swin_128, slice64)}

    # training cost per generator step (estimate, see module docstring)
    slices, volumes = 5 * 3 * (64 + 8) + 3 * 128, 5 * 1 + 2
    g = comp["generator"]["gflops_64"]
    critics = {"m1_cnn": ["cnn_critic"], "m2_swin": ["swin_critic_m2"], "m4_ensemble": ["cnn_critic", "swin_critic_m4"],
               "m4_ensemble_swin128": ["cnn_critic", "swin_critic_m4_128px"]}
    res["train_tflops_per_g_step"] = {
        k: 3 * (volumes * g + slices * sum(comp[c]["gflops_per_slice"] for c in v)) / 1e3 for k, v in critics.items()}
    res["train_minutes"] = training_times()

    # inference: generator only (all models share it); trained weights of M1
    run_dir = Path(cfg1["paths"]["models"]) / "m1_cnn"
    Gt = load_generator(cfg1, run_dir, device, checkpoint="G_last.pt") if (run_dir / "G_last.pt").exists() else G.to(device)
    with torch.no_grad():
        res["inference_s_per_volume"] = {
            f"gpu_64": time_forward(lambda: Gt(z64.to(device)), 20, device) if device.type == "cuda" else None,
            f"gpu_128": time_forward(lambda: Gt(z128.to(device)), 10, device) if device.type == "cuda" else None,
        }
        Gc = Gt.to("cpu")
        res["inference_s_per_volume"]["cpu_64"] = time_forward(lambda: Gc(z64), 3, torch.device("cpu"))

    if with_sam:
        from src.data.make_dataset import load_grayscale
        from src.features.sam_segment import build_generator as build_sam, generate_masks

        cfg3 = load_config("configs/m3_swin_sam.yaml", [ML])
        sam = build_sam(cfg3["sam"]["model_id"], "cuda" if device.type == "cuda" else "cpu")
        gray = load_grayscale(cfg3["data"]["raw_path"])
        t = time.perf_counter()
        masks = generate_masks(gray, sam, cfg3["sam"])
        res["sam"] = {**params(sam.model), "image": list(gray.shape), "n_masks": len(masks),
                      "seconds_microlib": time.perf_counter() - t}
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/compute_cost.json")
    parser.add_argument("--no-sam", action="store_true", help="skip timing the SAM front-end")
    args = parser.parse_args()
    setup_logging("logs", "compute_cost", "INFO")
    res = measure(not args.no_sam)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("%s", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
