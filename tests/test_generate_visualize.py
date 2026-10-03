"""End-to-end test of generate + visualize on a tiny synthetic dataset with an untrained generator (CPU)."""

import copy
import csv
from pathlib import Path

import torch

from src.data.make_dataset import build_dataset
from src.models.generate import evaluate, generate_volumes, load_generator
from src.models.slicegan_wrapper import build_generator
from src.utils import load_config
from src.visualization.visualize import (
    collect_metrics,
    plot_descriptor_curves,
    plot_pipeline,
    plot_qualitative_panel,
    write_metrics_csv,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(tmp_path, model="m1_cnn"):
    cfg = copy.deepcopy(load_config(CONFIGS / f"{model}.yaml"))
    root = tmp_path / "work"
    cfg["device"] = "cpu"
    cfg["paths"].update(interim=str(root / "interim"), models=str(root / "models"), logs=str(root / "logs"),
                        figures=str(root / "figures"), reports=str(root / "reports"))
    cfg["data"]["raw_path"] = str(root / "raw" / "micro_2d.png")
    cfg["data"]["train_dirs"] = {"raw": str(root / "processed" / "train_2d"),
                                 "sam": str(root / "processed" / "train_sam")}
    cfg["data"]["synthetic"]["canvas"] = 128
    cfg["generate"]["seeds"] = [0, 1]
    cfg["metrics"].update(n_volumes_eval=2, s2_rmax=8)
    return cfg


def test_generate_evaluate_and_visualize(tmp_path):
    cfg = _cfg(tmp_path)
    build_dataset(cfg)
    run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
    netG = build_generator(cfg, run_dir, training=True)
    torch.save(netG.state_dict(), run_dir / "G_last.pt")

    reloaded = load_generator(cfg, run_dir, torch.device("cpu"))
    a = generate_volumes(reloaded, [3], cfg["z_channels"], "cpu")[0]
    b = generate_volumes(reloaded, [3], cfg["z_channels"], "cpu")[0]
    assert a.shape == (64, 64, 64) and (a == b).all()  # seeded and deterministic in eval mode

    summary = evaluate(cfg)
    assert set(summary["references"]) == {"train", "common"}
    ref = summary["references"]["common"]
    assert ref["n_volumes"] == 2 and 0.0 <= ref["phi_mean"] <= 1.0 and "L_mae" in ref
    assert len(list((run_dir / "volumes").glob("*.tif"))) == 2

    csv_path = Path(cfg["paths"]["reports"]) / "metrics.csv"
    write_metrics_csv(collect_metrics([run_dir]), csv_path)
    write_metrics_csv(collect_metrics([run_dir]), csv_path)  # upsert, not duplicate
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 2 and {"dphi", "s2_mae", "s2_err", "L_mae", "phi_xy", "phi_xz", "phi_yz"} <= set(rows[0])
    for r in rows:  # bootstrap 95 % CIs bracket the reported values
        assert float(r["s2_mae_lo"]) <= float(r["s2_mae"]) <= float(r["s2_mae_hi"]) + 1e-9
    import numpy as np
    curves = np.load(run_dir / "curves.npz")
    assert curves["common_s2_per_volume"].shape == (2, 9) and curves["common_phi_per_volume"].shape == (2,)
    from src.visualization.visualize import write_comparison
    assert write_comparison([run_dir], tmp_path / "cmp.csv") == []   # only the baseline itself

    fig_dir = Path(cfg["paths"]["figures"])
    plot_descriptor_curves([run_dir], "synthetic", fig_dir / "curves.png")
    plot_qualitative_panel([cfg], fig_dir / "panel.png")
    plot_pipeline(fig_dir / "pipeline.png")
    for name in ("curves.png", "panel.png", "pipeline.png"):
        assert (fig_dir / name).stat().st_size > 10_000


def test_late_training_summary_uses_second_half_of_epochs(tmp_path):
    from src.visualization.visualize import late_training_summary

    with (tmp_path / "selection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "g_step", "val_phi", "val_s2_mae", "best"])
        for ep, (phi, s2) in enumerate([(0.0, 0.9), (0.01, 0.8), (0.2, 0.02), (0.3, 0.04)], start=1):
            w.writerow([ep, ep * 100, phi, s2, 0])
    out = late_training_summary(tmp_path)
    assert out["late_epochs"] == "3-4" and out["late_collapsed"] == 0
    assert abs(out["late_s2_median"] - 0.03) < 1e-12 and abs(out["late_phi_median"] - 0.25) < 1e-12
    assert late_training_summary(tmp_path / "missing") == {}
