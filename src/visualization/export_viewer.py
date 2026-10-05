"""Export generated volumes, the 2D training label maps and metrics as compact JSON for the web viewer.

One file holds every dataset: for each one, the training maps of each front-end (Otsu and, when present,
SAM) and, for each run with generated volumes, the first MAX_VOLUMES volumes plus its test metrics. Runs
that have not been generated yet are skipped. The Streamlit app reads the same file when no local
checkpoints are available.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import tifffile
import yaml

from src.models.slicegan_wrapper import load_label_map
from src.utils import get_logger, load_config, setup_logging
from src.visualization.visualize import MODEL_STYLE

log = get_logger(__name__)

MAX_VOLUMES = 4  # volumes per model embedded in the viewer

METRIC_KEYS = ("n_volumes", "phi_mean", "phi_std", "phi_train", "abs_dphi", "s2_mae", "s2_err", "L_mae", "L_err",
               "s2_mae_xy", "s2_mae_xz", "s2_mae_yz", "L_mae_xy", "L_mae_xz", "L_mae_yz")

# Main models first, then the repeat runs and variants of Section 5.7 (skipped where not generated)
VIEWER_CONFIGS = ("m1_cnn", "m1_cnn_diffaug", "m2_swin", "m3_swin_sam", "m4_ensemble", "m5_finetune", "m1_extended",
                  "m1_cnn_seed2", "m2_swin_seed2", "m4_ensemble_seed2", "m4_ensemble_swin128")
DATASETS = ("configs/data/microlib_000210.yaml", "synthetic")  # "synthetic" = no overlay (default dataset)


def pack_bits(labels: np.ndarray) -> str:
    """Base64 of the row-major {0,1} array packed 8 voxels per byte (MSB first, numpy packbits)."""
    return base64.b64encode(np.packbits(labels.astype(np.uint8).ravel()).tobytes()).decode("ascii")


def unpack_bits(b64: str, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of pack_bits."""
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), dtype=np.uint8))
    return bits[: int(np.prod(shape))].reshape(shape)


def _seed_of(path: Path) -> int:
    return int(path.stem.split("seed")[-1])


def export_dataset(cfgs: list[dict]) -> dict:
    """One dataset: training maps per front-end and every run of `cfgs` that has generated volumes."""
    data_cfg = cfgs[0]["data"]
    entry = {
        "dataset": data_cfg["name"],
        "reference": "ground-truth mask" if data_cfg["source"] == "synthetic" else "Otsu label map",
        "pixel_size_um": (data_cfg.get("microlib") or {}).get("pixel_size_um"),
        "train": {},
        "models": [],
    }
    for branch, folder in data_cfg["train_dirs"].items():
        path = Path(folder) / "image.png"
        if path.exists():
            labels = load_label_map(path)
            entry["train"][branch] = {"shape": list(labels.shape), "bits": pack_bits(labels),
                                      "phi": float(labels.mean())}
    for cfg in cfgs:
        run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
        vol_paths = sorted((run_dir / "volumes").glob("*.tif"), key=_seed_of)[:MAX_VOLUMES]
        if not vol_paths:
            log.info("%s/%s: no generated volumes, skipped", entry["dataset"], cfg["run_name"])
            continue
        style = MODEL_STYLE.get(cfg["run_name"], {"label": cfg["run_name"]})
        model = {"id": cfg["run_name"], "label": style["label"], "critic": cfg["model"]["discriminator"],
                 "input": cfg["data"]["branch"], "volumes": []}
        for p in vol_paths:
            vol = (tifffile.imread(p) > 127).astype(np.uint8)
            model["volumes"].append({"seed": _seed_of(p), "shape": list(vol.shape), "bits": pack_bits(vol),
                                     "phi": float(vol.mean())})
        metrics_file = run_dir / "metrics.yaml"
        if metrics_file.exists():
            m = yaml.safe_load(metrics_file.read_text(encoding="utf-8"))
            common = m["references"]["common"]
            model["metrics"] = {k: common[k] for k in METRIC_KEYS if k in common}
            model["checkpoint"] = m.get("checkpoint")
        entry["models"].append(model)
        log.info("%s/%s: %d volumes", entry["dataset"], cfg["run_name"], len(model["volumes"]))
    return entry


def export(datasets: list[list[dict]], out_path: Path) -> dict:
    """Write viewer_data.json (+ .js for file:// use) with one entry per dataset."""
    payload = {"datasets": [export_dataset(cfgs) for cfgs in datasets if cfgs]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    out_path.write_text(text, encoding="utf-8")
    # Same data as a script, so index.html also works when opened straight from disk (file://)
    out_path.with_suffix(".js").write_text(f"window.VIEWER_DATA={text};", encoding="utf-8")
    log.info("Wrote %s (+ .js, %.0f KB)", out_path, out_path.stat().st_size / 1024)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[f"configs/{m}.yaml" for m in VIEWER_CONFIGS])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        help="dataset overlay yamls; 'synthetic' means the default dataset (no overlay)")
    parser.add_argument("--out", default="reports/viewer/viewer_data.json")
    args = parser.parse_args()
    groups = []
    for ds in args.datasets:
        overlays = [] if ds == "synthetic" else [ds]
        groups.append([load_config(c, overlays) for c in args.configs if Path(c).exists()])
    setup_logging(groups[0][0]["paths"]["logs"], "export_viewer", groups[0][0]["logging"]["level"])
    export(groups, Path(args.out))


if __name__ == "__main__":
    main()
