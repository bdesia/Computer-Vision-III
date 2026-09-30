"""Export generated volumes, the 2D training label map and metrics as compact JSON for the web viewer."""

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

METRIC_KEYS = ("phi_mean", "phi_std", "phi_train", "abs_dphi", "s2_mae", "s2_err", "L_mae", "L_err",
               "s2_mae_xy", "s2_mae_xz", "s2_mae_yz", "L_mae_xy", "L_mae_xz", "L_mae_yz")


def pack_bits(labels: np.ndarray) -> str:
    """Base64 of the row-major {0,1} array packed 8 voxels per byte (MSB first, numpy packbits)."""
    return base64.b64encode(np.packbits(labels.astype(np.uint8).ravel()).tobytes()).decode("ascii")


def export(cfgs: list[dict], out_path: Path) -> dict:
    """Collect every model of one dataset; models without volumes are listed as pending."""
    data_cfg = cfgs[0]["data"]
    train = load_label_map(Path(data_cfg["train_dirs"]["raw"]) / "image.png")
    payload = {
        "dataset": data_cfg["name"],
        "reference": "ground-truth mask" if data_cfg["source"] == "synthetic" else "Otsu label map",
        "pixel_size_um": (data_cfg.get("microlib") or {}).get("pixel_size_um"),
        "train": {"shape": list(train.shape), "bits": pack_bits(train), "phi": float(train.mean())},
        "models": [],
    }
    for cfg in cfgs:
        run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
        entry = {"id": cfg["run_name"], "label": MODEL_STYLE[cfg["run_name"]]["label"],
                 "critic": cfg["model"]["discriminator"], "input": cfg["data"]["branch"],
                 "status": "pending", "volumes": []}
        vol_paths = sorted((run_dir / "volumes").glob("*.tif"))
        if vol_paths:
            entry["status"] = "ready"
            for p in vol_paths:
                vol = (tifffile.imread(p) > 127).astype(np.uint8)
                entry["volumes"].append({"seed": int(p.stem.split("seed")[-1]), "shape": list(vol.shape),
                                         "bits": pack_bits(vol), "phi": float(vol.mean())})
            metrics_file = run_dir / "metrics.yaml"
            if metrics_file.exists():
                common = yaml.safe_load(metrics_file.read_text(encoding="utf-8"))["references"]["common"]
                entry["metrics"] = {k: common[k] for k in METRIC_KEYS if k in common}
        payload["models"].append(entry)
        log.info("%s: %s (%d volumes)", cfg["run_name"], entry["status"], len(entry["volumes"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("Wrote %s (%.0f KB)", out_path, out_path.stat().st_size / 1024)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+",
                        default=["configs/m1_cnn.yaml", "configs/m2_swin.yaml", "configs/m3_swin_sam.yaml"])
    parser.add_argument("--data", action="append", default=[], help="dataset overlay yaml")
    parser.add_argument("--out", default="reports/viewer/viewer_data.json")
    args = parser.parse_args()
    cfgs = [load_config(c, args.data) for c in args.configs]
    setup_logging(cfgs[0]["paths"]["logs"], "export_viewer", cfgs[0]["logging"]["level"])
    export(cfgs, Path(args.out))


if __name__ == "__main__":
    main()
