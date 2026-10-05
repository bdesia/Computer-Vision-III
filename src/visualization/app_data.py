"""Data access for the Streamlit app: runs, stored and newly generated volumes, curves, metrics, meshes, RVE files.

Kept free of Streamlit so it can be tested. Three sources of volumes, in order of preference: generating
from the run's evaluated checkpoint (needs local checkpoints), the evaluation volumes on disk
(models/<dataset>/<run>/volumes), and the viewer export (reports/viewer/viewer_data.json).
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import yaml

from src.features.descriptors import lineal_path_volume, phase_fraction, relative_error, s2_mae, s2_volume
from src.utils import load_config
from src.visualization.export_viewer import DATASETS, VIEWER_CONFIGS, unpack_bits
from src.visualization.visualize import MODEL_STYLE

VIEWER_DATA = Path("reports/viewer/viewer_data.json")
METRICS_CSVS = (Path("reports/metrics.csv"), Path("reports/metrics_seeds_microlib.csv"))
RMAX = 32


@dataclass
class Run:
    """One trained model of one dataset and what is available for it locally."""

    run_name: str
    label: str
    dataset: str
    cfg: dict
    run_dir: Path
    checkpoint: str | None = None          # checkpoint chosen by the evaluation (metrics.yaml)
    metrics: dict = field(default_factory=dict)  # common-reference test metrics
    stored_seeds: list[int] = field(default_factory=list)

    @property
    def can_generate(self) -> bool:
        return self.checkpoint is not None and (self.run_dir / self.checkpoint).exists()


def dataset_overlays(dataset: str) -> list[str]:
    """Config overlays for a dataset name ('synthetic' is the default dataset)."""
    return [] if dataset == "synthetic" else [d for d in DATASETS if Path(d).stem == dataset]


def dataset_names() -> list[str]:
    return [Path(d).stem if d != "synthetic" else "synthetic" for d in DATASETS]


def _viewer_payload(path: Path = VIEWER_DATA) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"datasets": []}


def list_runs(dataset: str, configs: tuple[str, ...] = VIEWER_CONFIGS, viewer_data: Path = VIEWER_DATA) -> list[Run]:
    """Runs of `dataset` that have something to show (checkpoint, stored volumes or viewer volumes)."""
    viewer = next((d for d in _viewer_payload(viewer_data)["datasets"] if d["dataset"] == dataset), {"models": []})
    viewer_models = {m["id"]: m for m in viewer["models"]}
    runs = []
    for name in configs:
        cfg_path = Path("configs") / f"{name}.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_config(str(cfg_path), dataset_overlays(dataset))
        run_dir = Path(cfg["paths"]["models"]) / cfg["run_name"]
        run = Run(name, MODEL_STYLE.get(name, {"label": name})["label"], dataset, cfg, run_dir)
        metrics_file = run_dir / "metrics.yaml"
        if metrics_file.exists():
            m = yaml.safe_load(metrics_file.read_text(encoding="utf-8"))
            run.checkpoint = m.get("checkpoint")
            run.metrics = m["references"]["common"]
        elif name in viewer_models:
            run.metrics = viewer_models[name].get("metrics", {})
        tifs = sorted((run_dir / "volumes").glob("*.tif"))
        if tifs:
            run.stored_seeds = sorted(int(p.stem.split("seed")[-1]) for p in tifs)
        elif name in viewer_models:
            run.stored_seeds = [v["seed"] for v in viewer_models[name]["volumes"]]
        if run.can_generate or run.stored_seeds:
            runs.append(run)
    return runs


def stored_volume(run: Run, seed: int, viewer_data: Path = VIEWER_DATA) -> np.ndarray:
    """An evaluation volume from disk, or from the viewer export when the run folder is not available."""
    tif = run.run_dir / "volumes" / f"volume_64_seed{seed}.tif"
    if tif.exists():
        return (tifffile.imread(tif) > 127).astype(np.uint8)
    for ds in _viewer_payload(viewer_data)["datasets"]:
        if ds["dataset"] != run.dataset:
            continue
        for m in ds["models"]:
            if m["id"] == run.run_name:
                for v in m["volumes"]:
                    if v["seed"] == seed:
                        return unpack_bits(v["bits"], tuple(v["shape"]))
    raise FileNotFoundError(f"No stored volume for {run.dataset}/{run.run_name} seed {seed}")


def training_map(run: Run, viewer_data: Path = VIEWER_DATA) -> np.ndarray | None:
    """The 2D label map the run was trained on (Otsu, or SAM for M3)."""
    branch = run.cfg["data"]["branch"]
    path = Path(run.cfg["data"]["train_dirs"][branch]) / "image.png"
    if path.exists():
        from src.models.slicegan_wrapper import load_label_map

        return load_label_map(path)
    for ds in _viewer_payload(viewer_data)["datasets"]:
        if ds["dataset"] == run.dataset:
            t = ds["train"].get(branch) or ds["train"].get("raw")
            return unpack_bits(t["bits"], tuple(t["shape"])) if t else None
    return None


def load_curves(run: Run, reference: str = "common") -> dict[str, np.ndarray] | None:
    """Mean S2 / L curves of the evaluation volumes and of the reference (curves.npz written by generate)."""
    path = run.run_dir / "curves.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {"r": z["r"], "s2_ref": z[f"{reference}_s2_train"], "s2": z[f"{reference}_s2_generated"],
            "L_ref": z[f"{reference}_L_train"], "L": z[f"{reference}_L_generated"]}


def reference_curves(train: np.ndarray) -> dict[str, np.ndarray]:
    """S2 and L of a 2D map (the per-volume comparison in the app uses the run's own training map)."""
    from src.features.descriptors import lineal_path, s2_radial

    return {"s2": s2_radial(train, RMAX), "L": lineal_path(train, RMAX)}


def volume_metrics(vol: np.ndarray, ref: dict[str, np.ndarray]) -> dict[str, float]:
    """φ, S2 / L MAE and error rates of one volume (mean over all slices) against 2D reference curves."""
    s2 = s2_volume(vol, RMAX)
    lp = lineal_path_volume(vol, RMAX)
    out = {"phi": phase_fraction(vol), "s2_mae": s2_mae(s2["mean"], ref["s2"]),
           "s2_err": relative_error(s2["mean"], ref["s2"]), "L_mae": s2_mae(lp["mean"], ref["L"]),
           "L_err": relative_error(lp["mean"], ref["L"])}
    for plane in ("xy", "xz", "yz"):
        out[f"s2_mae_{plane}"] = s2_mae(s2[plane], ref["s2"])
    out["curves"] = {"s2": s2["mean"], "L": lp["mean"]}
    return out


def metrics_table(dataset: str, paths: tuple[Path, ...] = METRICS_CSVS) -> list[dict]:
    """Test metrics (common reference) of every run of a dataset: main table plus the repeat-run table."""
    rows, seen = [], set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if "dataset" in r and (r["dataset"] != dataset or r.get("reference") != "common"):
                    continue
                if "dataset" not in r and dataset != "microlib_000210":  # repeat-run table is MicroLib only
                    continue
                run = r.get("model") if "dataset" in r else r["run"]
                if run in seen:
                    continue
                seen.add(run)
                rows.append({
                    "run": run, "model": MODEL_STYLE.get(run, {"label": run})["label"],
                    "phi": float(r.get("phi_mean") or r.get("phi")),
                    "|dphi|": float(r.get("dphi") or r.get("abs_dphi")),
                    "S2 MAE": float(r["s2_mae"]), "L MAE": float(r["L_mae"]),
                    "S2 MAE 95% CI": f"[{float(r['s2_mae_lo']):.4f}, {float(r['s2_mae_hi']):.4f}]",
                    "typical S2": float(r.get("late_s2_median") or "nan"),
                })
    order = {m: i for i, m in enumerate(MODEL_STYLE)}
    return sorted(rows, key=lambda r: order.get(r["run"], 99))


def surface_mesh(vol: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle surface of the inclusion phase (marching cubes on the zero-padded volume).

    Returns vertices (V, 3) in (x, y, z) voxel coordinates and faces (F, 3).
    """
    from skimage.measure import marching_cubes

    padded = np.pad(vol.astype(np.float32), 1)
    if padded.max() == padded.min():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step)
    verts = verts - 1.0  # undo the padding offset
    return verts[:, ::-1].copy(), faces  # (z, y, x) -> (x, y, z)


def rve_zip(vol: np.ndarray, stem: str, formats: list[str], spacing: float, meta: dict) -> bytes:
    """All requested RVE files of one volume (export_rve writers) plus a JSON metadata file, as a zip."""
    from src.models.export_rve import write_rve

    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        files = write_rve(vol, Path(tmp), stem, formats, spacing)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in files:
                zf.write(Path(tmp) / name, name)
            zf.writestr(f"{stem}.json", json.dumps({**meta, "files": files}, indent=2))
    return buf.getvalue()
