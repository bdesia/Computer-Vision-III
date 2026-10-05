"""Tests for the viewer export and the Streamlit app's data helpers (no Streamlit needed)."""

import csv
import io
import zipfile

import numpy as np
import tifffile
import yaml
from PIL import Image

from src.visualization import app_data as ad
from src.visualization.export_viewer import export_dataset, pack_bits, unpack_bits


def _sphere(n=32, r=8):
    z, y, x = np.mgrid[:n, :n, :n]
    return (((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2) < r**2).astype(np.uint8)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    vol = (rng.random((5, 7, 9)) < 0.3).astype(np.uint8)
    assert np.array_equal(unpack_bits(pack_bits(vol), vol.shape), vol)


def test_surface_mesh_closes_inclusions_and_handles_empty_volume():
    verts, faces = ad.surface_mesh(_sphere())
    assert len(faces) > 100 and verts.min() >= -0.5 and verts.max() <= 31.5
    centre = verts.mean(axis=0)
    assert np.allclose(centre, 16, atol=0.5)  # symmetric sphere, (x, y, z) order
    v0, f0 = ad.surface_mesh(np.zeros((8, 8, 8), np.uint8))
    assert v0.shape == (0, 3) and f0.shape == (0, 3)


def test_volume_metrics_against_its_own_slices_is_small():
    rng = np.random.default_rng(1)
    vol = (rng.random((40, 40, 40)) < 0.25).astype(np.uint8)
    ref = ad.reference_curves(vol[20])
    m = ad.volume_metrics(vol, ref)
    assert abs(m["phi"] - vol.mean()) < 1e-12
    assert m["s2_mae"] < 0.01 and m["L_mae"] < 0.01
    assert set(m["curves"]) == {"s2", "L"} and len(m["curves"]["s2"]) == ad.RMAX + 1


def test_rve_zip_contains_every_format_and_metadata():
    data = ad.rve_zip(_sphere(16, 4), "rve_t", ["vti", "mhd", "npy"], 0.5, {"seed": 1})
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert {"rve_t.vti", "rve_t.mhd", "rve_t.raw", "rve_t.npy", "rve_t.json"} <= names


def test_metrics_table_merges_main_and_repeat_tables(tmp_path):
    main = tmp_path / "metrics.csv"
    with main.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["dataset", "model", "reference", "phi_mean", "dphi", "s2_mae", "L_mae", "s2_mae_lo",
                                "s2_mae_hi", "late_s2_median"])
        w.writeheader()
        w.writerow({"dataset": "microlib_000210", "model": "m1_cnn", "reference": "common", "phi_mean": 0.23,
                    "dphi": 0.001, "s2_mae": 0.003, "L_mae": 0.001, "s2_mae_lo": 0.001, "s2_mae_hi": 0.009,
                    "late_s2_median": 0.005})
        w.writerow({"dataset": "microlib_000210", "model": "m1_cnn", "reference": "train", "phi_mean": 0.1,
                    "dphi": 0.1, "s2_mae": 0.1, "L_mae": 0.1, "s2_mae_lo": 0, "s2_mae_hi": 1, "late_s2_median": 0})
    seeds = tmp_path / "seeds.csv"
    with seeds.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["run", "phi", "abs_dphi", "s2_mae", "L_mae", "s2_mae_lo", "s2_mae_hi",
                                "late_s2_median"])
        w.writeheader()
        w.writerow({"run": "m1_cnn", "phi": 0.5, "abs_dphi": 0.5, "s2_mae": 0.5, "L_mae": 0.5, "s2_mae_lo": 0,
                    "s2_mae_hi": 1, "late_s2_median": 0.5})  # duplicate of the main table: ignored
        w.writerow({"run": "m1_cnn_seed2", "phi": 0.22, "abs_dphi": 0.01, "s2_mae": 0.005, "L_mae": 0.004,
                    "s2_mae_lo": 0.001, "s2_mae_hi": 0.01, "late_s2_median": 0.012})
    rows = ad.metrics_table("microlib_000210", (main, seeds))
    assert [r["run"] for r in rows] == ["m1_cnn", "m1_cnn_seed2"]
    assert rows[0]["S2 MAE"] == 0.003 and rows[1]["model"] == "M1 CNN (run 2)"
    assert ad.metrics_table("synthetic", (main, seeds)) == []


def test_export_dataset_uses_each_models_training_map_and_skips_missing_runs(tmp_path):
    raw, sam = tmp_path / "raw", tmp_path / "sam"
    for folder, value in ((raw, 0), (sam, 255)):
        folder.mkdir()
        img = np.zeros((70, 80), np.uint8)
        img[:10] = value
        Image.fromarray(img).save(folder / "image.png")
    data = {"name": "toy", "source": "synthetic", "train_dirs": {"raw": str(raw), "sam": str(sam)}, "branch": "raw"}
    run_dir = tmp_path / "models" / "m1_cnn"
    (run_dir / "volumes").mkdir(parents=True)
    tifffile.imwrite(run_dir / "volumes" / "volume_64_seed3.tif", (_sphere(8, 3) * 255).astype(np.uint8))
    (run_dir / "metrics.yaml").write_text(yaml.safe_dump(
        {"checkpoint": "G_best.pt", "references": {"common": {"s2_mae": 0.01, "phi_mean": 0.2}}}), encoding="utf-8")
    cfgs = [{"data": data, "paths": {"models": str(tmp_path / "models")}, "run_name": name,
             "model": {"discriminator": "cnn"}} for name in ("m1_cnn", "m2_swin")]
    out = export_dataset(cfgs)
    assert set(out["train"]) == {"raw", "sam"} and out["train"]["sam"]["phi"] > 0
    assert [m["id"] for m in out["models"]] == ["m1_cnn"]  # m2_swin has no volumes
    m = out["models"][0]
    assert m["volumes"][0]["seed"] == 3 and m["metrics"]["s2_mae"] == 0.01 and m["checkpoint"] == "G_best.pt"
    assert np.array_equal(unpack_bits(m["volumes"][0]["bits"], (8, 8, 8)), _sphere(8, 3))
