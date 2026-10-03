"""Tests for the optional MLflow tracker and the backfill of finished runs (temporary file store)."""

import csv
from pathlib import Path

import mlflow
import yaml

from src.tracking import RUN_ID_FILE, Tracker, backfill_run, flatten


def _cfg(tmp_path, enabled):
    return {"run_name": "m1_cnn", "data": {"name": "toy", "branch": "raw"},
            "model": {"discriminator": "cnn", "swin": {"head": "linear"}}, "train": {"lr_d": 1e-4},
            "tracking": {"mlflow": enabled, "uri": (tmp_path / "mlruns").as_uri(), "version": "vtest"}}


def test_flatten_nested_config():
    assert flatten({"a": 1, "b": {"c": [1, 2], "d": {"e": None}}}) == {"a": "1", "b.c": "[1, 2]", "b.d.e": "None"}


def test_disabled_tracker_is_a_noop(tmp_path):
    t = Tracker(_cfg(tmp_path, False), tmp_path / "run")
    t.start(); t.metrics({"x": 1.0}, step=1); t.end()
    assert not (tmp_path / "mlruns").exists() and not (tmp_path / "run" / RUN_ID_FILE).exists()


def test_live_run_then_resume_adds_eval_metrics(tmp_path):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    cfg = _cfg(tmp_path, True)
    t = Tracker(cfg, run_dir); t.start()
    t.metrics({"val_s2_mae": 0.01, "note": "skip"}, step=1); t.end()
    t2 = Tracker(cfg, run_dir); t2.start(resume=True); t2.metrics({"eval_common_s2_mae": 0.005}); t2.end()
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    run = mlflow.get_run((run_dir / RUN_ID_FILE).read_text())
    assert run.data.metrics["val_s2_mae"] == 0.01 and run.data.metrics["eval_common_s2_mae"] == 0.005
    assert run.data.tags["version"] == "vtest" and run.data.params["train.lr_d"] == "0.0001"


def test_backfill_imports_logs_once(tmp_path):
    run_dir = tmp_path / "models" / "m1_cnn"; run_dir.mkdir(parents=True)
    cfg = _cfg(tmp_path, False); cfg.pop("tracking")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "g_step", "wasserstein", "gp"]); w.writerow([1, 10, 0.5, 0.1])
    with (run_dir / "selection.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "g_step", "val_phi", "val_s2_mae", "best"]); w.writerow([1, 100, 0.23, 0.02, 1])
    (run_dir / "metrics.yaml").write_text(yaml.safe_dump({"references": {"common": {"s2_mae": 0.01}}}))
    uri = (tmp_path / "mlruns").as_uri()
    run_id = backfill_run(run_dir, "v1", uri)
    assert run_id and backfill_run(run_dir, "v1", uri) is None          # second import skipped
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["eval_common_s2_mae"] == 0.01 and run.data.metrics["val_s2_mae"] == 0.02
    assert run.data.tags["source"] == "backfill" and run.data.tags["version"] == "v1"
