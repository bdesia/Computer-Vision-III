"""Optional MLflow experiment tracking: live logging of runs and backfill of runs already on disk.

Live logging is off unless `tracking.mlflow: true`. Every MLflow call is guarded: a tracking failure is
logged as a warning and never interrupts training. Runs go to a local file store (`mlruns/`, browse with
`mlflow ui`), one experiment per dataset. The MLflow run id of a training run is stored in
`<run_dir>/mlflow_run_id.txt`, so `generate` adds its evaluation metrics to the same MLflow run.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.utils import get_logger

log = get_logger(__name__)

RUN_ID_FILE = "mlflow_run_id.txt"
ARTIFACT_FILES = ("config.yaml", "history.csv", "selection.csv", "selection.yaml", "metrics.yaml",
                  "volume_slices.png")


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into dotted keys with string values (MLflow params)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = str(v)
    return out


def git_commit() -> str:
    """Short git commit of the working tree ('unknown' outside a repo)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def experiment_name(cfg: dict) -> str:
    """One MLflow experiment per dataset."""
    prefix = (cfg.get("tracking") or {}).get("experiment_prefix", "slicegan-vit")
    return f"{prefix}-{cfg['data']['name']}"


def run_tags(cfg: dict, version: str, source: str, run_dir: Path) -> dict[str, str]:
    """Searchable tags shared by live and backfilled runs."""
    swin = cfg.get("model", {}).get("swin", {})
    return {
        "version": version, "dataset": cfg["data"]["name"], "model": cfg["run_name"],
        "discriminator": cfg["model"]["discriminator"], "swin_head": str(swin.get("head", "linear")),
        "loss": str(cfg["train"].get("loss", "wgan-gp")), "training_image": cfg["data"]["branch"],
        "source": source, "run_dir": run_dir.as_posix(), "git_commit": git_commit(),
    }


class Tracker:
    """Thin, failure-tolerant wrapper around one MLflow run (no-op when tracking is disabled)."""

    def __init__(self, cfg: dict, run_dir: Path):
        """Remember the settings; nothing is contacted until start()."""
        tcfg = cfg.get("tracking") or {}
        self.enabled = bool(tcfg.get("mlflow"))
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.uri = tcfg.get("uri", "file:./mlruns")
        self.version = str(tcfg.get("version", "dev"))
        self._mlflow = None

    def _call(self, fn, *args, **kwargs) -> Any:
        """Run an MLflow call; on failure warn once and disable further tracking."""
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # tracking must never break training
            log.warning("MLflow tracking disabled after error: %s", exc)
            self.enabled = False
            return None

    def start(self, resume: bool = False) -> None:
        """Start (or, with resume, reopen) the MLflow run of this run directory."""
        if not self.enabled:
            return

        def _start():
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(experiment_name(self.cfg))
            id_file = self.run_dir / RUN_ID_FILE
            if resume:
                if not id_file.exists():
                    raise FileNotFoundError(f"no {RUN_ID_FILE} in {self.run_dir}")
                mlflow.start_run(run_id=id_file.read_text(encoding="utf-8").strip())
                return
            name = f"{self.version}/{self.cfg['run_name']}"
            run = mlflow.start_run(run_name=name, tags=run_tags(self.cfg, self.version, "live", self.run_dir))
            id_file.write_text(run.info.run_id, encoding="utf-8")
            params = flatten(self.cfg)
            for i in range(0, len(params), 100):  # MLflow limits params per batch
                mlflow.log_params(dict(list(params.items())[i : i + 100]))

        self._call(_start)

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics (non-numeric values are skipped)."""
        numeric = {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and v == v}
        if numeric:
            self._call(lambda: self._mlflow.log_metrics(numeric, step=step))

    def artifacts(self, paths: list[Path], subdir: str | None = None) -> None:
        """Log files / directories that exist."""
        for p in paths:
            p = Path(p)
            if p.is_dir():
                self._call(lambda p=p: self._mlflow.log_artifacts(str(p), artifact_path=subdir or p.name))
            elif p.exists():
                self._call(lambda p=p: self._mlflow.log_artifact(str(p), artifact_path=subdir))

    def end(self, status: str = "FINISHED") -> None:
        """Close the run."""
        self._call(lambda: self._mlflow.end_run(status=status))


# ----------------------------------------------------------------------------- backfill


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def eval_metrics(summary: dict) -> dict[str, float]:
    """Flatten metrics.yaml into eval_<reference>_<metric> scalars."""
    out = {}
    for ref, values in summary.get("references", {}).items():
        for k, v in values.items():
            f = _float(v)
            if f is not None:
                out[f"eval_{ref}_{k}"] = f
    return out


def backfill_run(run_dir: Path, version: str, uri: str = "file:./mlruns", prefix: str = "slicegan-vit",
                 note: str = "") -> str | None:
    """Import one finished run directory (config + logs + metrics + small artifacts) into MLflow."""
    import mlflow
    from mlflow.entities import Metric

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("tracking", {})["experiment_prefix"] = prefix
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(experiment_name(cfg))
    client = mlflow.tracking.MlflowClient()
    existing = client.search_runs([exp.experiment_id], f"tags.run_dir = '{run_dir.as_posix()}'")
    if existing:
        return None  # already imported

    tags = run_tags(cfg, version, "backfill", run_dir)
    tags["git_commit"] = "n/a (backfilled)"
    if note:
        tags["note"] = note
    with mlflow.start_run(run_name=f"{version}/{cfg['run_name']}", tags=tags) as run:
        params = flatten(cfg)
        for i in range(0, len(params), 100):
            mlflow.log_params(dict(list(params.items())[i : i + 100]))
        batch = []
        for row in _read_csv(run_dir / "history.csv"):
            step = int(float(row["g_step"]))
            for k in ("d_real", "d_fake", "wasserstein", "gp", "g_loss", "sec_per_g_step", "aux_real", "aux_fake"):
                f = _float(row.get(k))
                if f is not None:
                    batch.append(Metric(f"train_{k}", f, 0, step))
        for row in _read_csv(run_dir / "selection.csv"):
            step = int(float(row["epoch"]))
            for k in ("val_phi", "val_s2_mae"):
                f = _float(row.get(k))
                if f is not None:
                    batch.append(Metric(k, f, 0, step))
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i : i + 1000])
        if (run_dir / "metrics.yaml").exists():
            mlflow.log_metrics(eval_metrics(yaml.safe_load((run_dir / "metrics.yaml").read_text(encoding="utf-8"))))
        if (run_dir / "selection.yaml").exists():
            sel = yaml.safe_load((run_dir / "selection.yaml").read_text(encoding="utf-8")) or {}
            mlflow.log_metrics({f"best_{k}": f for k, v in sel.items() if (f := _float(v)) is not None})
        for name in ARTIFACT_FILES:
            if (run_dir / name).exists():
                mlflow.log_artifact(str(run_dir / name))
        if (run_dir / "previews").is_dir():
            mlflow.log_artifacts(str(run_dir / "previews"), artifact_path="previews")
        return run.info.run_id


def main() -> None:
    """CLI: backfill run directories into MLflow. Each root is `<dir>=<version>`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="dir=version, e.g. models/archive_v1=v1 (searched recursively)")
    parser.add_argument("--uri", default="file:./mlruns")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    from src.utils import setup_logging

    setup_logging("logs", "mlflow_backfill", "INFO")
    for spec in args.roots:
        root, _, version = spec.partition("=")
        for cfg_path in sorted(Path(root).rglob("config.yaml")):
            run_id = backfill_run(cfg_path.parent, version or "unknown", args.uri, note=args.note)
            log.info("%s %s -> %s", version, cfg_path.parent.as_posix(), run_id or "skipped (exists)")


if __name__ == "__main__":
    main()
