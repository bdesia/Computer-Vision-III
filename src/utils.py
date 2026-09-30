"""Shared helpers: YAML config loading with inheritance, logging, seeding and device selection."""

from __future__ import annotations

import copy
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER_NAME = "tfvit"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` recursively merged on top of `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional `base:` key relative to the file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path.resolve()}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    base_name = cfg.pop("base", None)
    if base_name is not None:
        cfg = deep_merge(load_config(path.parent / base_name), cfg)
    cfg.setdefault("run_name", path.stem)
    return cfg


def setup_logging(log_dir: str | Path, run_name: str, level: str = "INFO") -> logging.Logger:
    """Configure the project logger to write to console and `<log_dir>/<run_name>.log`."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / f"{run_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the project logger (e.g. `get_logger(__name__)`)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "cuda"):
    """Return the requested torch device, falling back to CPU with a warning."""
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        get_logger(__name__).warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
