"""Figures: orthogonal slices of generated volumes (more plots are added in the evaluation step)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_volume_slices(volume: np.ndarray, path: str | Path, title: str | None = None) -> None:
    """Save the central xy / xz / yz slices of a (Z, Y, X) label volume side by side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z, y, x = (s // 2 for s in volume.shape)
    panels = {"xy": volume[z], "xz": volume[:, y, :], "yz": volume[:, :, x]}

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (name, img) in zip(axes, panels.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
