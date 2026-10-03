"""Tests for RVE export: sizes, periodic generation, seam metric and file formats (read back)."""

import base64
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.export_rve import (
    generate_rve,
    latent_size_for,
    seam_quality,
    write_abaqus_inp,
    write_mhd,
    write_rve,
    write_vti,
)
from src.models.slicegan_wrapper import build_generator


def _vol(shape=(4, 5, 6), seed=0):
    return (np.random.default_rng(seed).random(shape) < 0.3).astype(np.uint8)


def test_latent_size_formula():
    assert [latent_size_for(s) for s in (64, 96, 128, 256)] == [4, 5, 6, 10]
    with pytest.raises(ValueError):
        latent_size_for(100)


def test_seam_quality_separates_periodic_from_non_periodic():
    n = 48
    z, y, x = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
    field = lambda period: (np.sin(2 * np.pi * x / period) + np.sin(2 * np.pi * y / period)  # noqa: E731
                            + np.sin(2 * np.pi * z / period) > 0.6).astype(np.uint8)
    periodic = seam_quality(field(24))          # period divides the box: continuous across the seam
    non_periodic = seam_quality(field(37))      # period does not divide the box: jump at the seam
    assert all(r < 1.5 for r in periodic.values())
    assert all(r > 3.0 for r in non_periodic.values())


def test_vti_roundtrip(tmp_path):
    vol = _vol()
    write_vti(vol, tmp_path / "a.vti", 0.5)
    text = (tmp_path / "a.vti").read_text()
    assert 'WholeExtent="0 6 0 5 0 4"' in text and 'Spacing="0.5 0.5 0.5"' in text
    raw = base64.b64decode(re.search(r'format="binary">([^<]+)<', text).group(1))
    n = struct.unpack("<I", raw[:4])[0]
    back = np.frombuffer(raw[4:4 + n], dtype="<i4").reshape(vol.shape)
    assert np.array_equal(back, vol)


def test_mhd_roundtrip(tmp_path):
    vol = _vol()
    write_mhd(vol, tmp_path / "a.mhd", 0.687)
    hdr = (tmp_path / "a.mhd").read_text()
    assert "DimSize = 6 5 4" in hdr and "ElementType = MET_UCHAR" in hdr
    back = np.fromfile(tmp_path / "a.raw", dtype=np.uint8).reshape(vol.shape)
    assert np.array_equal(back, vol)


def test_abaqus_inp_mesh(tmp_path):
    vol = _vol((3, 4, 5))
    counts = write_abaqus_inp(vol, tmp_path / "a.inp", 2.0, "t")
    text = (tmp_path / "a.inp").read_text()
    node_block = text.split("*Node\n")[1].split("*Element")[0].strip().splitlines()
    elem_block = text.split("*Element, type=C3D8\n")[1].split("*Elset")[0].strip().splitlines()
    assert len(node_block) == 6 * 5 * 4 and len(elem_block) == 60
    assert counts == {"matrix": int((vol == 0).sum()), "inclusion": int((vol == 1).sum())}
    # first element: nodes of voxel (0,0,0), last node at the far corner (5*2, 4*2, 3*2)
    assert [int(v) for v in elem_block[0].split(",")] == [1, 1, 2, 8, 7, 31, 32, 38, 37]
    assert node_block[-1].replace(" ", "") == "120,10,8,6"
    for face in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"):
        assert f"*Nset, nset={face}" in text


def test_generate_and_write_periodic_rve_cpu(tmp_path):
    g = build_generator({"n_phases": 2, "z_channels": 32}, tmp_path, training=True).eval()
    vol = generate_rve(g, 64, seed=3, z_channels=32, device="cpu", periodic=True)
    assert vol.shape == (62, 62, 62) and vol.dtype == np.uint8
    assert generate_rve(g, 64, seed=3, z_channels=32, device="cpu", periodic=False).shape == (64, 64, 64)
    files = write_rve(vol[:8, :8, :8], tmp_path / "out", "r", ["vti", "mhd", "npy", "tif"], 1.0)
    assert set(files) == {"r.vti", "r.mhd", "r.raw", "r.npy", "r.tif"}
