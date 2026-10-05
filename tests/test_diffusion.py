"""Tests for M6: DiT denoiser, noise schedule, DDIM, multi-plane 3D sampling and the training / loading path."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.models import diffusion as dm
from src.utils import load_config


def _tiny(img=16):
    return dm.DiT(img_size=img, patch=4, in_ch=1, dim=32, depth=2, heads=2).eval()


def test_dit_output_shape_and_zero_init():
    model = _tiny()
    x = torch.randn(3, 1, 16, 16)
    out = model(x, torch.tensor([0, 10, 999]))
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0  # adaLN-Zero + zero output layer: starts predicting zero noise


def test_dit_unpatchify_inverts_patch_layout():
    model = _tiny()
    torch.manual_seed(0)
    for p in model.parameters():  # non-zero output path so the layout matters
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, 1, 16, 16)
    # a patch-local change must only affect the matching output patch (no cross-patch scrambling by unpatchify)
    model.blocks = torch.nn.ModuleList()  # no attention: tokens stay independent
    with torch.no_grad():
        base = model(x, torch.tensor([5, 5]))
    x2 = x.clone()
    x2[:, :, 4:8, 8:12] += 1.0
    with torch.no_grad():
        diff = (model(x2, torch.tensor([5, 5])) - base).abs().sum(dim=(0, 1))
    assert diff[4:8, 8:12].sum() > 0 and diff.sum().item() == pytest.approx(diff[4:8, 8:12].sum().item())


def test_cosine_schedule_and_ddim_timesteps():
    ab = dm.cosine_alpha_bar(1000)
    assert ab.shape == (1000,) and ab[0] > 0.999 and ab[-1] < 1e-3
    assert torch.all(ab[1:] <= ab[:-1])
    ts = dm.ddim_timesteps(1000, 50)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 50 and ts == sorted(ts, reverse=True)


def test_ddim_step_with_true_noise_recovers_x0():
    ab = dm.cosine_alpha_bar(100)
    x0 = torch.where(torch.rand(4, 1, 8, 8) > 0.5, 1.0, -1.0)
    noise = torch.randn_like(x0)
    t = torch.full((4,), 60)
    xt = dm.q_sample(x0, t, noise, ab)
    out = dm.ddim_step(xt, noise, ab[60], torch.tensor(1.0))
    assert torch.allclose(out, x0, atol=1e-4)


class _Identity(torch.nn.Module):
    """'Denoiser' returning its input, to check that slices are cut and reassembled along the right axis."""

    img_size = 8

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t):
        return x


def test_predict_eps_volume_reassembles_slices_along_each_axis():
    vol = torch.randn(8, 8, 8)
    for axes in (("z",), ("y",), ("x",), dm.AXES):
        out = dm.predict_eps_volume(_Identity(), vol, 3, axes, chunk=3)
        assert torch.allclose(out, vol)


def test_sample_volume_is_binary_deterministic_and_validates_mode():
    model = _tiny()
    ab = dm.cosine_alpha_bar(50)
    a = dm.sample_volume(model, ab, 16, seed=3, steps=5, mode="average")
    b = dm.sample_volume(model, ab, 16, seed=3, steps=5, mode="average")
    c = dm.sample_volume(model, ab, 16, seed=3, steps=5, mode="cycle")
    assert a.shape == (16, 16, 16) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1}
    assert np.array_equal(a, b) and c.shape == a.shape
    with pytest.raises(ValueError):
        dm.sample_volume(model, ab, 16, seed=3, steps=5, mode="random")
    s = dm.sample_slices(model, ab, 4, seed=0, steps=5)
    assert s.shape == (4, 16, 16)


def test_train_then_generate_end_to_end_on_cpu(tmp_path):
    from src.models.generate import generate_volumes, load_generator
    from src.models.train_diffusion import train

    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(((rng.random((40, 40)) < 0.3) * 255).astype(np.uint8)).save(raw / "image.png")
    cfg = load_config("configs/m6_dit.yaml")
    cfg.update({"img_size": 16, "volume_size": 16, "epochs": 1, "iters_per_epoch": 3, "device": "cpu"})
    cfg["paths"]["models"] = str(tmp_path / "models")
    cfg["data"]["train_dirs"] = {"raw": str(raw)}
    cfg["model"]["dit"].update({"dim": 32, "depth": 1, "heads": 2, "timesteps": 50, "sample_steps": 4})
    cfg["train"].update({"batch": 4, "log_every": 1, "val_volumes": 2, "val_steps": 3, "snapshot_every": 1})
    cfg["metrics"]["s2_rmax"] = 6
    cfg["tracking"]["mlflow"] = False
    run_dir = train(cfg)
    for name in ("G_best.pt", "G_last.pt", "snapshots/G_epoch001.pt", "history.csv", "selection.csv",
                 "selection.yaml", "previews/epoch_001.png", "previews/epoch_001_2d.png"):
        assert (run_dir / name).exists(), name
    gen = load_generator(cfg, run_dir, torch.device("cpu"), checkpoint="snapshots/G_epoch001.pt")
    vols = generate_volumes(gen, [0, 1], cfg["z_channels"], torch.device("cpu"))
    assert len(vols) == 2 and vols[0].shape == (16, 16, 16)
