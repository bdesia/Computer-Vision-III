"""SliceGAN explorer (Streamlit): browse and generate 3D microstructures, compare descriptors, export RVEs.

Run from the repository root:  make app   (or: poetry run streamlit run app/streamlit_app.py)

With local checkpoints (models/<dataset>/<run>/) new volumes can be generated for any seed and size,
periodic or not; without them the app falls back to the evaluation volumes or the viewer export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # configs and data paths are relative to the repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.visualization import app_data as ad  # noqa: E402

PHASE, MATRIX = (227, 138, 78), (35, 42, 48)  # inclusion / matrix colours (same as the HTML viewer)
REF_COLOR = "#1a1a19"

st.set_page_config(page_title="SliceGAN explorer", page_icon="🧊", layout="wide")


# ----------------------------------------------------------------------------- cached data


@st.cache_data(show_spinner=False)
def runs_of(dataset: str) -> list[ad.Run]:
    return ad.list_runs(dataset)


def get_run(dataset: str, name: str) -> ad.Run:
    return next(r for r in runs_of(dataset) if r.run_name == name)


@st.cache_resource(show_spinner=False)
def generator(dataset: str, name: str):
    import torch

    from src.models.generate import load_generator

    run = get_run(dataset, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_generator(run.cfg, run.run_dir, device, checkpoint=run.checkpoint), device


@st.cache_data(show_spinner="Generating volume…")
def generated_volume(dataset: str, name: str, seed: int, size: int, periodic: bool) -> np.ndarray:
    from src.models.export_rve import generate_rve

    netG, device = generator(dataset, name)
    return generate_rve(netG, size, seed, get_run(dataset, name).cfg["z_channels"], device, periodic)


@st.cache_data(show_spinner=False)
def stored(dataset: str, name: str, seed: int) -> np.ndarray:
    return ad.stored_volume(get_run(dataset, name), seed)


@st.cache_data(show_spinner=False)
def train_map(dataset: str, name: str) -> np.ndarray | None:
    return ad.training_map(get_run(dataset, name))


@st.cache_data(show_spinner=False)
def ref_curves(dataset: str, name: str) -> dict[str, np.ndarray]:
    return ad.reference_curves(train_map(dataset, name))


@st.cache_data(show_spinner="Computing descriptors…")
def vol_metrics(vol: np.ndarray, dataset: str, name: str) -> dict:
    return ad.volume_metrics(vol, ref_curves(dataset, name))


@st.cache_data(show_spinner="Building surface…")
def mesh(vol: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    return ad.surface_mesh(vol, step)


# ----------------------------------------------------------------------------- drawing helpers


def section_image(labels: np.ndarray, scale: int = 4) -> np.ndarray:
    """RGB image of a 2D label map, enlarged with nearest-neighbour so voxels stay sharp."""
    rgb = np.where(labels[..., None] == 1, np.array(PHASE, np.uint8), np.array(MATRIX, np.uint8))
    return np.kron(rgb, np.ones((scale, scale, 1), np.uint8))


def surface_figure(vol: np.ndarray, cut_z: int | None) -> go.Figure:
    v = vol if cut_z is None else np.where(np.arange(vol.shape[0])[:, None, None] <= cut_z, vol, 0)
    step = 1 if max(v.shape) <= 64 else 2
    verts, faces = mesh(v, step)
    n = max(vol.shape)
    fig = go.Figure()
    if len(faces):
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1],
                                k=faces[:, 2], color="rgb(227,138,78)", flatshading=True,
                                lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.1}, hoverinfo="skip"))
    axis = {"range": [0, n], "showbackground": True, "backgroundcolor": "rgb(236,238,240)", "title": ""}
    fig.update_layout(scene={"xaxis": {**axis, "title": "x"}, "yaxis": {**axis, "title": "y"},
                             "zaxis": {**axis, "title": "z"}, "aspectmode": "cube"},
                      margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=520, showlegend=False)
    return fig


def curves_figure(series: list[tuple[str, np.ndarray, np.ndarray, str, str]], ref: tuple[np.ndarray, np.ndarray],
                  title: str) -> go.Figure:
    fig = go.Figure()
    r, y = ref
    fig.add_trace(go.Scatter(x=r, y=y, name="2D reference", line={"color": REF_COLOR, "width": 3}))
    for name, x, yv, color, dash in series:
        fig.add_trace(go.Scatter(x=x, y=yv, name=name, line={"color": color, "width": 2, "dash": dash}))
    fig.update_layout(title=title, xaxis_title="r [px]", height=380, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      legend={"font": {"size": 11}}, hovermode="x unified")
    return fig


DASH = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}


def style(run: str) -> tuple[str, str]:
    s = ad.MODEL_STYLE.get(run, {"color": "#888888", "ls": "-"})
    return s["color"], DASH.get(s["ls"], "solid")


# ----------------------------------------------------------------------------- sidebar

st.sidebar.title("SliceGAN explorer")
ds_labels = {"microlib_000210": "MicroLib 000210 (real)", "synthetic": "Synthetic (exact ground truth)"}
dataset = st.sidebar.radio("Dataset", ad.dataset_names(), format_func=lambda d: ds_labels.get(d, d))
runs = runs_of(dataset)
if not runs:
    st.error("No trained models found. Train and evaluate models (`make train-all`, `make eval`) or run "
             "`make viewer` to export volumes.")
    st.stop()
labels = {r.run_name: r.label for r in runs}
run = get_run(dataset, st.sidebar.selectbox("Model", list(labels), format_func=labels.get))

sources = (["Generate new volume"] if run.can_generate else []) + (["Evaluation volumes"] if run.stored_seeds else [])
source = st.sidebar.radio("Volume", sources, help="New volumes need the local checkpoint of the run.")
if source == "Generate new volume":
    seed = int(st.sidebar.number_input("Latent seed", min_value=0, max_value=10**9, value=12345, step=1))
    size = st.sidebar.select_slider("Generated edge (voxels)", options=[64, 96, 128], value=64,
                                    help="Larger volumes enlarge the latent input (edge = 32 · latent − 64).")
    periodic = st.sidebar.checkbox("Periodic (RVE for FEM / FFT)", value=False,
                                   help="Tiled latent; the volume is cropped by 2 voxels and tiles seamlessly.")
    vol = generated_volume(dataset, run.run_name, seed, size, periodic)
    vol_desc = f"generated · seed {seed} · {vol.shape[0]}³{' · periodic' if periodic else ''}"
else:
    seed = st.sidebar.selectbox("Evaluation seed", run.stored_seeds[:128])
    periodic = False
    vol = stored(dataset, run.run_name, seed)
    vol_desc = f"evaluation volume · seed {seed} · {vol.shape[0]}³"

st.sidebar.caption(f"Checkpoint: `{run.checkpoint or 'n/a'}` · training map: "
                   f"{'SAM' if run.cfg['data']['branch'] == 'sam' else 'Otsu'}")

# ----------------------------------------------------------------------------- header

st.title(f"{run.label} — {ds_labels.get(dataset, dataset)}")
st.caption(vol_desc)
tab_vol, tab_desc, tab_metrics, tab_export = st.tabs(["Volume", "Descriptors", "Test metrics", "Export RVE"])

# ----------------------------------------------------------------------------- volume tab

with tab_vol:
    n = vol.shape[0]
    c3d, c2d = st.columns([1.15, 1])
    with c2d:
        z = st.slider("xy section (z)", 0, n - 1, n // 2)
        y = st.slider("xz section (y)", 0, n - 1, n // 2)
        x = st.slider("yz section (x)", 0, n - 1, n // 2)
        scale = max(1, 256 // n)
        s1, s2 = st.columns(2)
        s1.image(section_image(vol[z], scale), caption=f"xy · z = {z}")
        s2.image(section_image(vol[::-1, y, :], scale), caption=f"xz · y = {y} (z up)")
        s3, s4 = st.columns(2)
        s3.image(section_image(vol[::-1, :, x], scale), caption=f"yz · x = {x} (z up)")
        tm = train_map(dataset, run.run_name)
        if tm is not None:
            rng = np.random.default_rng(seed)
            r0, c0 = rng.integers(0, tm.shape[0] - 64), rng.integers(0, tm.shape[1] - 64)
            s4.image(section_image(tm[r0:r0 + 64, c0:c0 + 64], 4),
                     caption=f"2D training crop ({'SAM' if run.cfg['data']['branch'] == 'sam' else 'Otsu'})")
    with c3d:
        cut = st.checkbox("Cut away above the xy section", value=False)
        st.plotly_chart(surface_figure(vol, z if cut else None), use_container_width=True)

    m = vol_metrics(vol, dataset, run.run_name)
    st.subheader("This volume vs the model's training map")
    k = st.columns(4)
    k[0].metric("φ", f"{m['phi']:.3f}", f"{m['phi'] - float(tm.mean()):+.3f} vs 2D" if tm is not None else None,
                delta_color="off")
    k[1].metric("S₂ MAE", f"{m['s2_mae']:.4f}", f"{m['s2_err']:.1%} rel. error", delta_color="off")
    k[2].metric("L MAE", f"{m['L_mae']:.4f}", f"{m['L_err']:.1%} rel. error", delta_color="off")
    k[3].metric("S₂ MAE xy / xz / yz", f"{m['s2_mae_xy']:.3f} / {m['s2_mae_xz']:.3f} / {m['s2_mae_yz']:.3f}")
    if run.metrics:
        st.caption(f"Test metrics of this model over {run.metrics.get('n_volumes', 128)} volumes (common reference): "
                   f"φ {run.metrics['phi_mean']:.3f} ± {run.metrics['phi_std']:.3f}, S₂ MAE {run.metrics['s2_mae']:.4f}, "
                   f"L MAE {run.metrics['L_mae']:.4f}. A single 64³ volume is a noisy sample (φ varies by about ±0.05).")

# ----------------------------------------------------------------------------- descriptors tab

with tab_desc:
    default = [r.run_name for r in runs if r.run_name in ("m1_cnn", "m2_swin", "m3_swin_sam", "m4_ensemble")]
    chosen = st.multiselect("Models (mean over the 128 evaluation volumes)", [r.run_name for r in runs],
                            default=default, format_func=lambda nme: get_run(dataset, nme).label)
    show_vol = st.checkbox("Overlay the current volume", value=True)
    s2_series, l_series, ref = [], [], None
    for nme in chosen:
        cv = ad.load_curves(get_run(dataset, nme))
        if cv is None:
            continue
        color, dash = style(nme)
        s2_series.append((get_run(dataset, nme).label, cv["r"], cv["s2"], color, dash))
        l_series.append((get_run(dataset, nme).label, cv["r"], cv["L"], color, dash))
        ref = ref or (cv["r"], cv["s2_ref"], cv["L_ref"])
    if ref is None:
        rc = ref_curves(dataset, run.run_name)
        ref = (np.arange(ad.RMAX + 1), rc["s2"], rc["L"])
    if show_vol:
        rr = np.arange(ad.RMAX + 1)
        s2_series.append((f"current volume ({run.label})", rr, m["curves"]["s2"], "#7a7a7a", "dot"))
        l_series.append((f"current volume ({run.label})", rr, m["curves"]["L"], "#7a7a7a", "dot"))
    a, b = st.columns(2)
    a.plotly_chart(curves_figure(s2_series, (ref[0], ref[1]), "Two-point correlation S₂(r)"), use_container_width=True)
    b.plotly_chart(curves_figure(l_series, (ref[0], ref[2]), "Lineal path L(r)"), use_container_width=True)
    st.caption("Reference: the common reference of the dataset (MicroLib: Otsu map; synthetic: exact ground-truth "
               "mask). The current-volume curve is computed live over all its xy, xz and yz slices.")

# ----------------------------------------------------------------------------- metrics tab

with tab_metrics:
    rows = ad.metrics_table(dataset)
    if rows:
        df = pd.DataFrame(rows).drop(columns=["run"]).set_index("model")
        st.dataframe(df.style.format({"phi": "{:.3f}", "|dphi|": "{:.4f}", "S2 MAE": "{:.4f}", "L MAE": "{:.4f}",
                                      "typical S2": "{:.4f}"}), use_container_width=True)
        st.caption("Test metrics on 128 volumes per run against the common reference, with 95 % bootstrap "
                   "intervals. 'Typical S2' is the median per-epoch held-out score over the second half of training. "
                   "'Run 2' rows are second training runs: the same model trained twice differs as much as the "
                   "models differ from each other (report Section 5.7).")
    else:
        st.info("No metrics tables found (reports/metrics.csv).")

# ----------------------------------------------------------------------------- export tab

with tab_export:
    st.write(f"Current volume: **{vol_desc}**. For a periodic RVE, choose *Generate new volume* and tick *Periodic*.")
    from src.models.export_rve import FORMATS, seam_quality

    formats = st.multiselect("Formats", FORMATS, default=["vti", "mhd", "npy"],
                             help="vti: ParaView / DAMASK · mhd: ITK, FFT solvers · inp: Abaqus voxel mesh (C3D8) · "
                                  "npy: NumPy · tif: image stack")
    spacing = float((run.cfg["data"].get("microlib") or {}).get("pixel_size_um") or 1.0)
    seams = seam_quality(vol)
    st.caption(f"Voxel size {spacing} {'µm' if spacing != 1.0 else 'voxel'} · seam ratios z / y / x "
               f"{seams['seam_ratio_z']:.2f} / {seams['seam_ratio_y']:.2f} / {seams['seam_ratio_x']:.2f} "
               "(about 1 = continues across the boundary as smoothly as inside, i.e. periodic)")
    if formats:
        stem = f"rve_{run.run_name}_{dataset}_{vol.shape[0]}{'p' if periodic else ''}_seed{seed}"
        meta = {"model": run.run_name, "dataset": dataset, "checkpoint": run.checkpoint, "seed": int(seed),
                "shape_zyx": list(vol.shape), "periodic": periodic, "voxel_size": spacing,
                "phase_fractions": {"matrix": float(1 - vol.mean()), "inclusion": float(vol.mean())}, **seams}
        st.download_button("Download RVE (zip)", ad.rve_zip(vol, stem, formats, spacing, meta), f"{stem}.zip",
                           mime="application/zip")
