# SliceGAN with a Vision Transformer discriminator and a SAM front-end

Final project — **Vision Transformers (FIUBA)**. Instructors: Abraham Rodriguez, Oksana Bokhonok.
Individual work.

## Goal

Generate 3D volumes (64³) that are statistically equivalent to a 2D micrograph of a two-phase,
isotropic material using **SliceGAN**, and measure:

1. whether a **Swin-T discriminator** (hierarchical Vision Transformer) improves on the original CNN
   discriminator (RQ1),
2. whether using **SAM** as a phase-segmentation front-end improves the generated volumes (RQ2), and
3. whether a pretrained Swin-T helps as an **additional** critic next to the CNN, as in Vision-aided GAN
   (RQ3, extension).

Quality is measured with the phase fraction `φ`, the two-point correlation `S₂(r)` and the lineal path `L(r)`.

| ID | Model | 2D input | 3D generator | 2D critic |
|----|-------|----------|--------------|-----------|
| M1 | SliceGAN baseline | Otsu map | SliceGAN 3D CNN | SliceGAN CNN |
| M1 + DiffAug | ablation | Otsu map | same G | SliceGAN CNN + DiffAug |
| M2 | SliceGAN–Swin | Otsu map | same G | Swin-T (timm, ImageNet), stages 3–4 trainable, DiffAug |
| M3 | SliceGAN–Swin+SAM | SAM phase map | same G | same critic as M2 |
| M4 | ensemble | Otsu map | same G | SliceGAN CNN + frozen Swin-T with per-position heads |
| M5 | fine-tune | Otsu map | M1's G + 20 epochs | as M4 |
| M1-ext | fair baseline for M5 | Otsu map | M1's G + 20 epochs | SliceGAN CNN |

- M1 vs M2 (and M1 + DiffAug vs M2): does the ViT critic help?
- M2 vs M3: does SAM preprocessing help?
- M1 vs M4, M1-ext vs M5: does a Swin critic help next to the CNN?

## Layout

```
configs/            default.yaml + one yaml per model (m1_cnn, m1_cnn_diffaug, m1_extended, m2_swin,
                    m3_swin_sam, m4_ensemble, m5_finetune), probe configs, data/ dataset overlays
data/               raw/ interim/ processed/<dataset>/ (not versioned, except sam_gt/ READMEs + GT crops)
external/SliceGAN/  upstream SliceGAN (git submodule, unmodified)
src/data/           make_dataset.py — downloads or generates the 2D image and 64x64 crops
src/features/       sam_segment.py (zero-shot SAM + reference scores), descriptors.py (φ, S₂, L, bootstrap)
src/models/         SliceGAN wrapper, CNN / Swin critics, DiffAug, train.py, generate.py, linear_probe.py
src/visualization/  figures, metrics.csv, comparison vs M1, viewer export, app data helpers, report PDF
app/                streamlit_app.py — interactive explorer (make app)
src/tracking.py     optional MLflow tracking and backfill of finished runs
tests/              pytest (91 tests)
notebooks/          00_eda.ipynb — training images, Otsu vs SAM, crop statistics, 2D descriptors
reports/            report.md / report_es.md (+ PDFs), figures/, metrics*.csv, comparison_vs_m1_*.csv,
                    probes/ and runs/ (per-epoch selection logs), viewer/ (interactive volume viewer)
models/             checkpoints per dataset/run + archives v1–v5 and probes (not versioned)
logs/               per-run logs (not versioned)
mlruns/             MLflow store (not versioned; `make mlflow-backfill`, `make mlflow-ui`)
```

## Setup

Requirements: Python 3.11, [Poetry](https://python-poetry.org/) ≥ 2.0, bash (Git Bash on Windows).
An NVIDIA GPU is recommended (tested on an RTX A2000 12 GB).

```bash
git clone --recurse-submodules https://github.com/bdesia/Computer-Vision-III.git
cd Computer-Vision-III
bash setup.sh               # CUDA 12.4 build of PyTorch
# DEVICE=cpu bash setup.sh  # CPU-only build
```

`setup.sh` creates an in-project `.venv`, installs the locked dependencies (main + dev), registers a
Jupyter kernel named `tf-vit-slicegan` and adds the repo root to the environment's `sys.path`.
Run commands with `poetry run ...` or activate `.venv` first.
If you cloned without submodules: `make vendor`.

Without a GPU everything runs on CPU (a WARNING is logged); lower `epochs` in the yaml in that case.

## Data

Two datasets are supported. Every command takes an optional dataset overlay with `--data`
(`DATA=...` in the Makefile); without it the synthetic dataset is used. Each dataset gets its own
folders (`data/*/<name>/`, `models/<name>/`, `logs/<name>/`) via the `{data_name}` placeholder.

```bash
poetry run python -m src.data.make_dataset                                              # synthetic
poetry run python -m src.data.make_dataset --data configs/data/microlib_000210.yaml     # real (main case)
# make data   /   make data DATA=configs/data/microlib_000210.yaml
```

**Label probe — `synthetic_sam`** (`configs/data/synthetic_sam.yaml`): a label probe, not a training set.

**Real micrograph — MicroLib 000210** (`configs/data/microlib_000210.yaml`, main case). An optical
micrograph from the [DoITPoMS micrograph library](https://www.doitpoms.ac.uk/miclib/) as curated in
[MicroLib](https://microlib.io) (Kench et al., 2022): dark islands in a light matrix. It was chosen at
random (seed-42 shuffle) among the MicroLib `twophase` entries whose annotated phase gray levels
differ by ≥ 80, and then checked for isotropy (x/y correlation-length ratio 1.11 after thresholding).
`make_dataset` downloads it from DoITPoMS, crops away the scale bar (rows 437–525, MicroLib
`barbox`) and keeps 437 × 800 px at 0.687 µm/px. Otsu gives `φ = 0.232` and 288 crops of 64×64.
DoITPoMS images are for educational/non-commercial use, so they are downloaded, not committed.

**Synthetic** (`data.name: synthetic`, default; debug and sanity check): non-overlapping discs
placed by random sequential adsorption (RSA), target `φ` 0.25, 512², fixed seed. The discs are
rendered as a blurred, noisy grayscale micrograph so that segmentation (Otsu for M1/M2, SAM for M3)
is non-trivial; the clean mask is kept as ground truth. `phi_true = 0.2505`, `phi_train = 0.2540`,
Otsu IoU vs GT = 0.913, 225 crops.

(`data.source: slicegan` reads `external/SliceGAN/Examples/<file>`, but upstream only ships
`NMC.tif`, which has 3 phases and is out of scope.)

Outputs, with `<name>` = `synthetic` or `microlib_000210`:

| Path | Content |
|------|---------|
| `data/raw/<name>/micro_2d.png` | grayscale micrograph (input to Otsu and SAM) |
| `data/raw/<name>/micro_2d.json` | provenance (MicroLib URL and crop) or synthetic `phi_true` |
| `data/raw/synthetic/micro_2d_gt.png` | synthetic only: clean ground-truth mask |
| `data/interim/<name>/micro_2d_gray.png`, `micro_2d_otsu.png` | normalized grayscale and Otsu label map |
| `data/processed/<name>/train_2d/image.png` | M1/M2 training label map (0 = matrix, 1 = inclusion) |
| `data/processed/<name>/train_2d/crops.npy` | 64×64 crops, stride 32, `(N, 64, 64)` uint8 |
| `data/processed/<name>/train_2d/meta.yaml` | `φ`, threshold, crop stats (+ Otsu IoU vs GT for synthetic) |

## Training

| Target | Model | Config |
|--------|-------|--------|
| `make train-m1` | M1: SliceGAN baseline (CNN critic) | `configs/m1_cnn.yaml` |
| `make train-m1-diffaug` | M1 + DiffAug (ablation) | `configs/m1_cnn_diffaug.yaml` |
| `make train-m1-ext` | M1-extended: M1's best G + 20 epochs, CNN only (baseline for M5) | `configs/m1_extended.yaml` |
| `make train-m2` | M2: Swin-T critic + DiffAug | `configs/m2_swin.yaml` |
| `make train-m3` | M3: as M2, on the SAM phase map (run `make sam` first) | `configs/m3_swin_sam.yaml` |
| `make train-m4` | M4: CNN + frozen-Swin critics (ensemble), from scratch | `configs/m4_ensemble.yaml` |
| `make train-m5` | M5: M1's best G + 20 epochs with the M4 ensemble (needs M1) | `configs/m5_finetune.yaml` |
| `make train-all` | all seven, M1 first | — |

Every target takes `DATA=configs/data/microlib_000210.yaml` for the real micrograph (default: synthetic),
e.g. `make train-all DATA=configs/data/microlib_000210.yaml`. `make train-<config>` trains any other config
(e.g. the probes). The underlying command is
`python -m src.models.train --config configs/<config>.yaml [--data <overlay>]`.

Quick smoke test (runs on CPU too):

```bash
poetry run python -m src.models.train --config configs/m1_cnn.yaml --epochs 2 --iters-per-epoch 1 --device cpu
```

`--epochs`, `--iters-per-epoch` and `--device` override the yaml. `train.max_minutes` sets an optional
wall-clock budget (training stops cleanly and checkpoints).

Each run writes to `models/<name>/<run_name>/`: `G_last.pt`, `D_last.pt`, the resolved `config.yaml`,
`history.csv` (critic real/fake scores, Wasserstein estimate, gradient penalty, G loss, s/step),
`previews/epochNNN.png` (central xy/xz/yz slices) and SliceGAN's `slicegan_params.data`.
Logs go to `logs/<name>/<run_name>.log`.

### How SliceGAN is integrated

`external/SliceGAN` is the unmodified upstream repo (git submodule, MIT). `src/models/slicegan_wrapper.py`
builds the generator and the CNN critic with upstream's `slicegan_rc_nets` and the exact layer lists of
`run_slicegan.py` (`z_channels = 32`, 4³ latent → 64³ volume), and reuses upstream's gradient penalty.
`src/models/train.py` is a fork of upstream `model.train` with the same WGAN-GP schedule (Adam
1e-4, β = (0.9, 0.99), λ = 10, 5 critic steps per G step, one isotropic critic shared by the three
axes). The differences are: config/logging/checkpointing, a pluggable critic (CNN or Swin-T), random
64×64 crops sampled on the fly on the GPU instead of 28 800 pre-built crops in RAM, and the paper's
`m_G = 2 m_D` batches with all slices (see below), and rigid augmentation of the real crops.

Upstream only ships `Examples/NMC.tif` (3 phases, out of scope), so the real 2D image has to come
from MicroLib.

### Swin-T critic (M2/M3)

`src/models/discriminator_swin.py` wraps the Microsoft Swin-T ImageNet-1k checkpoint
(`timm/swin_tiny_patch4_window7_224.ms_in1k` on the Hugging Face Hub, same weights as
`microsoft/swin-tiny-patch4-window7-224`) as a WGAN critic:

- **Native 64×64 input.** Swin-T has no absolute position embedding, only a relative-position bias
  inside each window, so it does not need 224×224 input. At 64 px the stage resolutions are
  16/8/4/2 and timm shrinks the last two windows to 4 and 2, resizing the pretrained bias table.
  (HF `transformers` 4.46 crashes on this case, hence timm.) This is ~9× cheaper than upsampling to
  224; `model.swin.input_size: 224` switches to bilinear upsampling if needed.
- **Two-phase patch embedding.** The RGB patch filter is replaced by a 2-channel one equivalent to
  feeding the pretrained filter a centred gray image (matrix −0.5, inclusion +0.5). Averaging the RGB
  weights would make one-hot inputs indistinguishable, since both channels always sum to 1.
- **Head:** global average pooling + linear → one unbounded score per slice. Stochastic depth and
  dropout are off. Patch embedding and stages 1–2 are frozen, stages 3–4, the final norm and the head
  are trained (26.3 M trainable parameters vs 2.8 M for the CNN critic).

**Same training protocol for all models (SliceGAN paper, Algorithm 1).** Each critic step generates
`m_D = 1` volume and shows the critic **all** 64 slices along each of the three axes; each generator
step uses `m_G = 2 m_D = 2` volumes, again with all slices (the paper finds `m_G = 2 m_D` most
efficient and that D must see all 64 slices per direction for every path through G to be trained).
Upstream code instead generates 8 volumes per step; `m_D = 1` keeps the Swin critic affordable while
preserving the "all slices" rule. Real crops (8 per axis, also used for the gradient penalty as
upstream) are augmented with random 90° rotations and flips only. TF32 matmuls are enabled for
every model. Measured on an RTX A2000 12 GB with MicroLib 000210: 0.30 s per G step for M1 and
1.9 s for M2, i.e. ~25 min vs ~2.6 h for the default 50 × 100 G steps.

**Swin critic stabilization.** Transformer critics are known to make GAN training unstable
(ViTGAN, Lee et al., ICLR 2022). Ten settings were tried on MicroLib 000210, scoring the generator after
every epoch on held-out seeds (φ and S₂ MAE vs the training image, target φ = 0.232; first 12 epochs):

| Run | Swin critic | Loss | DiffAug | Outcome |
|-----|-------------|------|---------|---------|
| v1 | stages 3–4 trainable, lr 1e-4 (as M1) | WGAN-GP | no | oscillates between realistic and empty volumes; ended collapsed |
| v2 | stages 3–4, lr 2e-5 | WGAN-GP | no | still oscillates, but reaches M1-level epochs |
| v2 rerun | identical | WGAN-GP | no | different trajectory (non-deterministic GPU kernels amplified by GAN dynamics) |
| v3 | v2 + ViTGAN: improved spectral norm, Adam β = (0, 0.99), G EMA | WGAN-GP | no | critic too strong; near-empty volumes |
| stage-4 probe | only stage 4 trainable | WGAN-GP | no | no full collapse, but blurry |
| A / C | frozen backbone, pooled (A) or per-position (C) heads | WGAN-GP | yes | fail: the gradient penalty cannot be met through a frozen backbone |
| A-hinge / C-hinge | frozen backbone, heads with spectral norm | hinge | yes | no collapse, but φ uncontrolled without a CNN critic |
| **B (final M2/M3)** | stages 3–4, lr 2e-5 | WGAN-GP | **yes** | **reaches M1-level quality; the only Swin-only critic that works** |

**Final setups.** M2/M3: probe B (Swin-T stages 3–4 + head trainable, lr 2e-5, WGAN-GP, DiffAug).
M4/M5: M1's CNN critic plus the frozen-Swin per-position heads of C-hinge, as in Vision-aided GAN
(Kumari et al., 2022). The generator and its optimizer are identical in every model. The ViTGAN
stabilizers stay in the code behind switches (`model.swin.isn`, `train.betas_d`, `train.ema_decay`),
off by default. Probe logs: `reports/probes/`, figure `reports/figures/stabilization_probes.png`.

**Checkpoint selection and evaluation.** After every epoch the generator produces 16 volumes from
held-out seeds (`train.select_seeds`, 1000–1015); the epoch with the lowest S₂ MAE vs the model's own
training image is saved as `G_best.pt`. That checkpoint is evaluated on 128 *different* seeds
(`generate.seeds`, 0–127) with bootstrap confidence intervals. Single 64³ volumes are small samples
(phase fraction varies by ±0.05 between volumes): earlier protocols with 2 selection and 4 evaluation
seeds (run v5) produced rankings that reversed on more seeds, and choosing the best of 50 noisy epoch
scores is optimistic (winner's curse), which independent evaluation seeds remove from the reported
numbers. A selection-free summary (median held-out S₂ MAE over the second half of training) is reported
as well.

Archived runs: `models/archive_v1/` … `models/archive_v5/` (v1: `reports/metrics_v1.csv`,
`reports/figures/v1/`; v5: `reports/metrics_v5_2seed_selection.csv`, `reports/figures/v5/`) and
`models/archive_probes/`; all of them can be imported into MLflow (`make mlflow-backfill`).

### SAM phase front-end (M3)

`src/features/sam_segment.py` segments the raw grayscale micrograph with SAM ViT-B
(`facebook/sam-vit-base`, Hugging Face `mask-generation` pipeline), zero-shot:

1. **Tiled automatic mask generation.** The particles are small relative to SAM's 1024 px input, so
   the image is split into 256 px tiles (32 px overlap), each with a 32 × 32 point grid. A single
   pass over the 800 px MicroLib image with the brief's 16 points/side found only `φ = 0.07`
   (Otsu: 0.23); tiling raises it to 0.22. (SAM's own multi-crop option is broken in the HF pipeline.)
2. **Merge:** masks with IoU > 0.3 are grouped (union-find over overlapping bounding boxes).
3. **Intensity rule:** the matrix gray level is the image median (majority phase); group mean
   gray levels are split with an area-weighted Otsu threshold and the groups on the far side from
   the matrix are the inclusion phase. Everything else, including pixels no mask covers, is matrix.

```bash
make sam                                          # synthetic
make sam DATA=configs/data/microlib_000210.yaml   # real
```

Outputs: `data/processed/<name>/train_sam/` (`image.png`, `crops.npy`, `meta.yaml`),
`data/interim/<name>/sam_overlay.png` and `sam_labels.png`, and the evaluation crops in
`data/processed/<name>/sam_gt/` (see the README there).

| Dataset | φ SAM | φ Otsu | Reference (φ) | SAM IoU / Dice | Otsu IoU / Dice |
|---------|-------|--------|---------------|----------------|-----------------|
| synthetic | 0.290 | 0.254 | exact ground-truth mask (0.250) | 0.864 / 0.927 | 0.913 / 0.955 |
| MicroLib 000210 | 0.217 | 0.232 | MicroLib-annotated threshold (0.219) | 0.845 / 0.916 | 0.941 / 0.970 |

SAM quality is measured against the exact ground truth on the synthetic image and, on MicroLib, against a
curated reference: the midpoint of the two phase gray levels annotated by the MicroLib authors (8 and 133)
applied to the raw micrograph (`python -m src.features.sam_segment --data ... --reference-only`). That
reference is itself a threshold, so it structurally favours Otsu; the synthetic ground truth is the
unbiased comparison. The 5 evaluation crops in `sam_gt/` are still written (exact ground truth for
synthetic); hand-corrected MicroLib crops were not produced.

On the synthetic image SAM's masks follow the blurred edges outwards, overestimating `φ`; Otsu,
whose threshold sits halfway between the two gray levels, is more accurate there. On MicroLib SAM
misses a few islands and shows some straight cuts at tile borders.

The notebook [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb) (`make eda`) looks at both segmentations in
more detail: where Otsu and SAM disagree (two thirds on interface rims), SAM's one-pixel dilation on the
synthetic image, the per-crop φ spread that motivates the many-seed evaluation, the 2D S₂ / L of each map
and the mild x-banding of the MicroLib micrograph.

## Evaluation

```bash
make eval                                          # synthetic
make eval DATA=configs/data/microlib_000210.yaml   # real
make test                                          # pytest
```

`make eval` runs `src.models.generate` for every model and then `src.visualization.visualize`:

- **generate** loads `G_best.pt` (`generate.checkpoint`), generates one 64³ volume per seed in
  `generate.seeds` (N = 128) and saves them as `models/<name>/<run>/volumes/*.tif` (0/255). It scores
  them against two 2D references: `train`, the image the model was trained on (Otsu map, SAM map for
  M3), and `common`, shared by all models of a dataset so that M2 vs M3 is a fair comparison (the exact
  ground-truth mask for synthetic, the Otsu map for MicroLib, which has no ground truth). Results go to
  `metrics.yaml` and `curves.npz` (incl. per-volume curves) in the run folder.
- **visualize** upserts `reports/metrics.csv` (one row per dataset × model × reference: `dphi`, `s2_mae`,
  `s2_err`, `L_mae`, `L_err`, per-plane values, 95 % bootstrap CIs `*_lo/*_hi`, and the late-training
  summary `late_*`), writes `reports/comparison_vs_m1_<name>.csv` (bootstrap difference to M1 with a
  better / worse / no-clear-difference verdict) and the figures in `reports/figures/`:
  `<name>_descriptors.png` (S₂ and L curves), `<name>_qualitative.png` (training input | SAM overlay |
  xy, xz, yz slices | 3D isosurface), `<name>_training.png` (critic Wasserstein estimate) and
  `pipeline.png`; then exports the viewer data.

Metric definitions are in `reports/report.md` §4.

## Exploring the volumes

**Static viewer.** `make viewer` exports 4 volumes per run of both datasets (the seven models, the second
training runs and the 128 px variant) to `reports/viewer/viewer_data.json`; open
`reports/viewer/index.html` in a browser (it also works from disk). It shows a rotatable 3D view, the
three orthogonal sections, a crop of the model's own training map (Otsu, or SAM for M3) and the test
metrics. A specific view can be linked with `index.html?dataset=synthetic&model=m3_swin_sam`.

**Streamlit app.** `make app` (or `poetry run streamlit run app/streamlit_app.py`) opens an explorer with:

- new volumes from any run's evaluated checkpoint, for any latent seed and an edge of 64, 96 or 128
  voxels, optionally periodic; or the stored evaluation volumes;
- a 3D surface view, section sliders and the descriptors of the current volume (φ, S₂ and L MAE, per plane);
- S₂ and L curves of several models against the dataset's reference;
- the test-metric table of the dataset, including the repeat runs;
- an RVE download (zip with VTI, MetaImage, Abaqus, NumPy and/or TIFF files plus JSON metadata).

Generating needs the local checkpoints (`models/`, not versioned); without them the app shows the
volumes from the viewer export. The app's dependencies are in the Poetry group `app`. `.streamlit/config.toml`
turns off Streamlit's file watcher, which stalls when it inspects `torch`.

## Exporting RVEs for FEM / FFT homogenization

```bash
make export-rve DATA=configs/data/microlib_000210.yaml                       # M4, 3 periodic 128^3 RVEs
make export-rve MODEL=m1_cnn SIZE=64 SEEDS="0" FORMATS="inp" DATA=...         # Abaqus voxel mesh
python -m src.models.export_rve --config configs/m4_ensemble.yaml --data configs/data/microlib_000210.yaml     --size 128 --periodic --seeds 0 1 2 --formats vti mhd inp npy tif
```

Files go to `exports/<dataset>/<model>/` (not versioned), one set per seed:

| Format | Content | Typical use |
|--------|---------|-------------|
| `.vti` | VTK XML ImageData, int32 cell array `material` (0 matrix, 1 inclusion), spacing in µm | ParaView, DAMASK, FFT solvers |
| `.mhd` + `.raw` | MetaImage header + uint8 voxels | ITK, ParaView, FFT homogenization codes |
| `.inp` | Abaqus C3D8 voxel mesh, element sets `MATRIX` / `INCLUSION`, node sets `XMIN` … `ZMAX` for periodic BCs, solid sections with placeholder materials | Abaqus; convertible for CalculiX / Code_Aster |
| `.npy`, `.tif` | uint8 label volume | Python, ImageJ |
| `.json` | model, checkpoint, seed, shape, voxel size, physical size, phase fractions, seam ratios, files | traceability |

- **Size.** Larger volumes than the 64³ training size come from enlarging the latent input (edge =
  32 · latent − 64: 64, 96, 128, … voxels), as in the SliceGAN paper. MicroLib voxels are 0.687 µm.
- **Periodicity.** `--periodic` uses SliceGAN's latent tiling (first two latent slices copied from the last
  two along each axis). The generated field then repeats with period N − 2 voxels, so 2 voxels are cropped
  per axis (upstream SliceGAN crops 1, which leaves a duplicated slice at the seam). `seam_ratio_*` in the
  JSON compares the wrap-around face mismatch with that of neighbouring interior slices: ≈ 1 for the
  periodic RVEs (1.0–2.0 measured) vs ≈ 8–13 without `--periodic`.
- **Representativeness.** The phase fraction of single RVEs varies (0.18–0.24 at 126³–128³ for M4); check
  convergence of the homogenized property with RVE size and average over several seeds.
- Homogenization itself (FEM/FFT solves) is outside the scope of this project.

## Experiment tracking (MLflow)

All runs, including the archived stabilization runs (v1–v5) and the critic probes, are tracked with
MLflow in a local file store (`mlruns/`, not versioned); nothing is uploaded anywhere.

```bash
make mlflow-backfill   # import finished runs from models/ (idempotent; each root tagged with its version)
make mlflow-ui         # http://127.0.0.1:5000
```

One experiment per dataset (`slicegan-vit-microlib_000210`, `slicegan-vit-synthetic`). Each run has the
flattened config as parameters; training curves (`train_*`, per generator step), held-out selection
scores (`val_phi`, `val_s2_mae`, per epoch), the best checkpoint (`best_*`) and evaluation metrics
(`eval_<reference>_<metric>`) as metrics; tags `version`, `model`, `discriminator`, `swin_head`, `loss`,
`training_image`, `git_commit`; and config, logs, metrics, curves and previews as artifacts. New runs
can be logged live with `tracking.mlflow: true` (the evaluation step then adds its metrics to the same
run).

## Results

Full report: [reports/report.pdf](reports/report.pdf) (English) and
[reports/report_es.pdf](reports/report_es.pdf) (Spanish). The numbers are in `reports/metrics.csv` and
`reports/comparison_vs_m1_<dataset>.csv`.

Test S₂ MAE on 128 volumes per model, against the common reference (MicroLib: Otsu map; synthetic: exact
ground-truth mask). "Worse" means the 95 % bootstrap interval of the difference to M1 excludes zero.

| Model | MicroLib 000210 | Synthetic |
|-------|-----------------|-----------|
| M1 CNN (SliceGAN) | 0.0029 | 0.0040 |
| M1 + DiffAug | 0.0009 | 0.0045 |
| M2 Swin + DiffAug | 0.0077 | 0.0093 (worse) |
| M3 Swin + DiffAug, SAM map | 0.0328 (worse) | 0.0298 (worse) |
| M4 CNN + frozen Swin | 0.0014 (φ exact) | 0.0039 (most stable training) |
| M5 M1 + Swin fine-tune | 0.0097 | 0.0013 |
| M1 extended | 0.0059 | 0.0020 |

- **RQ1:** a Swin-T critic is not better than the CNN critic. It needs DiffAug to train at all. On MicroLib its
  two runs bracket the CNN's; its single synthetic run is significantly worse and less isotropic.
- **RQ2:** zero-shot SAM segments this kind of high-contrast image worse than Otsu (it dilates every particle by
  about 1 px), and M3 inherits that bias.
- **RQ3:** the CNN + frozen Swin-T ensemble (M4) matches SliceGAN, with no measurable gain. Swin
  fine-tuning (M5) is not distinguishable from training the CNN for the same extra steps, and a Swin branch
  at 128 px (`configs/m4_ensemble_swin128.yaml`) did not help either.
- **Repeat runs** (second training seed of M1, M2 and M4 on MicroLib, `configs/*_seed2.yaml`,
  `reports/metrics_seeds_microlib.csv`): the same model trained twice differs as much as the models differ
  from each other (M2: S₂ MAE 0.0077 vs 0.0017), so single-run rankings are not reliable.
- No model is significantly better than M1. Random volumes with the correct φ score 0.042, so all
  models except M3 sit an order of magnitude below that floor.

## References

- S. Kench, S. J. Cooper. *Generating three-dimensional structures from a two-dimensional slice with
  generative adversarial network-based dimensionality expansion.* Nature Machine Intelligence, 2021.
  Code: https://github.com/stke9/SliceGAN
- Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- A. Kirillov et al. *Segment Anything.* ICCV 2023.
- A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- S. Kench et al. *MicroLib: A library of 3D microstructures generated from 2D micrographs using
  SliceGAN.* Scientific Data, 2022 (if MicroLib is used).

## License

MIT (see `LICENSE`). SliceGAN keeps its own license.
