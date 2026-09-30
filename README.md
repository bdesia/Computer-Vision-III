# SliceGAN with a Vision Transformer discriminator and a SAM front-end

Final project — **Vision Transformers (FIUBA)**. Instructors: Abraham Rodriguez, Oksana Bokhonok.
Individual work.

## Goal

Generate 3D volumes (64³) that are statistically equivalent to a 2D micrograph of a two-phase,
isotropic material using **SliceGAN**, and measure:

1. whether a **Swin-T discriminator** (hierarchical Vision Transformer) improves on the original CNN
   discriminator, and
2. whether using **SAM** as a phase-segmentation front-end improves the descriptors of the generated
   volume (phase fraction `φ` and two-point correlation `S₂`).

| ID | Model | 2D input | 3D generator | 2D discriminator |
|----|-------|----------|--------------|------------------|
| M1 | SliceGAN baseline | training image | SliceGAN 3D CNN | SliceGAN CNN |
| M2 | SliceGAN–Swin | same as M1 | same G | Swin-T (Hugging Face) |
| M3 | SliceGAN–Swin+SAM | SAM phase map | same G | same Swin-T as M2 |

- M1 vs M2: does the ViT discriminator help?
- M2 vs M3: does SAM preprocessing help?

## Layout

```
configs/            default.yaml + one yaml per model (m1_cnn, m2_swin, m3_swin_sam)
data/               raw/ interim/ processed/ (not versioned, except processed/sam_gt/)
external/SliceGAN/  upstream SliceGAN (git submodule, unmodified)
src/data/           make_dataset.py — downloads or generates the 2D image and 64x64 crops
src/features/       sam_segment.py (zero-shot SAM), descriptors.py (φ, S₂)
src/models/         SliceGAN wrapper, CNN/Swin discriminators, train.py, generate.py
src/visualization/  figures and metrics table
tests/              pytest
reports/            report.md (PDF source), figures/, metrics.csv
models/             checkpoints (not versioned)
logs/               per-run logs
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

```bash
poetry run python -m src.data.make_dataset --config configs/default.yaml   # make data
```

Source is set in `configs/default.yaml → data.source`:

- `synthetic` (current default): non-overlapping discs placed by random sequential adsorption (RSA),
  target `φ` 0.25, 512², fixed seed. The discs are rendered as a blurred, noisy grayscale micrograph
  so that segmentation (Otsu for M1/M2, SAM for M3) is non-trivial; the clean mask is kept as ground truth.
- `slicegan`: a micrograph from `external/SliceGAN/Examples/` (`data.slicegan_example`). Upstream only
  ships `NMC.tif` (3 phases, out of scope), so this source is only useful with a custom file.
- `microlib`: one entry from [MicroLib](https://microlib.io), placed manually at `data.raw_path`.
  TODO: cite the chosen ID.

Outputs:

| Path | Content |
|------|---------|
| `data/raw/micro_2d.png` | grayscale micrograph (input to Otsu and SAM) |
| `data/raw/micro_2d_gt.png`, `micro_2d.json` | synthetic only: clean mask and `phi_true` |
| `data/interim/micro_2d_gray.png`, `micro_2d_otsu.png` | normalized grayscale and Otsu label map |
| `data/processed/train_2d/image.png` | M1/M2 training label map (0 = matrix, 1 = inclusion) |
| `data/processed/train_2d/crops.npy` | 64×64 crops, stride 32, `(N, 64, 64)` uint8 |
| `data/processed/train_2d/meta.yaml` | `φ`, threshold, crop stats, Otsu IoU vs GT (synthetic) |

With the default config: `phi_true = 0.2505`, `phi_train = 0.2540`, Otsu IoU vs GT = 0.913, 225 crops.

## Training

```bash
poetry run python -m src.models.train --config configs/m1_cnn.yaml          # make train-m1
poetry run python -m src.features.sam_segment --config configs/m3_swin_sam.yaml   # make sam (needed by M3)
poetry run python -m src.models.train --config configs/m2_swin.yaml         # make train-m2
poetry run python -m src.models.train --config configs/m3_swin_sam.yaml     # make train-m3
```

Quick smoke test (runs on CPU too):

```bash
poetry run python -m src.models.train --config configs/m1_cnn.yaml --epochs 2 --iters-per-epoch 1 --device cpu
```

`--epochs`, `--iters-per-epoch` and `--device` override the yaml. `train.max_minutes` sets an optional
wall-clock budget (training stops cleanly and checkpoints).

Each run writes to `models/<run_name>/`: `G_last.pt`, `D_last.pt`, the resolved `config.yaml`,
`history.csv` (critic real/fake scores, Wasserstein estimate, gradient penalty, G loss, s/step),
`previews/epochNNN.png` (central xy/xz/yz slices) and SliceGAN's `slicegan_params.data`.
Logs go to `logs/<run_name>.log`.

### How SliceGAN is integrated

`external/SliceGAN` is the unmodified upstream repo (git submodule, MIT). `src/models/slicegan_wrapper.py`
builds the generator and the CNN critic with upstream's `slicegan_rc_nets` and the exact layer lists of
`run_slicegan.py` (`z_channels = 32`, 4³ latent → 64³ volume), and reuses upstream's gradient penalty.
`src/models/train.py` is a fork of upstream `model.train` with the same WGAN-GP schedule (Adam
1e-4, β = (0.9, 0.99), λ = 10, 5 critic steps per G step, one isotropic critic shared by the three
axes). The differences are: config/logging/checkpointing, a pluggable critic (CNN or Swin-T), random
64×64 crops sampled on the fly on the GPU instead of 28 800 pre-built crops in RAM, and an optional
cap on the number of fake slices per axis (`train.fake_slices`).

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

**Same training protocol for all models.** Upstream feeds all 512 slices per axis (64 slices ×
8 volumes) to the critic at every step, which is prohibitive for Swin. Every model (M1 included)
therefore uses `train.fake_slices: 64` random slices per axis and TF32 matmuls, so M1 vs M2 differs
only in the critic. Measured on an RTX A2000 12 GB: 0.55 s per G step for M1 and 2.4 s for M2,
i.e. ~46 min vs ~3.3 h for the default 50 × 100 G steps.

## Evaluation

```bash
make eval      # generates N=4 64³ cubes per model, writes reports/metrics.csv + figures
make test      # pytest
```

Metrics: `φ` mean ± std and `|Δφ|` vs the 2D training image; `S₂(r)` MAE up to `r = 32`
(averaged over xy/xz/yz slices); SAM IoU/Dice on 5 crops with manual ground truth.

## Results

TBD — see `reports/report.md`.

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
