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

- `synthetic` (current default): RSA circular inclusions, target `φ` 0.25, 512², fixed seed.
- `slicegan`: a two-phase micrograph from `external/SliceGAN/Examples/`. TODO: pick the file.
- `microlib`: one entry from [MicroLib](https://microlib.io). TODO: cite the chosen ID.

## Training

```bash
poetry run python -m src.models.train --config configs/m1_cnn.yaml          # make train-m1
poetry run python -m src.features.sam_segment --config configs/m3_swin_sam.yaml   # make sam (needed by M3)
poetry run python -m src.models.train --config configs/m2_swin.yaml         # make train-m2
poetry run python -m src.models.train --config configs/m3_swin_sam.yaml     # make train-m3
```

Checkpoints go to `models/<run_name>/`, logs to `logs/<run_name>.log`.

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
