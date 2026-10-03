# SliceGAN with Vision Transformer discriminators and a SAM front-end

Vision Transformers — FIUBA. Individual work.

> Draft. Values marked **TBD** are filled in from the final runs (`reports/metrics.csv`).

## 1. Project goal

Three-dimensional microstructures are needed to compute effective material properties (for example as
representative volume elements for finite-element analysis), but 3D imaging (micro-CT, FIB-SEM) is
expensive and often unavailable, while 2D micrographs are cheap. SliceGAN (Kench & Cooper, 2021)
generates 3D volumes that are statistically equivalent to a single 2D micrograph of an isotropic
material by training a 3D generator against a 2D discriminator that sees slices of the generated
volume.

This project generates 64³ two-phase volumes from one 2D micrograph with SliceGAN and studies where
Vision Transformers help:

- **RQ1 — ViT critic.** Does replacing SliceGAN's CNN discriminator by a Swin Transformer (Swin-T)
  improve the generated microstructures? (M1 vs M2)
- **RQ2 — SAM front-end.** Does segmenting the micrograph with the Segment Anything Model (SAM),
  instead of a global threshold, improve them? (M2 vs M3)
- **RQ3 — ViT as an additional critic (extension).** Does adding a pretrained Swin-T critic next to
  SliceGAN's CNN, as in Vision-aided GAN (Kumari et al., 2022), help, either from scratch (M4) or as a
  fine-tuning stage of a trained SliceGAN (M5, compared against M1 trained for the same extra steps)?

Quality is measured with the statistical descriptors used in the microstructure-generation literature
(phase fraction, two-point correlation, lineal path), since no 3D ground truth exists.

## 2. Overall architecture

![Pipeline](figures/pipeline.png)

**Data.** The main case is MicroLib entry 000210 (DoITPoMS micrograph library): an optical micrograph
of dark islands in a light matrix, 437 × 800 px after removing the scale bar, 0.687 µm/px. It was
chosen at random among the MicroLib two-phase entries with a phase gray-level gap ≥ 80 and accepted
after an isotropy check (x/y correlation-length ratio 1.11). A synthetic micrograph (non-overlapping
discs, φ = 0.25, rendered with blur and noise, exact ground-truth mask) is used as a sanity check.

**Segmentation (front-end).** M1, M2, M4 and M5 train on an Otsu label map (φ = 0.232). M3 trains on a
SAM label map: zero-shot SAM ViT-B on 256 px tiles, overlapping masks merged (IoU > 0.3), and groups
whose mean gray level departs from the matrix labelled as inclusions (φ = 0.217).

**Generator.** SliceGAN's 3D resize-convolution generator, unchanged for every model (40.1 M
parameters): a 32 × 4 × 4 × 4 Gaussian latent → 64³ volume, softmax over the two phases.

**Slicer and critics.** Each generated volume is cut into all 64 slices along x, y and z. The slices
and real 64 × 64 crops of the training image are scored by a 2D critic:

| Model | Training image | 2D critic | Loss | Trainable critic params |
|-------|----------------|-----------|------|-------------------------|
| M1 | Otsu | SliceGAN CNN (5 strided convs) | WGAN-GP | 2.8 M |
| M2 | Otsu | Swin-T (ImageNet), stages 3–4 + head trainable, DiffAug, lr 2e-5 | WGAN-GP | 26.3 M |
| M3 | SAM | same critic as M2 | same as M2 | same as M2 |
| M4 | Otsu | SliceGAN CNN **+** frozen Swin-T with per-scale heads (ensemble) | CNN: WGAN-GP, Swin: hinge | 2.8 M + 0.18 M |
| M5 | Otsu | as M4, generator initialised from M1's best checkpoint | as M4 | as M4 |
| M1-ext | Otsu | as M1, generator initialised from M1's best checkpoint | WGAN-GP | 2.8 M |

**Metrics.** For each model, four 64³ volumes (seeds 0–3) are compared with the 2D image through φ,
S₂(r) and L(r), overall and per slice orientation (Section 4).

## 3. Technical implementation

**Stack.** Python 3.11, PyTorch 2.5.1 (CUDA 12.4), timm 1.0.11, Hugging Face `transformers` 4.46.3,
NumPy/SciPy/scikit-image, Poetry environment (`setup.sh`), YAML configs with inheritance and dataset
overlays, logging to file and console, 68 pytest tests. Trained on one NVIDIA RTX A2000 (12 GB).

**Pretrained models.**

- Swin-T: `timm/swin_tiny_patch4_window7_224.ms_in1k` on the Hugging Face Hub (the Microsoft ImageNet-1k
  checkpoint, same weights as `microsoft/swin-tiny-patch4-window7-224`). timm is used instead of
  `transformers` because Swin-T runs natively on 64 px slices only if the attention windows of stages
  3–4 shrink to 4 × 4 and 2 × 2 with a resized relative-position bias, which timm supports and
  `transformers` 4.46 does not. The RGB patch embedding is converted to two phase channels so that a
  one-hot slice is seen as a centred gray image (matrix −0.5, inclusion +0.5).
- SAM: `facebook/sam-vit-base` through the `transformers` mask-generation pipeline, zero-shot.

**SliceGAN integration.** The upstream repository is a git submodule (`external/SliceGAN`, MIT) and is
not modified. `src/models/slicegan_wrapper.py` builds the generator and CNN critic with upstream's
network factory and the exact layer lists of `run_slicegan.py`, and reuses its gradient penalty.
`src/models/train.py` is a fork of upstream's training loop with the same WGAN-GP schedule (Adam
1e-4, β = (0.9, 0.99), λ_GP = 10, 5 critic steps per generator step) and SliceGAN's Algorithm 1 batch
rule: the critic sees all 64 slices per axis of m_D = 1 volume, the generator step uses m_G = 2 m_D
volumes. Additions: critic branches (single critic or CNN + Swin ensemble), hinge loss, differentiable
augmentation of critic inputs (DiffAug: translation, cutout, 90° rotations/flips), generator warm
start, and checkpoint selection.

**Checkpoint selection.** After every epoch (100 generator steps) the generator produces volumes from
two held-out seeds (1000, 1001), never used for evaluation; the epoch with the lowest S₂ MAE against
the model's own training image is kept and evaluated. This is applied identically to every model.
Selection and evaluation use the same 2D reference image, so absolute errors are slightly optimistic,
equally for all models.

**Main modules.** `src/data/make_dataset.py` (download, crop, Otsu, crops), `src/features/sam_segment.py`
(SAM front-end), `src/features/descriptors.py` (φ, S₂, L), `src/models/discriminator_swin.py` (Swin
critics), `src/models/train.py`, `src/models/generate.py` (volumes + metrics),
`src/visualization/visualize.py` (figures, `metrics.csv`) and an interactive volume viewer
(`reports/viewer/`).

## 4. Evaluation

There is no 3D ground truth (no micro-CT), so generated volumes are compared statistically with the
2D training image. For an isotropic material, any planar section of the 3D microstructure has the
same statistics as the 2D micrograph, so the 2D descriptors of the training image must be matched by
the slice-averaged descriptors of the generated volume; this is the justification for the metrics
below (SliceGAN, Kench & Cooper 2021). All are implemented in `src/features/descriptors.py` and
tested in `tests/test_descriptors.py` against cases with analytic answers. Label 1 is the
inclusion/pore phase and `I(x)` its indicator function; curves are evaluated for `r = 0 … 32` px on
N = 4 generated 64³ volumes per model (different seeds).

- **Phase fraction** `φ = ⟨I(x)⟩`, reported as mean ± std over the N volumes, together with
  `|Δφ| = |φ̄_gen − φ_train|`.
- **Two-point correlation** `S₂(r) = P[I(x) = 1, I(x + r) = 1]`, computed by FFT autocorrelation
  (each lag normalized by its number of valid pixel pairs, no periodicity assumed) and radially
  averaged over lag vectors with `round(|r|) = r`. `S₂(0) = φ` and `S₂(r) → φ²` for uncorrelated
  points. For a volume, `S₂` is averaged over all xy, xz and yz slices (192 slices for 64³).
- **Lineal path** `L(r)` (as in Micro3Diff, Lee & Yun 2024; Lyu & Ren 2024): probability that a
  straight segment of `r + 1` pixels lies entirely in the inclusion phase, averaged over both in-plane
  directions (2D) or the three axes (3D). `L(0) = φ`; unlike `S₂`, `L` is sensitive to connectivity
  along lines.
- **Errors**: MAE over `r` of the generated vs training curves (`S₂ MAE`, `L MAE`, using the mean
  curve over the N volumes), and the Micro3Diff error rate
  `err(S₂) = mean|S₂_gen − S₂_train| / mean|S₂_train|` (likewise `err(L)`).
- **3D isotropy check**: `φ`, `S₂ MAE` and `L MAE` per slice orientation (xy, xz, yz). If one plane
  diverges, the volume is not isotropic. Averaged over all slices, `φ_xy = φ_xz = φ_yz = φ` by
  construction, so the spread of the per-slice `φ` along each axis is reported too.
- **Common reference.** Every model is scored against its own training image and against a reference
  shared by all models of a dataset, so that M3 (trained on the SAM map) is comparable with M2: the
  Otsu map for MicroLib (no ground truth exists) and the exact ground-truth mask for synthetic data.
- **SAM segmentation quality** (M3 front-end): IoU `= |P ∩ G| / |P ∪ G|` and Dice
  `= 2|P ∩ G| / (|P| + |G|)` of the label map `P` against the exact ground truth `G` on the synthetic
  image, and, on MicroLib, against a curated reference: the midpoint of the two phase gray levels
  annotated by the MicroLib authors (8 and 133) applied to the raw micrograph.

Reference scales: on the synthetic dataset, the Otsu segmentation vs the ground-truth mask gives
`S₂ MAE = 0.0021`. Uncorrelated random 64³ volumes with the correct `φ` of MicroLib 000210 give
`S₂ MAE = 0.042` (`err = 0.42`) and `L MAE = 0.077` (`err = 0.89`), a floor any useful model must beat.

FID and 3D SSIM are not used (they need volumetric ground truth), and no classification accuracy is
reported.

## 5. Results and examples

### 5.1 Main results (MicroLib 000210, common reference = Otsu map, best checkpoint, N = 4)

| Model | φ (mean ± std) | \|Δφ\| | S₂ MAE | err(S₂) | L MAE | S₂ MAE xy / xz / yz |
|-------|----------------|--------|--------|---------|-------|---------------------|
| M1 CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| M2 Swin | TBD | TBD | TBD | TBD | TBD | TBD |
| M3 Swin + SAM | TBD | TBD | TBD | TBD | TBD | TBD |
| M4 CNN + Swin | TBD | TBD | TBD | TBD | TBD | TBD |
| M5 M1 + Swin fine-tune | TBD | TBD | TBD | TBD | TBD | TBD |
| M1 extended | TBD | TBD | TBD | TBD | TBD | TBD |

Figures: `figures/microlib_000210_qualitative.png` (training input | SAM overlay | xy, xz, yz slices |
3D isosurface), `figures/microlib_000210_descriptors.png` (S₂ and L curves),
`figures/microlib_000210_training.png`, and the interactive viewer.

### 5.2 Training a Swin critic: stabilization study

Transformer critics are known to make GAN training unstable (ViTGAN, Lee et al., 2022). With SliceGAN's
WGAN-GP setup the Swin-T critic was unstable in every configuration tried; the table summarizes the
first 12 epochs on MicroLib (held-out φ target 0.232; S₂ MAE of M1's best epochs: 0.002–0.004).

| Run | Swin critic | Loss | DiffAug | Epochs 1–12 | Outcome |
|-----|-------------|------|---------|-------------|---------|
| v1 | stages 3–4 trainable, lr 1e-4, last checkpoint | WGAN-GP | no | realistic at epoch 10, empty at 15 | oscillates, ended collapsed (φ = 0.003) |
| v2 | stages 3–4, lr 2e-5 | WGAN-GP | no | empty at 4–5, S₂ MAE 0.010 at 9–11 | oscillates, M1-level epochs exist |
| v2 rerun | same as v2, new seed | WGAN-GP | no | good at epoch 2, then φ ≈ 0 for 10 epochs | strongly seed-dependent |
| v3 | v2 + ViTGAN: improved spectral norm, Adam β₁ = 0, G EMA | WGAN-GP | no | φ 0.002–0.08 throughout | critic too strong |
| stage-4 probe | only stage 4 trainable | WGAN-GP | no | φ 0.08–0.29, best S₂ MAE 0.037 | no full collapse, blurry |
| A | frozen, pooled multi-scale head | WGAN-GP | yes | φ = 1.0 at epochs 1–6, then 0.51 → 0.31; best S₂ MAE 0.049 | fails: gradient penalty cannot be met through a frozen backbone |
| B | stages 3–4, lr 2e-5 | WGAN-GP | yes | φ ≈ 0 at epochs 1–5; from epoch 7 φ 0.20–0.30, S₂ MAE 0.0091 / 0.018 / **0.0035** / 0.024 / 0.049 / **0.0035** | passes: M1-level epochs, more consistent than v2 |
| C | frozen, per-position multi-scale heads | WGAN-GP | yes | φ = 1.0 at epochs 1–2, 0.89–0.95 to epoch 8, then 0.38 → 0.08 → 0.11; best S₂ MAE 0.060 | fails like A: the GP, not the pooling, is the problem |
| A-hinge | frozen, pooled head, head spectral norm | hinge | yes | φ 0.57–1.0, drifting up; best S₂ MAE 0.26 | fails: no collapse, but φ uncontrolled |
| C-hinge | frozen, per-position heads, head spectral norm | hinge | yes | φ 0.31–0.56; best S₂ MAE 0.12 | fails alone; better than A-hinge, used inside M4/M5 |

![Swin critic probes](figures/stabilization_probes.png)

Per-epoch logs of every probe are in `reports/probes/`.

**Decision.** M2/M3 use probe B (the only Swin-only critic reaching M1-level epochs); M4/M5 use the per-position frozen heads of C-hinge inside an ensemble with the CNN critic.

Three findings shaped the final models. First, a frozen pretrained backbone cannot be trained with WGAN-GP:
the gradient penalty asks for unit input-gradient norm, which the frozen Swin layers fix and the small
head can only rescale, so the penalty dominates (values of 10–355 instead of ≈ 1) and the critic gives
no useful signal. Frozen-backbone critics in the literature (Projected GAN, Vision-aided GAN) use
hinge or BCE losses instead. Second, Vision-aided GAN reports that pretrained critics used *alone*
diverge and help only in an ensemble with the original discriminator, which motivates M4 and M5. Third, with a hinge loss the frozen Swin critic no longer collapses but does not control the phase fraction on its own (φ drifts to 0.3–0.8); a plausible cause is Swin's per-token LayerNorm, which removes much of the absolute phase-intensity information. The CNN critic in M4/M5 supplies that constraint. Finally, DiffAug is what made the trainable Swin critic work (B vs v2), in line with the limited-data GAN literature (Zhao et al., 2020; Karras et al., 2020): the critic otherwise overfits the few hundred distinct views of a single micrograph.

**Linear-probe diagnostic** (Vision-aided GAN, Sec. 3.2): a logistic regression on frozen, pooled Swin
features separates real 64 × 64 crops from slices of M1's best generator with 73 % held-out accuracy at
64 px input (stage 2 alone: 75 %), 83 % at 128 px and 83 % at 224 px; random volumes with the correct φ
are separated at 100 %. The frozen features therefore carry a usable real-vs-fake signal, strongest at
the 8 × 8 resolution of stage 2, and a larger input could add about 10 points.

### 5.3 SAM front-end

| Dataset | φ SAM | φ Otsu | Reference (φ) | SAM IoU / Dice | Otsu IoU / Dice |
|---------|-------|--------|---------------|----------------|-----------------|
| synthetic | 0.290 | 0.254 | exact ground truth (0.250) | 0.864 / 0.927 | 0.913 / 0.955 |
| MicroLib 000210 | 0.217 | 0.232 | MicroLib-annotated threshold (0.219) | 0.845 / 0.916 | 0.941 / 0.970 |

Zero-shot SAM needs tiling on this image (a single pass with 16 points per side finds φ = 0.07), and
even tiled it is less accurate than Otsu on both references: on the synthetic image its masks follow the
blurred edges outwards (φ overestimated by 16 %), on MicroLib it misses a few islands. The MicroLib
reference is itself a threshold and structurally favours Otsu; the synthetic ground truth is the
unbiased comparison and leads to the same conclusion.

## 6. Conclusions and future work

**TBD** after the final runs. Points already supported by the experiments:

- SliceGAN with its CNN critic is stable and matches the 2D statistics of the micrograph closely
  (best held-out S₂ MAE 0.002–0.004, an order of magnitude below the random-volume floor of 0.042).
- A Swin-T critic trained in SliceGAN's WGAN-GP setup is unstable and seed-dependent; standard and
  ViT-specific stabilizers (lower learning rate, improved spectral norm, Adam β₁ = 0, generator EMA,
  smaller trainable part) did not remove the oscillation. With checkpoint selection, M1-level epochs can
  still be recovered.
- For this two-phase, high-contrast micrograph, a global threshold is a better segmentation than
  zero-shot SAM; SAM's value lies in harder images (low contrast, texture, multiple phases).

SliceGAN is the 2021 literature baseline; 2024 works (Micro3Diff, DDPM-GAN) improve descriptors and
stability with diffusion models, but are outside the scope of a Vision Transformer course project. The
contribution here is the evaluation of Swin critics and SAM as a phase front-end.

**Future work.** Anisotropic materials (three-view SliceGAN), 128³ volumes, periodic volumes as
representative volume elements and homogenization (FEM/FNO), Swin critics on 128 px inputs (suggested by
the linear probe), multi-seed statistics for the Swin-based models, and diffusion-based generators.

## 7. Planning

| Task | Owner | Status |
|------|-------|--------|
| Repo skeleton, configs, Makefile, Poetry setup | Student | Done |
| Datasets: synthetic + MicroLib 000210 | Student | Done |
| Descriptors φ / S₂ / L + tests | Student | Done |
| M1 SliceGAN baseline | Student | Done (final run in progress) |
| M2 Swin-T critic + stabilization study | Student | In progress |
| M3 SAM front-end | Student | Done (final run pending) |
| M4 / M5 Vision-aided extensions | Student | In progress |
| Generation, metrics, figures, viewer | Student | Done (final numbers pending) |
| Report | Student | Draft |
| Presentation (12 Oct) | Student | Pending |

## References

- S. Kench, S. J. Cooper. Generating three-dimensional structures from a two-dimensional slice with
  generative adversarial network-based dimensionality expansion. *Nature Machine Intelligence*, 2021.
- S. Kench et al. MicroLib: A library of 3D microstructures generated from 2D micrographs using SliceGAN.
  *Scientific Data*, 2022.
- Z. Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*, 2021.
- A. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
  *ICLR*, 2021.
- A. Kirillov et al. Segment Anything. *ICCV*, 2023.
- I. Gulrajani et al. Improved Training of Wasserstein GANs. *NeurIPS*, 2017.
- K. Lee et al. ViTGAN: Training GANs with Vision Transformers. *ICLR*, 2022.
- N. Kumari, R. Zhang, E. Shechtman, J.-Y. Zhu. Ensembling Off-the-shelf Models for GAN Training.
  *CVPR*, 2022.
- A. Sauer et al. Projected GANs Converge Faster. *NeurIPS*, 2021.
- S. Zhao et al. Differentiable Augmentation for Data-Efficient GAN Training. *NeurIPS*, 2020.
- T. Karras et al. Training Generative Adversarial Networks with Limited Data. *NeurIPS*, 2020.
- K.-H. Lee, G. J. Yun. Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D
  reconstruction of microstructures with harmonized sampling (Micro3Diff). *npj Computational
  Materials*, 2024. doi:10.1038/s41524-024-01280-z
- J. Phan, M. Sarmad, L. Ruspini, G. Kiss, F. Lindseth. Generating 3D images of material microstructures
  from a single 2D image: a denoising diffusion approach. *Scientific Reports*, 2024.
- X. Lyu, X. Ren. Microstructure reconstruction of 2D/3D random materials via diffusion-based deep
  generative models. *Scientific Reports*, 2024.
