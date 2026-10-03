# SliceGAN with Vision Transformer discriminators and a SAM front-end

Vision Transformers — FIUBA. Individual work.

> Draft. Values marked **TBD** are filled in from the final runs (`reports/metrics.csv`).

## 1. Project goal

Three-dimensional microstructures are needed to compute effective material properties (for example as
representative volume elements for finite-element analysis), but 3D imaging (micro-CT, FIB-SEM) is
expensive and often unavailable, while 2D micrographs are cheap, fast and usually of higher resolution.
This project generates 3D two-phase microstructures from a single 2D micrograph with SliceGAN and studies
whether Vision Transformers improve it, either as the adversarial critic or as a segmentation front-end.

### 1.1 Background: SliceGAN

**The problem.** A 2D micrograph of a material contains, statistically, much of the information of its 3D
structure: for an *isotropic* material (no preferred direction), every planar cut through the volume has
the same statistics (phase fractions, feature sizes and shapes, spatial correlations) as any other. The
task is therefore to produce 3D volumes whose 2D sections are indistinguishable, statistically, from the
micrograph. Classical reconstruction methods optimize a 3D volume to match chosen statistical
descriptors (e.g. the two-point correlation); they are slow (hours for 10⁶ voxels) and only reproduce the
descriptors they were told to match.

**Generative adversarial networks.** A GAN trains two networks against each other: a *generator* G that
maps random noise to samples, and a *discriminator* (critic) D that tries to tell generated samples from
real ones. G is updated to fool D; at equilibrium the generated distribution matches the real one. GANs
learn the statistics directly from the data instead of from hand-picked descriptors, and once trained
they generate new samples in seconds.

**SliceGAN's key idea: dimensionality expansion through slicing** (Kench & Cooper, 2021). A GAN normally
needs training data of the same dimensionality as its output, but no 3D training data exists here.
SliceGAN combines a **3D generator** with a **2D discriminator** and bridges them with a slicing step:

1. G maps a latent tensor z (Gaussian noise of shape 32 × 4 × 4 × 4 in this project) to a 64³ volume
   with one channel per phase (softmax, i.e. a soft one-hot encoding).
2. The generated volume is cut into all 64 slices along each of the three axes (3 × 64 = 192 slices).
3. D, a 2D convolutional network, scores each slice and an equal-sized random crop of the real
   micrograph. For an isotropic material one D serves all three directions.
4. The losses of all slices are combined, and both networks are updated. Because every slice of the
   volume must look like the micrograph, G learns a 3D structure whose sections in x, y and z all match
   the 2D statistics.

![SliceGAN training](figures/slicegan_schematic.png)

Training uses the Wasserstein loss with gradient penalty (WGAN-GP; Gulrajani et al., 2017), in which D is
an unbounded *critic* estimating the Wasserstein distance between real and generated slices, with 5
critic updates per generator update. The paper's Algorithm 1 shows D all 64 slices per direction of every
generated volume and uses a generator batch twice the critic batch (m_G = 2 m_D), which the authors found
most efficient.

**Generator design: uniform information density.** Early SliceGAN versions produced worse quality near
volume edges. The cause is transpose convolution: a voxel near the edge of the output receives
contributions from fewer kernel positions than a central voxel, so information is unevenly distributed.
For microstructures, where edges matter as much as the centre, the authors derive rules for the kernel
size k, stride s and padding p (s < k, k mod s = 0, p ≥ k − s) and use {k, s, p} = {4, 2, 2}. They also
give the latent z a spatial size of 4 instead of 1, so that the first layer already learns overlapping
kernel outputs; as a consequence, volumes larger than 64³ can be generated after training by simply
enlarging z. The released code (used unchanged here) replaces the last transpose convolution by an
upsample + convolution ("resize-convolution") to avoid checkerboard artifacts. The critic is a plain
2D CNN of five strided convolutions (64 × 64 slice → one score).

**Scope and limits.** SliceGAN reproduces the micrograph's statistics without hand-picked descriptors,
trains in a few hours on one GPU and generates volumes in seconds. It was validated against real 3D
data of a battery electrode and later applied to 87 materials in the MicroLib library (Kench et al.,
2022). Its main assumptions are isotropy (anisotropic materials need two or three perpendicular
micrographs and separate critics) and a field of view that is representative of the material.

### 1.2 Background: Swin-T and SAM, and where they enter the pipeline

**Swin Transformer (Swin-T; Liu et al., 2021)** is a hierarchical Vision Transformer:

- the image is split into 4 × 4 patches, each embedded as a token;
- self-attention is computed inside local 7 × 7-token windows, which shift between consecutive layers so
  information also flows across windows;
- four stages merge patches between them, giving feature maps at decreasing resolution (like a CNN);
- position is encoded by a relative-position bias inside each window, not by absolute embeddings, so the
  network also runs on small 64 × 64 slices;
- Swin-T has 28 M parameters and is pretrained on ImageNet-1k.

In this project Swin-T replaces (M2, M3) or complements (M4, M5) SliceGAN's CNN as the 2D critic, bringing
pretrained features and attention over the whole slice. Transformer critics are known to destabilize GAN
training (ViTGAN; Lee et al., 2022), which turned out to be the central difficulty (Section 5.2).

**Segment Anything (SAM; Kirillov et al., 2023)** is a promptable segmentation model: a ViT image encoder,
a prompt encoder (points, boxes) and a light mask decoder, trained on 1 billion masks. In automatic mode a
grid of point prompts yields a mask for every object, zero-shot.

**How SAM is included.** SliceGAN needs a segmented micrograph (one label per phase) as training data;
the baseline obtains it with a global gray-level threshold (Otsu). SAM is used as an alternative
*front-end* for this segmentation step, and the rest of the pipeline is unchanged:

1. SAM ViT-B (`facebook/sam-vit-base`, no fine-tuning) runs on overlapping 256 px tiles of the micrograph
   with a 32 × 32 point grid per tile (a single pass over the whole image misses most small islands);
2. masks that overlap (IoU > 0.3) are merged;
3. each merged group is labelled inclusion or matrix by its mean gray level (a two-class split of the
   group means; uncovered pixels are matrix);
4. the resulting phase map replaces the Otsu map as the training image of model M3.

![SAM front-end](figures/sam_frontend.png)

### 1.3 Research questions

- **RQ1 — ViT critic.** Does replacing SliceGAN's CNN discriminator by a Swin Transformer (Swin-T)
  improve the generated microstructures? (M1 vs M2, with an M1 + DiffAug ablation)
- **RQ2 — SAM front-end.** Does segmenting the micrograph with SAM, instead of a global threshold,
  improve them? (M2 vs M3)
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

**Metrics.** For each model, 128 64³ volumes (seeds 0–127) are compared with the 2D image through φ,
S₂(r) and L(r), overall and per slice orientation, with bootstrap confidence intervals (Section 4).

## 3. Technical implementation

**Stack.** Python 3.11, PyTorch 2.5.1 (CUDA 12.4), timm 1.0.11, Hugging Face `transformers` 4.46.3,
NumPy/SciPy/scikit-image, Poetry environment (`setup.sh`), YAML configs with inheritance and dataset
overlays, logging to file and console, MLflow experiment tracking, 74 pytest tests. Trained on one NVIDIA RTX A2000 (12 GB).

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

**Checkpoint selection and evaluation protocol.** After every epoch (100 generator steps) the generator
produces 16 volumes from held-out seeds (1000–1015), and the epoch with the lowest S₂ MAE against the
model's own training image is kept. The kept checkpoint is then evaluated on 128 *different* seeds
(0–127), so the reported numbers are never measured on the volumes used to choose the checkpoint. The
same rule is applied to every model. The protocol was tightened twice during the project, because single
64³ volumes are small samples of the microstructure (≈ 30 islands; phase fraction varies by ±0.05 from
volume to volume): with 2 selection seeds and 4 evaluation seeds (run v5), model rankings reversed when
re-evaluated on 32 seeds, and a checkpoint picked as best on 2 seeds could be three times worse than the
last epoch on fresh seeds. Even with 16 seeds, choosing the best of 50 noisy epoch scores is optimistic
(the winner's curse: M1's selected checkpoint scored S₂ MAE 0.0016 on its selection seeds but 0.020 on
128 fresh seeds); evaluation on independent seeds removes that bias from the reported numbers.

**RVE export.** `src/models/export_rve.py` writes generated microstructures for FEM/FFT homogenization
codes (VTK `.vti`, MetaImage `.mhd/.raw`, Abaqus `.inp` voxel mesh with phase element sets and face node
sets, NumPy, TIFF, plus JSON metadata). Volumes larger than 64³ are obtained by enlarging the latent
input; periodic RVEs use SliceGAN's latent tiling with a 2-voxel crop, which we found to be the correct
period (the upstream 1-voxel crop duplicates a slice at the seam); periodicity is verified by comparing
the wrap-around face mismatch with interior slice-to-slice mismatch (ratio 1.0–2.0 vs 8–13 without
tiling).

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
N = 128 generated 64³ volumes per model (different seeds; the brief requires N ≥ 4).

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
- **Uncertainty.** 95 % confidence intervals for `|Δφ|`, `S₂ MAE` and `L MAE` come from a bootstrap over
  the 128 volumes (2000 resamples, each recomputing the metric exactly as reported). Each model is
  compared with the baseline M1 through the bootstrap distribution of the difference; a model is called
  better or worse only if that interval excludes zero.
- **Typical quality during training (selection-free).** The per-epoch held-out scores are unbiased
  individually; only picking their minimum is optimistic. The median and interquartile range of the
  per-epoch held-out S₂ MAE over the second half of training, and the number of collapsed epochs
  (held-out φ < 0.05), summarize how good and how stable a model is without any selection.
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

### 5.1 Main results (MicroLib 000210)

Train / validation / test protocol (Section 3): each model's checkpoint is chosen among the per-epoch
best, the last epoch and the training snapshots on 128 validation seeds, then evaluated on 128 test
volumes against the common reference (Otsu map, φ = 0.232). Brackets: 95 % bootstrap intervals.
"Typical" is the selection-free median held-out S₂ MAE over the second half of training (16 seeds per
epoch, so on a different scale from the test columns).

| Model | Checkpoint | φ | \|Δφ\| | S₂ MAE | L MAE | Typical S₂ | vs M1 |
|-------|------------|---|--------|--------|-------|------------|-------|
| M1 CNN (SliceGAN) | last | 0.235 | 0.003 [0.000, 0.012] | 0.0029 [0.0008, 0.0087] | 0.0011 [0.0003, 0.0059] | **0.0051** | — |
| M1 + DiffAug | last | 0.229 | 0.003 [0.000, 0.012] | **0.0009** [0.0006, 0.0066] | 0.0014 [0.0010, 0.0059] | 0.0136 | no clear difference |
| M2 Swin + DiffAug | best | 0.239 | 0.007 [0.000, 0.017] | 0.0077 [0.0017, 0.0141] | 0.0055 [0.0010, 0.0109] | 0.0132 | no clear difference |
| M3 Swin + DiffAug, SAM map | last | 0.174 | 0.058 [0.050, 0.066] | 0.0328 [0.0281, 0.0371] | 0.0262 [0.0223, 0.0299] | 0.0143 | **worse** (all metrics) |
| M4 CNN + frozen Swin | best (= last) | **0.232** | **0.000** [0.000, 0.009] | 0.0014 [0.0006, 0.0067] | 0.0015 [0.0006, 0.0058] | 0.0096 | no clear difference |
| M5 M1 + Swin fine-tune | last | 0.243 | 0.010 [0.001, 0.020] | 0.0097 [0.0033, 0.0161] | 0.0092 [0.0038, 0.0147] | 0.0072 | worse on L |
| M1 extended | last | 0.238 | 0.006 [0.000, 0.015] | 0.0059 [0.0010, 0.0124] | 0.0059 [0.0019, 0.0114] | 0.0097 | no clear difference |

All errors are far below the random-volume floor (S₂ MAE 0.042, L MAE 0.077). The CNN-based models
(M1, M1 + DiffAug, M4) form the best group: their point estimates are the lowest and their intervals
overlap. M4, the CNN + frozen-Swin ensemble, reproduces the phase fraction exactly (0.232) and needed no
checkpoint selection (its best epoch is its last). No model is significantly better than M1 with 128 test
volumes; M3 is significantly worse, because its training image (the SAM map) has a lower phase fraction
and different morphology than the Otsu reference, and its last checkpoint undershoots φ further.

![Descriptor curves](figures/microlib_000210_descriptors.png)

![Qualitative comparison](figures/microlib_000210_qualitative.png)

Per-model training curves: `figures/microlib_000210_training.png`; all volumes can be explored in the
interactive viewer (`reports/viewer/`).

### 5.2 Training a Swin critic: stabilization study

Transformer critics are known to make GAN training unstable (ViTGAN, Lee et al., 2022). With SliceGAN's
WGAN-GP setup the Swin-T critic was unstable in every configuration tried; the table summarizes the
first 12 epochs on MicroLib (held-out φ target 0.232; S₂ MAE of M1's best epochs: 0.002–0.004).

| Run | Swin critic | Loss | DiffAug | Epochs 1–12 | Outcome |
|-----|-------------|------|---------|-------------|---------|
| v1 | stages 3–4 trainable, lr 1e-4, last checkpoint | WGAN-GP | no | realistic at epoch 10, empty at 15 | oscillates, ended collapsed (φ = 0.003) |
| v2 | stages 3–4, lr 2e-5 | WGAN-GP | no | empty at 4–5, S₂ MAE 0.010 at 9–11 | oscillates, M1-level epochs exist |
| v2 rerun | identical settings and seed | WGAN-GP | no | good at epoch 2, then φ ≈ 0 for 10 epochs | run-to-run variability: non-deterministic GPU kernels, amplified by the GAN dynamics |
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

### 5.3 Fine-tuning with a Swin critic (M5 vs M1-extended)

Both runs start from M1's selected checkpoint and train 20 more epochs with fresh critics; they differ
only in the critics (CNN vs CNN + frozen Swin with per-position heads). Neither improves on M1: on the
test volumes M1-extended reaches S₂ MAE 0.0059 and M5 0.0097 (M1: 0.0029), and their best per-epoch
held-out scores were reached in the first epoch, i.e. at the starting point. Once SliceGAN has converged,
adding the Swin critic as a fine-tuning stage (the Vision-aided GAN recipe) does not help here.
An earlier version of this comparison (run v5, selection on 2 seeds) suggested the opposite; that result
was selection noise and disappeared on more seeds (Section 3).

![M5 vs M1 extended (run v5 selection traces)](figures/m5_vs_m1_extended.png)

### 5.4 SAM front-end

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

**Future work.** Anisotropic materials (three-view SliceGAN), homogenization of the exported periodic
RVEs (FEM/FFT/FNO) with an RVE-size convergence study, Swin critics on 128 px inputs (suggested by
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
