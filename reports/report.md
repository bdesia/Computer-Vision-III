# SliceGAN with a Vision Transformer discriminator and a SAM front-end

Vision Transformers — FIUBA. Individual work.

## 1. Project goal

TBD

## 2. Overall architecture

![Pipeline](figures/pipeline.png)

TBD — components: data, SAM, 3D generator, CNN D, Swin D, slicer, metrics.

## 3. Technical implementation

TBD — tools, modules, Hugging Face model IDs, SliceGAN fork.

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
- **SAM segmentation quality** (M3 front-end only): IoU `= |P ∩ G| / |P ∪ G|` and Dice
  `= 2|P ∩ G| / (|P| + |G|)` of the SAM label map `P` against manually corrected masks `G` on
  5 crops of 64×64.

Reference scales: on the synthetic dataset, the Otsu segmentation vs the ground-truth mask gives
`S₂ MAE = 0.0021`. Uncorrelated random 64³ volumes with the correct `φ` of MicroLib 000210 give
`S₂ MAE = 0.042` (`err = 0.42`) and `L MAE = 0.077` (`err = 0.89`), a floor any useful model must beat.

FID and 3D SSIM are not used (they need volumetric ground truth), and no classification accuracy is
reported.

## 5. Results and examples

| Model | φ (mean ± std) | \|Δφ\| | S₂ MAE |
|-------|----------------|--------|--------|
| M1 CNN | TBD | TBD | TBD |
| M2 Swin | TBD | TBD | TBD |
| M3 Swin+SAM | TBD | TBD | TBD |

## 6. Conclusions and future work

TBD

## 7. Planning

| Task | Owner | Status |
|------|-------|--------|
| Repo skeleton, configs, Makefile, setup | Student | Done |
| Dataset (synthetic + MicroLib 000210) | Student | Done |
| φ / S₂ descriptors + tests | Student | Done |
| M1 SliceGAN baseline (integration done; full run pending) | Student | In progress |
| M2 Swin-T discriminator (integration done; full run pending) | Student | In progress |
| M3 SAM front-end (done; MicroLib manual GT crops pending) | Student | In progress |
| Generation, metrics and figures | Student | Pending |
| Report and presentation | Student | Pending |
