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

There is no 3D ground truth, so generated volumes are compared statistically with the 2D training
image (implemented in `src/features/descriptors.py`, tested in `tests/test_descriptors.py`).
Label 1 is the inclusion/pore phase and `I(x)` its indicator function.

- **Phase fraction** `φ = ⟨I(x)⟩`. For each model, `φ` is averaged over N ≥ 4 generated 64³ volumes
  (different seeds) and reported as mean ± std, together with `|Δφ| = |φ̄_gen − φ_train|`.
- **Two-point correlation** `S₂(r) = P[I(x) = 1, I(x + r) = 1]`, computed by FFT autocorrelation.
  For non-periodic images each lag is normalized by its number of valid pixel pairs. The 2D map is
  radially averaged over lag vectors with `round(|r|) = r`, for `r = 0 … 32`. `S₂(0) = φ` and
  `S₂(r) → φ²` for uncorrelated points.
  For a volume, `S₂` is averaged over all xy, xz and yz slices (192 slices for 64³), so the metric
  also penalizes anisotropy between the three orientations.
- **S₂ MAE** `= mean_r |S₂_gen(r) − S₂_train(r)|` over `r = 0 … 32`, using the mean curve of the
  N volumes; the std of the per-volume MAE is also reported. Reference scale: on the synthetic
  dataset, the Otsu segmentation vs the ground-truth mask gives an `S₂` MAE of 0.0021.
- **SAM segmentation quality** (M3 front-end only): IoU `= |P ∩ G| / |P ∪ G|` and Dice
  `= 2|P ∩ G| / (|P| + |G|)` of the SAM label map `P` against manually corrected masks `G` on
  5 crops of 64×64.

3D SSIM is not used (there is no volumetric ground truth), and no classification accuracy is reported.

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
| M3 SAM front-end | Student | Pending |
| Generation, metrics and figures | Student | Pending |
| Report and presentation | Student | Pending |
