# sam_gt — SAM evaluation crops (synthetic)

`make sam` writes 5 non-overlapping 64x64 crops (positions from `sam.n_gt_crops` and the
global seed, listed in `metrics.yaml`):

- `crop_XX_gray.png` — grayscale micrograph crop (not versioned)
- `crop_XX_sam.png` — SAM label map for that crop (not versioned)
- `crop_XX_gt.png` — ground truth, 0 = matrix (black), 1 = inclusion/pore (white)

For the synthetic dataset the GT crops are cut from the exact ground-truth mask
(`data/raw/synthetic/micro_2d_gt.png`), so no manual work is needed.

IoU/Dice of SAM (and of Otsu, for comparison) against the GT crops are written to `metrics.yaml`
and to `train_sam/meta.yaml`.
