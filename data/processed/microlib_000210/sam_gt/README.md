# sam_gt — SAM evaluation crops (MicroLib 000210)

`make sam DATA=configs/data/microlib_000210.yaml` writes 5 non-overlapping 64x64 crops (positions from `sam.n_gt_crops` and the
global seed, listed in `metrics.yaml`):

- `crop_XX_gray.png` — grayscale micrograph crop (not versioned)
- `crop_XX_sam.png` — SAM label map for that crop (not versioned)
- `crop_XX_gt.png` — ground truth, 0 = matrix (black), 1 = inclusion/pore (white)

For MicroLib there is no ground truth, so the GT crops are **manual corrections of the SAM
prediction**: open `crop_XX_sam.png` next to `crop_XX_gray.png` in an image editor (zoom in),
paint missed inclusion pixels white and false positives black, and save the result as
`crop_XX_gt.png` (8-bit grayscale PNG, 64x64). Then rerun `make sam DATA=...` to compute the
metrics. Document here what was corrected (e.g. "crop 02: added one missed island").

Status: TODO — crops not corrected yet.

IoU/Dice of SAM (and of Otsu, for comparison) against the GT crops are written to `metrics.yaml`
and to `train_sam/meta.yaml`.
