# sam_gt — SAM evaluation crops (MicroLib 000210)

`make sam DATA=configs/data/microlib_000210.yaml` writes 5 non-overlapping 64x64 crops (positions from `sam.n_gt_crops` and the
global seed, listed in `metrics.yaml`):

- `crop_XX_gray.png` — grayscale micrograph crop (not versioned)
- `crop_XX_sam.png` — SAM label map for that crop (not versioned)
- `crop_XX_gt.png` — ground truth, 0 = matrix (black), 1 = inclusion/pore (white)

For MicroLib there is no ground truth and the crops were not hand-corrected. SAM quality on MicroLib is
reported against the MicroLib-annotated threshold instead (see the main README, SAM section), and against
the exact ground truth on the synthetic dataset.
