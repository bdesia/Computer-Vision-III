# Run v1 (2026-09-30 / 10-01) — shared critic lr 1e-4, last-checkpoint evaluation

Superseded by v2 (Swin critic lr 2e-5 + best-checkpoint selection on held-out seeds).
Kept because v1 shows the Swin-critic instability that motivated v2.

- microlib_000210/m2_swin: `history.csv` was rebuilt from the training log (d_real/d_fake are not
  in the log, so those columns are empty); `config.yaml`, `G_last.pt` and `D_last.pt` were
  overwritten by a 2-epoch smoke test of the v2 code and then removed; the smoke-test lines were
  also cut from the end of its log. The v1 config equals the
  current m2_swin.yaml without the `train.lr_d: 2.0e-5` override and without checkpoint selection.
  Its evaluation outputs (volumes/, metrics.yaml, curves.npz) are intact.
