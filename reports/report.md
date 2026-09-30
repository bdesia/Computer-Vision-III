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

TBD — definitions of `φ`, `|Δφ|`, `S₂(r)` and its MAE, SAM IoU/Dice.

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
| Dataset (synthetic + real) | Student | Pending |
| φ / S₂ descriptors + tests | Student | Pending |
| M1 SliceGAN baseline | Student | Pending |
| M2 Swin-T discriminator | Student | Pending |
| M3 SAM front-end | Student | Pending |
| Generation, metrics and figures | Student | Pending |
| Report and presentation | Student | Pending |
