# external/

Código de terceros vendorizado. No se modifica en el lugar: los cambios (reemplazo de `netD`)
viven en `src/models/`.

- `SliceGAN/` — https://github.com/stke9/SliceGAN (Kench & Cooper, 2021), agregado como
  submódulo git. Para bajarlo: `make vendor` (equivale a `git submodule update --init --recursive`).
