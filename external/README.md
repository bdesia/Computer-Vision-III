# external/

Vendored third-party code. It is not modified in place: our changes (replacing `netD`) live in
`src/models/`.

- `SliceGAN/` — https://github.com/stke9/SliceGAN (Kench & Cooper, 2021), added as a git submodule.
  To fetch it: `make vendor` (same as `git submodule update --init --recursive`).
