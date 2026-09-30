# SliceGAN con discriminador Vision Transformer y front-end SAM

Trabajo final — **Vision Transformers (FIUBA)**. Docentes: Abraham Rodriguez, Oksana Bokhonok.
Trabajo individual.

## Objetivo

Generar volúmenes 3D (64³) estadísticamente equivalentes a una micrografía 2D de un material
bifásico e isotrópico usando **SliceGAN**, y medir:

1. si un **discriminador Swin-T** (Vision Transformer jerárquico) mejora al discriminador CNN original, y
2. si usar **SAM** como front-end de segmentación de fases mejora los descriptores del volumen
   generado (fracción de fase `φ` y correlación de dos puntos `S₂`).

| ID | Modelo | Entrada 2D | Generador 3D | Discriminador 2D |
|----|--------|------------|--------------|------------------|
| M1 | SliceGAN baseline | imagen de entrenamiento | CNN 3D SliceGAN | CNN SliceGAN |
| M2 | SliceGAN–Swin | la misma que M1 | el mismo G | Swin-T (Hugging Face) |
| M3 | SliceGAN–Swin+SAM | mapa de fases de SAM | el mismo G | el mismo Swin-T que M2 |

- M1 vs M2: ¿aporta el ViT en el discriminador?
- M2 vs M3: ¿aporta SAM como preprocesamiento?

## Estructura

```
configs/            default.yaml + un yaml por modelo (m1_cnn, m2_swin, m3_swin_sam)
data/               raw/ interim/ processed/ (no versionado, salvo processed/sam_gt/)
external/SliceGAN/  SliceGAN upstream (submódulo, sin modificar)
src/data/           make_dataset.py — descarga o genera la 2D y los crops 64x64
src/features/       sam_segment.py (SAM zero-shot), descriptors.py (φ, S₂)
src/models/         wrapper SliceGAN, discriminadores CNN/Swin, train.py, generate.py
src/visualization/  figuras y tabla de métricas
tests/              pytest
reports/            informe.md (fuente del PDF), figures/, metrics.csv
models/             checkpoints (no versionados)
logs/               logs de cada corrida
```

## Setup

Requiere Python ≥ 3.10. GPU NVIDIA recomendada (probado en RTX A2000 12 GB, driver CUDA 12.2).

```bash
git clone --recurse-submodules https://github.com/bdesia/Computer-Vision-III.git
cd Computer-Vision-III
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

O bien `make venv` (con GNU make; en Windows, `gmake`). Si ya clonaste sin submódulos: `make vendor`.

Sin GPU todo corre en CPU (con un WARNING en el log); en ese caso bajar `epochs` en el yaml.

## Datos

```bash
python -m src.data.make_dataset --config configs/default.yaml   # make data
```

Fuente configurada en `configs/default.yaml → data.source`:

- `synthetic` (default actual): inclusiones circulares por RSA, `φ` objetivo 0.25, 512², seed fija.
- `slicegan`: micrografía 2 fases de `external/SliceGAN/Examples/`. TODO: elegir archivo.
- `microlib`: una entrada de [MicroLib](https://microlib.io). TODO: citar el ID elegido.

## Entrenamiento

```bash
python -m src.models.train --config configs/m1_cnn.yaml       # make train-m1
python -m src.features.sam_segment --config configs/m3_swin_sam.yaml   # make sam (requerido por M3)
python -m src.models.train --config configs/m2_swin.yaml      # make train-m2
python -m src.models.train --config configs/m3_swin_sam.yaml  # make train-m3
```

Checkpoints en `models/<run_name>/`, logs en `logs/<run_name>.log`.

## Evaluación

```bash
make eval      # genera N=4 cubos 64³ por modelo y escribe reports/metrics.csv + figuras
make test      # pytest
```

Métricas: `φ` media ± std y `|Δφ|` vs la 2D de entrenamiento; MAE de `S₂(r)` hasta `r = 32`
(promedio de cortes xy/xz/yz); IoU/Dice de SAM sobre 5 crops con GT manual.

## Resultados

TBD — ver `reports/informe.md`.

## Citas

- S. Kench, S. J. Cooper. *Generating three-dimensional structures from a two-dimensional slice with
  generative adversarial network-based dimensionality expansion.* Nature Machine Intelligence, 2021.
  Código: https://github.com/stke9/SliceGAN
- Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- A. Kirillov et al. *Segment Anything.* ICCV 2023.
- A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- S. Kench et al. *MicroLib: A library of 3D microstructures generated from 2D micrographs using
  SliceGAN.* Scientific Data, 2022 (si se usa MicroLib).

## Licencia

MIT (ver `LICENSE`). SliceGAN mantiene su propia licencia.
