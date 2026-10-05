# Entry points for the TF pipeline. Run `make help` for the list of targets.
# Windows: use GNU make (e.g. Strawberry `gmake`) or run the python commands directly.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
JEXEC ?= .venv/Scripts/jupyter-execute.exe
else
PY ?= .venv/bin/python
JEXEC ?= .venv/bin/jupyter-execute
endif

# Dataset overlay, e.g. `make train-all DATA=configs/data/microlib_000210.yaml` (default: synthetic)
DATA ?=
DATA_ARG := $(if $(DATA),--data $(DATA))

# Models (configs/<name>.yaml). TRAIN_ORDER puts M1 first: M1-extended and M5 warm-start from its G_best.
MODELS := m1_cnn m1_cnn_diffaug m1_extended m2_swin m3_swin_sam m4_ensemble m5_finetune
TRAIN_ORDER := m1_cnn m1_cnn_diffaug m1_extended m5_finetune m2_swin m3_swin_sam m4_ensemble
CONFIGS := $(foreach m,$(MODELS),configs/$(m).yaml)

.PHONY: help setup vendor data sam train-all train-m1 train-m1-diffaug train-m1-ext train-m2 train-m3 \
        train-m4 train-m5 generate eval viewer test report-pdf mlflow-backfill mlflow-ui export-rve eda

help:
	@echo "setup              create .venv with Poetry (DEVICE=cpu for CPU torch)"
	@echo "vendor             fetch SliceGAN into external/SliceGAN (git submodule)"
	@echo "data               download/generate the 2D micrograph and 64x64 crops"
	@echo "sam                SAM zero-shot phase map for M3 (+ IoU/Dice vs references)"
	@echo "eda                execute notebooks/00_eda.ipynb in place (needs make data + make sam, both datasets)"
	@echo "train-m1           M1  SliceGAN baseline (CNN critic)"
	@echo "train-m1-diffaug   M1 + DiffAug ablation (CNN critic with DiffAug)"
	@echo "train-m1-ext       M1-extended: M1's best G + 20 epochs, CNN critic (baseline for M5)"
	@echo "train-m2           M2  Swin-T critic + DiffAug"
	@echo "train-m3           M3  Swin-T critic + DiffAug on the SAM phase map"
	@echo "train-m4           M4  CNN + frozen-Swin critics (ensemble), from scratch"
	@echo "train-m5           M5  M1's best G + 20 epochs with the M4 ensemble (needs train-m1)"
	@echo "train-<config>     any configs/<config>.yaml, e.g. make train-m2_swin_patchheads"
	@echo "train-all          all seven models in dependency order"
	@echo "generate           128 volumes per model + metrics (metrics.yaml, curves.npz)"
	@echo "eval               generate + metrics.csv, comparison vs M1, figures, viewer data"
	@echo "viewer             export 4 volumes per run (both datasets) for reports/viewer/index.html"
	@echo "export-rve         RVEs for FEM/FFT codes (MODEL=m4_ensemble SIZE=128 SEEDS=\"0 1 2\" FORMATS=\"vti mhd inp\" PERIODIC=1)"
	@echo "test               run pytest"
	@echo "report-pdf         reports/report(_es).md -> PDF (pandoc + headless Edge)"
	@echo "mlflow-backfill    import all finished runs (archives, probes, current) into ./mlruns"
	@echo "mlflow-ui          browse ./mlruns at http://127.0.0.1:5000"
	@echo "All model targets take DATA=configs/data/microlib_000210.yaml (default: synthetic)."

setup:
	bash setup.sh

vendor:
	git submodule update --init --recursive

data:
	$(PY) -m src.data.make_dataset --config configs/default.yaml $(DATA_ARG)

sam:
	$(PY) -m src.features.sam_segment --config configs/m3_swin_sam.yaml $(DATA_ARG)

eda:
	$(JEXEC) --inplace notebooks/00_eda.ipynb

# ---------------------------------------------------------------- training

train-%:
	$(PY) -m src.models.train --config configs/$*.yaml $(DATA_ARG)

train-m1: train-m1_cnn
train-m1-diffaug: train-m1_cnn_diffaug
train-m1-ext: train-m1_extended
train-m2: train-m2_swin
train-m3: train-m3_swin_sam
train-m4: train-m4_ensemble
train-m5: train-m5_finetune

define run_train
	$(PY) -m src.models.train --config configs/$(1).yaml $(DATA_ARG)

endef

train-all:
	$(foreach m,$(TRAIN_ORDER),$(call run_train,$(m)))

# ---------------------------------------------------------------- evaluation

define run_generate
	$(PY) -m src.models.generate --config configs/$(1).yaml $(DATA_ARG)

endef

generate:
	$(foreach m,$(MODELS),$(call run_generate,$(m)))

eval: generate
	$(PY) -m src.visualization.visualize --configs $(CONFIGS) $(DATA_ARG)
	$(PY) -m src.visualization.export_viewer

viewer:
	$(PY) -m src.visualization.export_viewer

test:
	$(PY) -m pytest

MODEL ?= m4_ensemble
SIZE ?= 128
SEEDS ?= 0 1 2
FORMATS ?= vti mhd npy
PERIODIC ?= 1

export-rve:
	$(PY) -m src.models.export_rve --config configs/$(MODEL).yaml $(DATA_ARG) --size $(SIZE) --seeds $(SEEDS) --formats $(FORMATS) $(if $(filter 1,$(PERIODIC)),--periodic)

report-pdf:
	$(PY) -m src.visualization.build_report

# ---------------------------------------------------------------- experiment tracking

BACKFILL := models/archive_v1=v1 models/archive_v2=v2 models/archive_v3=v3 models/archive_v4=v4 \
            models/archive_v5=v5 models/archive_probes=probes models/microlib_000210=v6 models/synthetic=v6

mlflow-backfill:
	$(PY) -m src.tracking $(BACKFILL)

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns
