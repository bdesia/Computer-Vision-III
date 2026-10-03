# Entry points for the TF pipeline. Run `make help` for the list of targets.
# Windows: use GNU make (e.g. Strawberry `gmake`) or run the python commands directly.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

CFG_M1 := configs/m1_cnn.yaml
CFG_M2 := configs/m2_swin.yaml
CFG_M3 := configs/m3_swin_sam.yaml

# Dataset overlay, e.g. `make train-m1 DATA=configs/data/microlib_000210.yaml` (default: synthetic)
DATA ?=
DATA_ARG := $(if $(DATA),--data $(DATA))

.PHONY: report-pdf help setup vendor data sam train-m1 train-m2 train-m3 generate eval viewer test mlflow-backfill mlflow-ui

help:
	@echo "setup     create .venv with Poetry (DEVICE=cpu for CPU torch)"
	@echo "vendor    fetch SliceGAN into external/SliceGAN (git submodule)"
	@echo "data      download/generate the 2D micrograph and 64x64 crops"
	@echo "sam       SAM zero-shot phase map for M3 (+ IoU/Dice vs sam_gt)"
	@echo "train-m1  SliceGAN baseline (CNN D)"
	@echo "train-m2  SliceGAN with Swin-T D"
	@echo "train-m3  SliceGAN with Swin-T D on SAM phase map"
	@echo "generate  generate N 64^3 volumes per model + per-model metrics"
	@echo "eval      generate + aggregate reports/metrics.csv and figures"
	@echo "viewer    export volumes for reports/viewer/index.html"
	@echo "test      run pytest"
	@echo "mlflow-backfill  import all finished runs (archives, probes, current) into ./mlruns"
	@echo "mlflow-ui        browse ./mlruns at http://127.0.0.1:5000"
	@echo "report-pdf       reports/report(_es).md -> PDF (pandoc + headless Edge)"

setup:
	bash setup.sh

vendor:
	git submodule update --init --recursive

data:
	$(PY) -m src.data.make_dataset --config configs/default.yaml $(DATA_ARG)

sam:
	$(PY) -m src.features.sam_segment --config $(CFG_M3) $(DATA_ARG)

train-m1:
	$(PY) -m src.models.train --config $(CFG_M1) $(DATA_ARG)

train-m2:
	$(PY) -m src.models.train --config $(CFG_M2) $(DATA_ARG)

train-m3:
	$(PY) -m src.models.train --config $(CFG_M3) $(DATA_ARG)

generate:
	$(PY) -m src.models.generate --config $(CFG_M1) $(DATA_ARG)
	$(PY) -m src.models.generate --config $(CFG_M2) $(DATA_ARG)
	$(PY) -m src.models.generate --config $(CFG_M3) $(DATA_ARG)

eval: generate
	$(PY) -m src.visualization.visualize --configs $(CFG_M1) $(CFG_M2) $(CFG_M3) $(DATA_ARG)

viewer:
	$(PY) -m src.visualization.export_viewer $(DATA_ARG)

test:
	$(PY) -m pytest

BACKFILL := models/archive_v1=v1 models/archive_v2=v2 models/archive_v3=v3 models/archive_v4=v4             models/archive_v5=v5 models/archive_probes=probes models/microlib_000210=v6 models/synthetic=v6

mlflow-backfill:
	$(PY) -m src.tracking $(BACKFILL)

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns

report-pdf:
	$(PY) -m src.visualization.build_report
