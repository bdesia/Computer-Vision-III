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

.PHONY: help venv vendor data sam train-m1 train-m2 train-m3 generate eval test

help:
	@echo "venv      create .venv and install pinned requirements"
	@echo "vendor    fetch SliceGAN into external/SliceGAN (git submodule)"
	@echo "data      download/generate the 2D micrograph and 64x64 crops"
	@echo "sam       SAM zero-shot phase map for M3 (+ IoU/Dice vs sam_gt)"
	@echo "train-m1  SliceGAN baseline (CNN D)"
	@echo "train-m2  SliceGAN with Swin-T D"
	@echo "train-m3  SliceGAN with Swin-T D on SAM phase map"
	@echo "generate  generate N 64^3 volumes per model + per-model metrics"
	@echo "eval      generate + aggregate reports/metrics.csv and figures"
	@echo "test      run pytest"

venv:
	python -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

vendor:
	git submodule update --init --recursive

data:
	$(PY) -m src.data.make_dataset --config configs/default.yaml

sam:
	$(PY) -m src.features.sam_segment --config $(CFG_M3)

train-m1:
	$(PY) -m src.models.train --config $(CFG_M1)

train-m2:
	$(PY) -m src.models.train --config $(CFG_M2)

train-m3:
	$(PY) -m src.models.train --config $(CFG_M3)

generate:
	$(PY) -m src.models.generate --config $(CFG_M1)
	$(PY) -m src.models.generate --config $(CFG_M2)
	$(PY) -m src.models.generate --config $(CFG_M3)

eval: generate
	$(PY) -m src.visualization.visualize --configs $(CFG_M1) $(CFG_M2) $(CFG_M3)

test:
	$(PY) -m pytest
