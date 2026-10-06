#!/bin/bash
# Creates the in-project Poetry environment (.venv), installs PyTorch for $DEVICE,
# registers a Jupyter kernel and adds the repo root to the environment's sys.path.
#
# Usage:
#   bash setup.sh               # CUDA 12.4 wheels (default)
#   DEVICE=cpu bash setup.sh    # CPU-only wheels
set -e  # Exit on first error

PYPROJECT="pyproject.toml"
PROJECT_NAME=$(grep -m1 '^name =' "$PYPROJECT" | awk -F '"' '{print $2}')

PYTHON_VERSION="3.11"
DEVICE=${DEVICE:-cuda}
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"

# Validate that the project name was extracted
if [ -z "$PROJECT_NAME" ]; then
    echo "Error: could not extract the project name from $PYPROJECT"
    exit 1
fi
echo "Project name: $PROJECT_NAME"

echo "Configuring environment with Poetry..."

# Keep the virtual environment inside the repo (.venv)
poetry config virtualenvs.in-project true --local

# Clear the cache before creating the environment
poetry cache clear pypi --all -q

# Remove any previous .venv to avoid issues with a broken environment
rm -rf .venv

# Use or create a virtualenv with the given Python version
poetry env use "$PYTHON_VERSION"

# PyTorch: pin the wheel source for the requested device (updates pyproject + lock only)
echo "Selecting PyTorch source ($DEVICE)..."
poetry remove torch torchvision --lock --quiet || true

if [ "$DEVICE" = "cpu" ]; then
    echo "-> CPU build"
    poetry add "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --source pytorch-cpu --lock
else
    echo "-> CUDA 12.4 build"
    poetry add "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --source pytorch-cu124 --lock
fi

# Install exactly what the lock file specifies (main, dev and app groups)
poetry sync --with dev,app

# Register the Jupyter kernel
poetry run python -m ipykernel install --user --name="$PROJECT_NAME" --display-name "Python ($PROJECT_NAME)"

# Add the repo root to sys.path via a .pth file, so `import src...` works from anywhere
# (notebooks, other working directories). Python resolves the site-packages path itself.
poetry run python - "$PROJECT_NAME" <<'EOF'
import os
import sys
import sysconfig
from pathlib import Path

pth = Path(sysconfig.get_paths()["purelib"]) / f"{sys.argv[1]}.pth"
pth.parent.mkdir(parents=True, exist_ok=True)
pth.write_text(os.getcwd() + "\n", encoding="utf-8")
print(f".pth file created at: {pth}")
print(f"Contents: {pth.read_text(encoding='utf-8').strip()}")
EOF

# Sanity check
poetry run python -c "import torch; print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

echo "The Poetry environment is ready."
