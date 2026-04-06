#!/bin/bash
#SBATCH --job-name=trm-mnist
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

# Move to this bundle root regardless of submission directory.
BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$BUNDLE_ROOT/src"

mkdir -p "$BUNDLE_ROOT/logs"
cd "$SRC_DIR"

# Optional: load your site modules.
# module load cuda/12.1

# Create venv only once.
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# Install deps once.
pip install --upgrade pip wheel setuptools
pip install -r ../requirements/requirements.txt

# Build MNIST dataset (idempotent).
python dataset/build_mnist_dataset.py --output-dir data/mnist

# Train MNIST.
python pretrain.py --config-name cfg_pretrain_mnist arch=trm +run_name=pretrain_mnist_cluster
