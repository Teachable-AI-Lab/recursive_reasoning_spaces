#!/bin/bash
#SBATCH --job-name=trm-cifar10
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00

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

# Build CIFAR-10 dataset (idempotent).
python dataset/build_cifar10_dataset.py --output-dir data/cifar10

# Train CIFAR-10 (current fast fixed-depth profile used in experiments).
python pretrain.py \
  --config-name cfg_pretrain_cifar10 \
  arch=trm \
  +run_name=pretrain_cifar10_cluster \
  data_paths="[data/cifar10]" \
  epochs=60 \
  global_batch_size=8 \
  arch.halt_max_steps=2 \
  arch.force_fixed_halt_steps=True \
  arch.loss.depth_diversity_coef=0.0
