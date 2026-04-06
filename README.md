# TRM MNIST/CIFAR Cluster Handoff

## What this package is
This directory is a portable subset of the TinyRecursiveModels repo containing what is needed to:
- Build MNIST and CIFAR-10 tokenized datasets for this codebase
- Train TRM on MNIST and CIFAR-10
- Evaluate CIFAR-10 and MNIST checkpoints
- Reproduce the latent extraction/clustering/visualization scripts used in the experiments

This package was prepared to move runs from local development to a GPU cluster.

## Directory layout
- `src/`
  - `pretrain.py`, `puzzle_dataset.py`
  - `config/`
    - `cfg_pretrain_mnist.yaml`
    - `cfg_pretrain_cifar10.yaml`
    - `arch/trm.yaml`
  - `dataset/`
    - `build_mnist_dataset.py`
    - `build_cifar10_dataset.py`
    - `common.py`
  - `evaluators/`
    - `mnist.py`
    - `cifar10.py`
  - `models/`
    - full model package required by dynamic imports
  - `utils/functions.py`
  - Optional analysis scripts:
    - `extract_*_latents.py`
    - `cluster_*_latent_tree.py`
    - `visualize_*_latent_tree*.py`
- `requirements/`
  - `requirements.txt`
  - `specific_requirements.txt`

## What happened in this project (summary)
1. Added a full CIFAR-10 pipeline analogous to MNIST:
   - dataset builder
   - evaluator
   - training config
   - checkpoint evaluation script
   - latent extraction + tree clustering + visualizations
2. Ran CIFAR-10 training with fixed-depth ACT settings.
3. Main observed behavior:
   - Class-token range adherence improved strongly in later runs (`pred_in_class_range` near 0.99).
   - CIFAR-10 top-1 remained modest (~0.28) with tiny/fast settings.
4. Main bottleneck:
   - CIFAR sequence length is much larger than MNIST in this tokenization, so compute and optimization are harder even at similar parameter counts.

## Cluster setup
From inside this handoff directory:

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r ../requirements/requirements.txt
```

If you need exact pinned versions from the original environment:

```bash
pip install -r ../requirements/specific_requirements.txt
```

## Build datasets
Run from `src/`:

```bash
# MNIST
python dataset/build_mnist_dataset.py --output-dir data/mnist

# CIFAR-10
python dataset/build_cifar10_dataset.py --output-dir data/cifar10
```

## Train runs
Run from `src/`.

### MNIST baseline command
```bash
python pretrain.py --config-name cfg_pretrain_mnist arch=trm +run_name=pretrain_mnist_cluster
```

### CIFAR-10 baseline command
```bash
python pretrain.py --config-name cfg_pretrain_cifar10 arch=trm +run_name=pretrain_cifar10_cluster data_paths="[data/cifar10]"
```

### CIFAR-10 fast fixed-depth variant used in experiments
```bash
python pretrain.py \
  --config-name cfg_pretrain_cifar10 \
  arch=trm \
  +run_name=pretrain_cifar10_fixeddepth_cluster \
  data_paths="[data/cifar10]" \
  epochs=60 \
  global_batch_size=8 \
  arch.halt_max_steps=2 \
  arch.force_fixed_halt_steps=True \
  arch.loss.depth_diversity_coef=0.0
```

## Evaluate checkpoints
Run from `src/`.

```bash
python evaluate_mnist.py --checkpoint checkpoints/MNIST-ACT-torch/<run_name>/step_<N>
python evaluate_cifar10.py --checkpoint checkpoints/CIFAR10-ACT-torch/<run_name>/step_<N>
```

## Checkpoint locations
By default checkpoints are saved at:
- `checkpoints/<project_name>/<run_name>/step_<N>`

Examples:
- `checkpoints/MNIST-ACT-torch/...`
- `checkpoints/CIFAR10-ACT-torch/...`

## Optional latent hierarchy analysis
Run from `src/` with a trained checkpoint.

```bash
python extract_cifar10_latents.py --checkpoint checkpoints/CIFAR10-ACT-torch/<run_name>/step_<N> --num-samples 512 --output latents/cifar10_step_latents.npz
python cluster_cifar10_latent_tree.py --latents latents/cifar10_step_latents.npz --latent-key z_h_pool --root-clusters 2 --leaf-clusters 10 --save-json latents/cifar10_step_tree.json
python visualize_cifar10_latent_tree_images.py --tree-json latents/cifar10_step_tree.json --latents latents/cifar10_step_latents.npz --latent-key z_h_pool --output latents/cifar10_step_tree_images.png
```

## Notes for cluster jobs
- If you use SLURM, keep working directory at `src/` so relative paths resolve.
- If GPU memory is tight, reduce `global_batch_size` first.
- If training is too slow, reduce `arch.halt_max_steps` and/or model width.
- If quality stalls, increase model width and/or recursion depth.

## Included SLURM templates
This handoff includes ready-to-edit scripts in `jobs/`:
- `jobs/train_mnist.slurm`
- `jobs/train_cifar10.slurm`

Submit from the handoff root:

```bash
mkdir -p logs
sbatch jobs/train_mnist.slurm
sbatch jobs/train_cifar10.slurm
```

Before first submit, edit the SBATCH lines for your cluster:
- `--partition`
- `--gres=gpu:*`
- `--time`
- `--mem`
- any account/QoS directives required by your site
