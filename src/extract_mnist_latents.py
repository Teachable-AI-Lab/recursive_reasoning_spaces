import argparse
import os
from typing import List

import numpy as np
import torch

from evaluate_mnist import load_config
from pretrain import create_dataloader, create_model, get_default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Extract intermediate MNIST latents across recursion steps.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config-dir", type=str, default="config", help="Hydra config directory")
    parser.add_argument("--config-name", type=str, default="cfg_pretrain_mnist", help="Hydra config name fallback")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples to export")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Optional eval batch-size override")
    parser.add_argument("--data-path", type=str, default=None, help="Optional dataset path override")
    parser.add_argument("--output", type=str, default="latents/mnist_step_latents.npz", help="Output .npz path")
    parser.add_argument("--sanitize-latents", action="store_true", help="Replace NaN/Inf values with 0 in saved latent tensors")
    return parser.parse_args()


def _stack_steps(step_list: List[np.ndarray]) -> np.ndarray:
    # step_list is [num_steps] each with shape [N, D]
    return np.stack(step_list, axis=1)


def main():
    args = parse_args()
    config = load_config(args)

    device = get_default_device()
    print(f"Using device: {device}")

    loader, metadata = create_dataloader(
        config,
        args.split,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    model, _, _ = create_model(config, metadata, rank=0, world_size=1, device=device)
    model.eval()

    # Loss head wraps inner model.
    trm = model.model
    puzzle_emb_len = trm.inner.puzzle_emb_len

    sample_ids: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    images: List[np.ndarray] = []
    final_preds_full: List[np.ndarray] = []
    final_preds_class_only: List[np.ndarray] = []

    z_h_cls_steps: List[List[np.ndarray]] = []
    z_h_pool_steps: List[List[np.ndarray]] = []
    z_l_cls_steps: List[List[np.ndarray]] = []
    z_l_pool_steps: List[List[np.ndarray]] = []

    exported = 0
    seen = 0

    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in loader:
            if exported >= args.num_samples:
                break

            remaining = args.num_samples - exported
            take = min(remaining, batch["inputs"].shape[0])
            batch = {k: v[:take] for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.device(device):
                carry = model.initial_carry(batch)

            step_h_cls: List[np.ndarray] = []
            step_h_pool: List[np.ndarray] = []
            step_l_cls: List[np.ndarray] = []
            step_l_pool: List[np.ndarray] = []

            while True:
                carry, _loss, _metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=["logits"])

                z_h = carry.inner_carry.z_H[:, puzzle_emb_len:]
                z_l = carry.inner_carry.z_L[:, puzzle_emb_len:]

                step_h_cls.append(z_h[:, 0, :].detach().cpu().numpy())
                step_h_pool.append(z_h.mean(dim=1).detach().cpu().numpy())
                step_l_cls.append(z_l[:, 0, :].detach().cpu().numpy())
                step_l_pool.append(z_l.mean(dim=1).detach().cpu().numpy())

                if bool(all_finish):
                    break

            logits = preds["logits"][:, 0, :]
            pred_full = logits.argmax(dim=-1)
            pred_cls_only = logits[:, 257:267].argmax(dim=-1) + 257

            lbl = batch["labels"][:, 0]
            inp = batch["inputs"][:, 1:]

            sample_ids.append(np.arange(seen, seen + take, dtype=np.int32))
            labels.append((lbl - 257).detach().cpu().numpy().astype(np.int32))
            images.append((inp - 1).reshape(take, 28, 28).detach().cpu().numpy().astype(np.uint8))
            final_preds_full.append((pred_full - 257).detach().cpu().numpy().astype(np.int32))
            final_preds_class_only.append((pred_cls_only - 257).detach().cpu().numpy().astype(np.int32))

            z_h_cls_steps.append(step_h_cls)
            z_h_pool_steps.append(step_h_pool)
            z_l_cls_steps.append(step_l_cls)
            z_l_pool_steps.append(step_l_pool)

            seen += take
            exported += take

    if exported == 0:
        raise RuntimeError("No samples were exported. Check dataset path/split.")

    # Convert list-of-batches-of-steps to dense arrays [N, S, D]
    num_steps = len(z_h_cls_steps[0])
    for sample_steps in z_h_cls_steps:
        if len(sample_steps) != num_steps:
            raise RuntimeError("Inconsistent recursion step count across batches.")

    z_h_cls_by_step = [np.concatenate([batch_steps[s] for batch_steps in z_h_cls_steps], axis=0) for s in range(num_steps)]
    z_h_pool_by_step = [np.concatenate([batch_steps[s] for batch_steps in z_h_pool_steps], axis=0) for s in range(num_steps)]
    z_l_cls_by_step = [np.concatenate([batch_steps[s] for batch_steps in z_l_cls_steps], axis=0) for s in range(num_steps)]
    z_l_pool_by_step = [np.concatenate([batch_steps[s] for batch_steps in z_l_pool_steps], axis=0) for s in range(num_steps)]

    z_h_cls = _stack_steps(z_h_cls_by_step)
    z_h_pool = _stack_steps(z_h_pool_by_step)
    z_l_cls = _stack_steps(z_l_cls_by_step)
    z_l_pool = _stack_steps(z_l_pool_by_step)

    def _nonfinite_count_by_step(x: np.ndarray) -> np.ndarray:
        return np.array([np.size(x[:, s, :]) - np.isfinite(x[:, s, :]).sum() for s in range(x.shape[1])], dtype=np.int64)

    nonfinite_stats = {
        "z_h_cls_nonfinite_count_by_step": _nonfinite_count_by_step(z_h_cls),
        "z_h_pool_nonfinite_count_by_step": _nonfinite_count_by_step(z_h_pool),
        "z_l_cls_nonfinite_count_by_step": _nonfinite_count_by_step(z_l_cls),
        "z_l_pool_nonfinite_count_by_step": _nonfinite_count_by_step(z_l_pool),
    }

    if args.sanitize_latents:
        z_h_cls = np.nan_to_num(z_h_cls, nan=0.0, posinf=0.0, neginf=0.0)
        z_h_pool = np.nan_to_num(z_h_pool, nan=0.0, posinf=0.0, neginf=0.0)
        z_l_cls = np.nan_to_num(z_l_cls, nan=0.0, posinf=0.0, neginf=0.0)
        z_l_pool = np.nan_to_num(z_l_pool, nan=0.0, posinf=0.0, neginf=0.0)

    output = {
        "sample_id": np.concatenate(sample_ids, axis=0),
        "label": np.concatenate(labels, axis=0),
        "image": np.concatenate(images, axis=0),
        "pred_full_vocab": np.concatenate(final_preds_full, axis=0),
        "pred_class_only": np.concatenate(final_preds_class_only, axis=0),
        "z_h_cls": z_h_cls,
        "z_h_pool": z_h_pool,
        "z_l_cls": z_l_cls,
        "z_l_pool": z_l_pool,
        "sanitize_latents": np.array([1 if args.sanitize_latents else 0], dtype=np.int32),
        **nonfinite_stats,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **output)

    print(f"Saved {exported} samples to {args.output}")
    print(f"Latent tensor shape z_h_pool: {output['z_h_pool'].shape} (N, steps, hidden)")
    for k, v in nonfinite_stats.items():
        print(f"{k}: {v.tolist()}")


if __name__ == "__main__":
    main()