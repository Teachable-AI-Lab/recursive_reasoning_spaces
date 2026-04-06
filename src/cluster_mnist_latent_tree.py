import argparse
import json
import os
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_args():
    parser = argparse.ArgumentParser(description="Build a merge tree from MNIST latents across recursion depths.")
    parser.add_argument("--latents", type=str, required=True, help="Path to .npz from extract_mnist_latents.py")
    parser.add_argument("--latent-key", type=str, default="z_h_pool", choices=["z_h_pool", "z_h_cls", "z_l_pool", "z_l_cls"], help="Latent tensor key")
    parser.add_argument("--leaf-clusters", type=int, default=10, help="Number of clusters at final depth")
    parser.add_argument("--root-clusters", type=int, default=2, help="Number of clusters at first depth")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=300, help="KMeans max iterations")
    parser.add_argument("--min-valid-samples", type=int, default=50, help="Minimum valid samples required after NaN/Inf filtering")
    parser.add_argument("--save-json", type=str, default="latents/mnist_latent_tree.json", help="Output JSON path")
    return parser.parse_args()


def make_cluster_schedule(num_steps: int, root_k: int, leaf_k: int) -> List[int]:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if root_k < 1 or leaf_k < 1:
        raise ValueError("cluster counts must be >= 1")
    if root_k > leaf_k:
        raise ValueError("root_clusters must be <= leaf_clusters")

    if num_steps == 1:
        return [leaf_k]

    schedule = []
    for i in range(num_steps):
        frac = i / (num_steps - 1)
        k = int(round(root_k + frac * (leaf_k - root_k)))
        schedule.append(k)

    schedule[0] = root_k
    schedule[-1] = leaf_k

    # Ensure monotonic non-decreasing schedule.
    for i in range(1, len(schedule)):
        if schedule[i] < schedule[i - 1]:
            schedule[i] = schedule[i - 1]

    return schedule


def main():
    args = parse_args()
    data = np.load(args.latents)

    if args.latent_key not in data:
        raise KeyError(f"{args.latent_key} not found in {args.latents}")

    latents = data[args.latent_key]  # [N, S, D]
    labels = data["label"]

    if latents.ndim != 3:
        raise ValueError(f"Expected [N, steps, D], got {latents.shape}")

    # Filter samples that contain any non-finite value across any depth/dimension.
    finite_sample_mask = np.isfinite(latents).all(axis=(1, 2))
    dropped = int((~finite_sample_mask).sum())
    if dropped > 0:
        print(f"Dropping {dropped} samples with NaN/Inf latents before clustering")

    latents = latents[finite_sample_mask]
    labels = labels[finite_sample_mask]

    if latents.shape[0] < args.min_valid_samples:
        raise ValueError(
            f"Too few valid samples after filtering: {latents.shape[0]} < {args.min_valid_samples}. "
            "Try extracting more samples or inspect latent NaNs."
        )

    # Final safety: convert remaining non-finite values (if any) to finite numbers.
    latents = np.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

    n, num_steps, d = latents.shape
    schedule = make_cluster_schedule(num_steps, args.root_clusters, args.leaf_clusters)

    assignments = []
    per_step_metrics = []

    for step in range(num_steps):
        x = latents[:, step, :]
        k = schedule[step]
        kmeans = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=10,
            max_iter=args.max_iter,
        )
        cluster_id = kmeans.fit_predict(x)
        assignments.append(cluster_id)

        per_step_metrics.append({
            "step": int(step + 1),
            "num_clusters": int(k),
            "ari_vs_label": float(adjusted_rand_score(labels, cluster_id)),
            "nmi_vs_label": float(normalized_mutual_info_score(labels, cluster_id)),
            "inertia": float(kmeans.inertia_),
        })

    # Build tree edges from depth t+1 children to depth t parents by overlap majority.
    tree_edges = []
    transition_stats = []
    for step in range(num_steps - 1):
        parent = assignments[step]
        child = assignments[step + 1]

        parent_k = schedule[step]
        child_k = schedule[step + 1]
        contingency = np.zeros((parent_k, child_k), dtype=np.int64)

        for i in range(n):
            contingency[parent[i], child[i]] += 1

        # Child -> parent mapping via max overlap (merge links).
        child_to_parent = contingency.argmax(axis=0)
        child_purity = contingency.max(axis=0) / np.maximum(contingency.sum(axis=0), 1)

        edges_at_step = []
        for c in range(child_k):
            p = int(child_to_parent[c])
            w = int(contingency[p, c])
            purity = float(child_purity[c])
            edge = {
                "from_step": int(step + 2),
                "from_cluster": int(c),
                "to_step": int(step + 1),
                "to_cluster": p,
                "overlap": w,
                "purity": purity,
            }
            edges_at_step.append(edge)
            tree_edges.append(edge)

        transition_stats.append({
            "from_step": int(step + 2),
            "to_step": int(step + 1),
            "avg_child_purity": float(np.mean(child_purity)),
            "min_child_purity": float(np.min(child_purity)),
            "max_child_purity": float(np.max(child_purity)),
            "nmi_parent_child": float(normalized_mutual_info_score(parent, child)),
            "ari_parent_child": float(adjusted_rand_score(parent, child)),
            "edges": edges_at_step,
        })

    print(f"Loaded {n} samples, {num_steps} recursion steps, latent dim {d}")
    print(f"Cluster schedule (step 1 -> {num_steps}): {schedule}")
    print("\nPer-step quality vs digit labels:")
    for m in per_step_metrics:
        print(
            f"step={m['step']:02d} k={m['num_clusters']:02d} "
            f"ARI={m['ari_vs_label']:.4f} NMI={m['nmi_vs_label']:.4f}"
        )

    print("\nTree merge quality (child step -> parent step):")
    for t in transition_stats:
        print(
            f"{t['from_step']}->{t['to_step']} "
            f"avg_purity={t['avg_child_purity']:.4f} "
            f"NMI={t['nmi_parent_child']:.4f} ARI={t['ari_parent_child']:.4f}"
        )

    out = {
        "meta": {
            "latents_path": args.latents,
            "latent_key": args.latent_key,
            "num_samples": int(n),
            "num_steps": int(num_steps),
            "hidden_dim": int(d),
            "schedule": schedule,
            "seed": int(args.seed),
        },
        "per_step_metrics": per_step_metrics,
        "transition_stats": transition_stats,
        "tree_edges": tree_edges,
        "assignments": [a.tolist() for a in assignments],
    }

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved tree metrics to {args.save_json}")


if __name__ == "__main__":
    main()