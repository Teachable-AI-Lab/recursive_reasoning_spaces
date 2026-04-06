import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Mermaid tree diagram from CIFAR-10 latent tree JSON.")
    parser.add_argument("--tree-json", type=str, required=True, help="Path to JSON produced by cluster_cifar10_latent_tree.py")
    parser.add_argument("--output", type=str, default="latents/cifar10_tree.mmd", help="Output Mermaid file")
    parser.add_argument("--min-overlap", type=int, default=1, help="Hide edges with overlap below this value")
    parser.add_argument("--min-purity", type=float, default=0.0, help="Hide edges with purity below this value")
    return parser.parse_args()


def node_id(step: int, cluster: int) -> str:
    return f"s{step}_c{cluster}"


def main():
    args = parse_args()

    with open(args.tree_json, "r") as f:
        data = json.load(f)

    schedule = data["meta"]["schedule"]
    edges = data["tree_edges"]
    assignments = data.get("assignments", None)

    node_counts = {}
    if assignments is not None:
        for step_i, step_assign in enumerate(assignments, start=1):
            counts = {}
            for c in step_assign:
                counts[c] = counts.get(c, 0) + 1
            node_counts[step_i] = counts

    lines = ["flowchart LR"]

    for step, k in enumerate(schedule, start=1):
        lines.append(f"  subgraph Step_{step}[Step {step}]")
        for c in range(k):
            count = node_counts.get(step, {}).get(c, 0)
            lines.append(f"    {node_id(step, c)}[\"S{step}:C{c} n={count}\"]")
        lines.append("  end")

    kept = 0
    for e in edges:
        if e["overlap"] < args.min_overlap:
            continue
        if e["purity"] < args.min_purity:
            continue

        src = node_id(e["from_step"], e["from_cluster"])
        dst = node_id(e["to_step"], e["to_cluster"])
        lbl = f"o={e['overlap']} p={e['purity']:.2f}"
        lines.append(f"  {src} -- \"{lbl}\" --> {dst}")
        kept += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved Mermaid tree to {args.output}")
    print(f"Rendered edges: {kept}/{len(edges)} (filters: min_overlap={args.min_overlap}, min_purity={args.min_purity})")


if __name__ == "__main__":
    main()
