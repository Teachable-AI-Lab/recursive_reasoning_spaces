import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent hierarchy with CIFAR-10 image overlays per stage.")
    parser.add_argument("--tree-json", type=str, required=True, help="Path to JSON from cluster_cifar10_latent_tree.py")
    parser.add_argument("--latents", type=str, required=True, help="Path to .npz from extract_cifar10_latents.py")
    parser.add_argument("--latent-key", type=str, default="z_h_pool", choices=["z_h_pool", "z_h_cls", "z_l_pool", "z_l_cls"], help="Latent key used for representative selection")
    parser.add_argument("--output", type=str, default="latents/cifar10_tree_images.png", help="Output PNG path")
    parser.add_argument("--samples-per-cluster", type=int, default=9, help="Number of sample images to show per cluster for non-final stages")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling images")
    parser.add_argument("--min-edge-overlap", type=int, default=1, help="Hide connector edges below this overlap")
    parser.add_argument("--min-edge-purity", type=float, default=0.0, help="Hide connector edges below this purity")
    return parser.parse_args()


def _pick_representative(latents_step: np.ndarray, idx: np.ndarray) -> int:
    x = latents_step[idx]
    center = x.mean(axis=0, keepdims=True)
    dist = np.sum((x - center) ** 2, axis=1)
    return int(idx[np.argmin(dist)])


def _majority_label(labels: np.ndarray) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def _to_rgb_image(arr: np.ndarray, tile: int) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB").resize((tile, tile), resample=Image.Resampling.NEAREST)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB").resize((tile, tile), resample=Image.Resampling.NEAREST)


def _draw_image_grid(
    draw_img: Image.Image,
    images: np.ndarray,
    indices: np.ndarray,
    x0: int,
    y0: int,
    tile: int,
    grid_side: int,
    padding: int,
    rng: np.random.Generator,
):
    draw = ImageDraw.Draw(draw_img)

    if len(indices) == 0:
        draw.rectangle([x0, y0, x0 + grid_side * tile + (grid_side - 1) * padding, y0 + grid_side * tile + (grid_side - 1) * padding], outline=(120, 120, 120), width=1)
        draw.text((x0 + 4, y0 + 4), "empty", fill=(120, 120, 120))
        return

    n_show = min(len(indices), grid_side * grid_side)
    pick = rng.choice(indices, size=n_show, replace=False)

    for i, idx in enumerate(pick):
        r = i // grid_side
        c = i % grid_side
        xi = x0 + c * (tile + padding)
        yi = y0 + r * (tile + padding)

        im = _to_rgb_image(images[idx], tile)
        draw_img.paste(im, (xi, yi))


def _build_edge_map(tree_edges, min_overlap: int, min_purity: float):
    edge_map = {}
    for e in tree_edges:
        if e["overlap"] < min_overlap or e["purity"] < min_purity:
            continue
        edge_map[(e["from_step"], e["from_cluster"])] = e
    return edge_map


def _compute_step_orders(schedule, assignments, edge_map):
    orders = []

    first = np.array(assignments[0], dtype=np.int32)
    first_counts = [(c, int((first == c).sum())) for c in range(schedule[0])]
    first_counts.sort(key=lambda kv: (-kv[1], kv[0]))
    orders.append([c for c, _n in first_counts])

    for step_idx in range(1, len(schedule)):
        step = step_idx + 1
        assign = np.array(assignments[step_idx], dtype=np.int32)
        prev_order = orders[step_idx - 1]
        prev_pos = {c: i for i, c in enumerate(prev_order)}

        children = []
        for c in range(schedule[step_idx]):
            n = int((assign == c).sum())
            e = edge_map.get((step, c), None)
            if e is None:
                parent = -1
                overlap = 0
            else:
                parent = int(e["to_cluster"])
                overlap = int(e["overlap"])

            parent_rank = prev_pos.get(parent, 10**9)
            children.append((c, n, parent_rank, overlap))

        children.sort(key=lambda x: (x[2], -x[3], -x[1], x[0]))
        orders.append([c for c, _n, _p, _o in children])

    return orders


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.tree_json, "r") as f:
        tree = json.load(f)

    npz = np.load(args.latents)

    schedule = tree["meta"]["schedule"]
    assignments = tree["assignments"]
    labels = npz["label"]
    images = npz["image"]
    latents = npz[args.latent_key]

    num_steps = len(schedule)
    max_k = max(schedule)

    if len(assignments) != num_steps:
        raise ValueError("Assignments length does not match schedule length")

    edge_map = _build_edge_map(tree.get("tree_edges", []), args.min_edge_overlap, args.min_edge_purity)
    step_orders = _compute_step_orders(schedule, assignments, edge_map)

    title_h = 56
    row_header_w = 128
    cell_w = 136
    cell_h = 154
    gap_x = 10
    gap_y = 24

    width = row_header_w + max_k * (cell_w + gap_x) + 24
    height = title_h + num_steps * (cell_h + gap_y) + 24

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((12, 12), "CIFAR-10 Latent Hierarchy by Recursion Step", fill=(20, 20, 20))
    draw.text((12, 32), f"Sorted by parent links; edges filtered by overlap>={args.min_edge_overlap}, purity>={args.min_edge_purity}", fill=(70, 70, 70))

    box_pos = {}
    y_bases = []
    for step_idx in range(num_steps):
        step = step_idx + 1
        k = schedule[step_idx]
        y_base = title_h + step_idx * (cell_h + gap_y)
        y_bases.append(y_base)

        left_pad = ((max_k - k) * (cell_w + gap_x)) // 2
        x_start = row_header_w + left_pad

        for col, c in enumerate(step_orders[step_idx]):
            x_base = x_start + col * (cell_w + gap_x)
            box_pos[(step, c)] = (x_base, y_base, x_base + cell_w, y_base + cell_h)

    for (from_step, from_cluster), e in edge_map.items():
        to_step = int(e["to_step"])
        to_cluster = int(e["to_cluster"])

        src_box = box_pos.get((from_step, from_cluster))
        dst_box = box_pos.get((to_step, to_cluster))
        if src_box is None or dst_box is None:
            continue

        sx = (src_box[0] + src_box[2]) // 2
        sy = src_box[1]
        tx = (dst_box[0] + dst_box[2]) // 2
        ty = dst_box[3]

        mid_y = (sy + ty) // 2
        purity = float(e["purity"])
        overlap = int(e["overlap"])
        width_px = max(1, min(7, int(round(overlap / 20))))

        r = int(220 * (1.0 - purity) + 40 * purity)
        g = int(120 * (1.0 - purity) + 170 * purity)
        b = int(60 * (1.0 - purity) + 80 * purity)
        color = (r, g, b)

        draw.line([(sx, sy), (sx, mid_y), (tx, mid_y), (tx, ty)], fill=color, width=width_px)

    for step_idx in range(num_steps):
        step = step_idx + 1
        k = schedule[step_idx]
        assign = np.array(assignments[step_idx], dtype=np.int32)

        y_base = y_bases[step_idx]
        draw.text((12, y_base + 4), f"Step {step} (k={k})", fill=(30, 30, 30))

        for c in step_orders[step_idx]:
            x0, y0, x1, y1 = box_pos[(step, c)]
            idx = np.where(assign == c)[0]
            n = len(idx)

            draw.rectangle([x0, y0, x1, y1], outline=(140, 140, 140), width=2)
            draw.text((x0 + 4, y0 + 4), f"C{c} n={n}", fill=(20, 20, 20))

            if step < num_steps:
                grid_side = int(np.ceil(np.sqrt(max(1, args.samples_per_cluster))))
                tile = 30
                padding = 2
                _draw_image_grid(
                    draw_img=canvas,
                    images=images,
                    indices=idx,
                    x0=x0 + 4,
                    y0=y0 + 24,
                    tile=tile,
                    grid_side=grid_side,
                    padding=padding,
                    rng=rng,
                )
            else:
                if n > 0:
                    rep_idx = _pick_representative(latents[:, step_idx, :], idx)
                    maj = _majority_label(labels[idx])
                    rep = _to_rgb_image(images[rep_idx], 96)
                    canvas.paste(rep, (x0 + 20, y0 + 32))
                    class_name = CIFAR10_CLASS_NAMES[maj] if 0 <= maj < len(CIFAR10_CLASS_NAMES) else "unknown"
                    draw.text((x0 + 4, y0 + 134), f"class~{maj} {class_name}", fill=(20, 20, 20))
                else:
                    draw.text((x0 + 40, y0 + 70), "empty", fill=(120, 120, 120))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    canvas.save(args.output)
    print(f"Saved image hierarchy visualization to {args.output}")


if __name__ == "__main__":
    main()
