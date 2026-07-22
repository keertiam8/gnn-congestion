"""
Convert CircuitNet congestion-subset grid maps into PyG graph Data objects.

Each sample's H x W feature maps become a grid graph:
    - one node per grid tile (x, y)
    - node features = [x_norm, y_norm, macro_region, RUDY, RUDY_pin, ...]
    - edges = 4-neighbor grid adjacency (bidirectional)
    - label (y) = congestion_label value at that tile

IMPORTANT: file names below (feature_names / label_name) are best guesses
based on CircuitNet's published congestion task. Run inspect_data.py first
and update FEATURE_FILES / LABEL_FILE to match what you actually see.

Usage:
    python scripts/preprocess_circuitnet.py \
        --root data/circuitnet_raw \
        --out data/circuitnet_graphs \
        --limit 0   # 0 = process all samples
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data

# Update these after running inspect_data.py
FEATURE_FILES = ["macro_region.npy", "RUDY.npy", "RUDY_pin.npy"]
LABEL_FILE = "congestion_label.npy"


def build_grid_graph(feature_maps, label_map):
    """
    feature_maps: list of (H, W) arrays
    label_map: (H, W) array
    """
    H, W = label_map.shape
    num_nodes = H * W

    # node coords, normalized to [0, 1]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_norm = (xs / max(W - 1, 1)).astype(np.float32).reshape(-1)
    y_norm = (ys / max(H - 1, 1)).astype(np.float32).reshape(-1)

    feat_flat = [x_norm, y_norm] + [f.astype(np.float32).reshape(-1) for f in feature_maps]
    node_features = np.stack(feat_flat, axis=1)  # (num_nodes, 2 + num_feature_maps)

    labels = label_map.astype(np.float32).reshape(-1)

    # 4-neighbor grid adjacency
    def node_id(r, c):
        return r * W + c

    src, dst = [], []
    for r in range(H):
        for c in range(W):
            nid = node_id(r, c)
            if c + 1 < W:
                src += [nid, node_id(r, c + 1)]
                dst += [node_id(r, c + 1), nid]
            if r + 1 < H:
                src += [nid, node_id(r + 1, c)]
                dst += [node_id(r + 1, c), nid]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, y=y, grid_shape=(H, W))


def process_sample(sample_dir):
    feature_maps = []
    for fname in FEATURE_FILES:
        path = os.path.join(sample_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing -- check FEATURE_FILES matches your data")
        feature_maps.append(np.load(path))

    label_path = os.path.join(sample_dir, LABEL_FILE)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path} missing -- check LABEL_FILE matches your data")
    label_map = np.load(label_path)

    return build_grid_graph(feature_maps, label_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sample_dirs = sorted(
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    )
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    num_ok, num_fail = 0, 0
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        try:
            data = process_sample(sample_dir)
        except FileNotFoundError as e:
            print(f"[skip] {sample_id}: {e}")
            num_fail += 1
            continue

        torch.save(data, os.path.join(args.out, f"{sample_id}.pt"))
        num_ok += 1

    print(f"Done. {num_ok} graphs written to {args.out}, {num_fail} skipped.")


if __name__ == "__main__":
    main()
