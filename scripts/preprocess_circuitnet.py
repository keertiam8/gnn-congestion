"""
Convert CircuitNet congestion-subset grid maps into PyG graph Data objects.

Each sample's H x W feature maps become a grid graph:
    - one node per grid tile (x, y)
    - node features = [x_norm, y_norm, macro_region, RUDY, RUDY_pin]
    - edges = 4-neighbor grid adjacency (bidirectional)
    - label (y) = congestion value at that tile

Expects CircuitNet's packaged congestion samples, as produced by CircuitNet's
own preprocess_scripts/generate_training_set.py --task congestion:
    <root>/feature/<sample_id>.npy   (H, W, 3) -- macro_region, RUDY, RUDY_pin
    <root>/label/<sample_id>.npy     (H, W, 1) -- combined GR overflow congestion
already resized to 256x256 and min-max normalized per channel.

Usage:
    python scripts/preprocess_circuitnet.py \
        --root data/circuitnet_raw/congestion \
        --out data/circuitnet_graphs \
        --limit 0   # 0 = process all samples
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data


def build_grid_graph(feature_map, label_map):
    """
    feature_map: (H, W, C) array
    label_map: (H, W) or (H, W, 1) array
    """
    H, W, _ = feature_map.shape
    label_map = label_map.reshape(H, W)

    # node coords, normalized to [0, 1]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_norm = (xs / max(W - 1, 1)).astype(np.float32).reshape(-1)
    y_norm = (ys / max(H - 1, 1)).astype(np.float32).reshape(-1)

    feat_channels = [feature_map[:, :, c].astype(np.float32).reshape(-1) for c in range(feature_map.shape[2])]
    feat_flat = [x_norm, y_norm] + feat_channels
    node_features = np.stack(feat_flat, axis=1)  # (num_nodes, 2 + num_channels)

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


def process_sample(root, sample_id):
    feature_path = os.path.join(root, "feature", sample_id)
    label_path = os.path.join(root, "label", sample_id)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path} missing -- no matching label for {sample_id}")

    feature_map = np.load(feature_path)
    label_map = np.load(label_path)

    return build_grid_graph(feature_map, label_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dir containing feature/ and label/ subfolders")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    feature_dir = os.path.join(args.root, "feature")
    sample_ids = sorted(os.listdir(feature_dir))
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    num_ok, num_fail = 0, 0
    for sample_id in sample_ids:
        try:
            data = process_sample(args.root, sample_id)
        except FileNotFoundError as e:
            print(f"[skip] {sample_id}: {e}")
            num_fail += 1
            continue

        out_name = os.path.splitext(sample_id)[0] + ".pt"
        torch.save(data, os.path.join(args.out, out_name))
        num_ok += 1

    print(f"Done. {num_ok} graphs written to {args.out}, {num_fail} skipped.")


if __name__ == "__main__":
    main()
