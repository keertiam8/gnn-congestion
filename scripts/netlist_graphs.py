"""
Build netlist-based graphs combining:
  - Topology (node_attr, pin_attr from graph_information.tar.gz -> cell-to-cell edges)
  - Positions (instance_placement_micron -> x,y per cell)
  - Spatial context (macro_region, RUDY grids -> sampled per cell)
  - Labels (congestion grid -> sampled per cell)

Usage:
    python scripts/build_netlist_graphs.py \
        --graph-info /content/graph_info_extracted \
        --placement /content/placement_micron/instance_placement_micron \
        --congestion-root data/circuitnet_raw/congestion \
        --out /content/circuitnet_netlist_graphs \
        --limit 500
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data


def load_netlist_topology(design_family, graph_info_dir):
    """Load node names/types and build cell-to-cell edges from pin connectivity."""
    node_attr = np.load(f"{graph_info_dir}/node_attr/{design_family}_node_attr.npy", allow_pickle=True)
    pin_attr = np.load(f"{graph_info_dir}/pin_attr/{design_family}_pin_attr.npy", allow_pickle=True)

    instance_names = node_attr[0]
    num_nodes = len(instance_names)

    net_indices = pin_attr[1].astype(int)
    node_indices = pin_attr[2].astype(int)

    net_to_nodes = defaultdict(set)
    for net_idx, node_idx in zip(net_indices, node_indices):
        net_to_nodes[net_idx].add(node_idx)

    src, dst = [], []
    for nodes in net_to_nodes.values():
        nodes = list(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                src += [nodes[i], nodes[j]]
                dst += [nodes[j], nodes[i]]

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return instance_names, edge_index, num_nodes


def sample_grid_at_position(grid, x, y, die_xmax, die_ymax):
    gx, gy = grid.shape[:2]
    col = int((x / (die_xmax + 1e-9)) * (gx - 1))
    row = int((y / (die_ymax + 1e-9)) * (gy - 1))
    col = max(0, min(gx - 1, col))
    row = max(0, min(gy - 1, row))
    return grid[row, col]


def design_family_from_sample_id(sample_id):
    """
    sample_id like '7393-zero-riscy-a-1-c5-u0.75-m1-p1-f0'
    design_family like 'zero-riscy-a-1-c5' (drop leading numeric id and
    trailing u/m/p/f variation tags).
    """
    parts = sample_id.split('-')
    # drop leading numeric id
    parts = parts[1:]
    # drop trailing tags starting with u/m/p/f followed by digits
    cut = len(parts)
    for i, p in enumerate(parts):
        if len(p) > 1 and p[0] in "ump" and p[1:].replace('.', '', 1).isdigit():
            cut = i
            break
        if p.startswith('f') and p[1:].isdigit():
            cut = min(cut, i)
    return '-'.join(parts[:cut])


def build_netlist_graph(sample_id, graph_info_dir, placement_dir, congestion_root):
    design_family = design_family_from_sample_id(sample_id)

    node_attr_path = f"{graph_info_dir}/node_attr/{design_family}_node_attr.npy"
    placement_path = f"{placement_dir}/{sample_id}.npy"
    feature_path = f"{congestion_root}/feature/{sample_id}.npy"
    label_path = f"{congestion_root}/label/{sample_id}.npy"

    for p in [node_attr_path, placement_path, feature_path, label_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")

    instance_names, edge_index, num_nodes = load_netlist_topology(design_family, graph_info_dir)
    placement = np.load(placement_path, allow_pickle=True).item()

    feature = np.load(feature_path)  # (H, W, 2) macro_region, RUDY
    label = np.load(label_path)      # (H, W, 1) congestion

    macro_grid = feature[:, :, 0]
    rudy_grid = feature[:, :, 1]
    congestion_grid = label[:, :, 0]

    die_xmax = max(v[2] for v in placement.values())
    die_ymax = max(v[3] for v in placement.values())

    x_coords, y_coords = [], []
    macro_ctx, rudy_ctx, congestion_labels = [], [], []

    for name in instance_names:
        if name in placement:
            xmin, ymin, xmax, ymax = placement[name]
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        else:
            cx, cy = 0.0, 0.0

        x_coords.append(cx)
        y_coords.append(cy)
        macro_ctx.append(sample_grid_at_position(macro_grid, cx, cy, die_xmax, die_ymax))
        rudy_ctx.append(sample_grid_at_position(rudy_grid, cx, cy, die_xmax, die_ymax))
        congestion_labels.append(sample_grid_at_position(congestion_grid, cx, cy, die_xmax, die_ymax))

    x_coords = np.array(x_coords, dtype=np.float32) / (die_xmax + 1e-9)
    y_coords = np.array(y_coords, dtype=np.float32) / (die_ymax + 1e-9)

    node_features = np.stack(
        [x_coords, y_coords, np.array(macro_ctx, dtype=np.float32), np.array(rudy_ctx, dtype=np.float32)],
        axis=1,
    )

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(congestion_labels, dtype=torch.float32),
        num_nodes=num_nodes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-info", required=True, help="dir with node_attr/, net_attr/, pin_attr/")
    parser.add_argument("--placement", required=True, help="dir with instance_placement_micron .npy files")
    parser.add_argument("--congestion-root", required=True, help="dir with feature/ and label/ subfolders")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    feature_dir = os.path.join(args.congestion_root, "feature")
    sample_ids = sorted(f.replace(".npy", "") for f in os.listdir(feature_dir))
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    num_ok, num_fail = 0, 0
    for sample_id in sample_ids:
        try:
            data = build_netlist_graph(sample_id, args.graph_info, args.placement, args.congestion_root)
        except FileNotFoundError as e:
            print(f"[skip] {sample_id}: {e}")
            num_fail += 1
            continue
        except Exception as e:
            print(f"[error] {sample_id}: {e}")
            num_fail += 1
            continue

        torch.save(data, os.path.join(args.out, f"{sample_id}.pt"))
        num_ok += 1
        if num_ok % 50 == 0:
            print(f"  ...built {num_ok} graphs so far")

    print(f"Done. {num_ok} netlist graphs written to {args.out}, {num_fail} skipped.")


if __name__ == "__main__":
    main()