"""
Build a netlist graph from extracted GCD data.
Output: PyG Data object in same format as CircuitNet graphs.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict

def build_gcd_graph(extracted_dir, output_path):
    """
    Build graph from extract_gcd_data.tcl outputs.
    
    Args:
        extracted_dir: directory with instances.csv, connectivity.csv
        output_path: where to save gcd_graph.pt
    """
    
    # Load instances (cells)
    instances_df = pd.read_csv(os.path.join(extracted_dir, 'instances.csv'))
    instance_names = instances_df['instance_name'].tolist()
    num_cells = len(instance_names)
    
    instance_to_idx = {name: i for i, name in enumerate(instance_names)}
    
    # Extract positions (center coordinates)
    x_coords = (instances_df['x_min'] + instances_df['x_max']) / 2
    y_coords = (instances_df['y_min'] + instances_df['y_max']) / 2
    
    # Normalize to [0, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_norm = (x_coords - x_min) / (x_max - x_min + 1e-9)
    y_norm = (y_coords - y_min) / (y_max - y_min + 1e-9)
    
    # Load connectivity (nets)
    conn_df = pd.read_csv(os.path.join(extracted_dir, 'connectivity.csv'))
    
    # Build star expansion: cells connect to virtual net nodes (matches CircuitNet structure)
    net_to_cells = defaultdict(set)
    for _, row in conn_df.iterrows():
        net_name = row['net_name']
        inst_name = row['instance_name']
        if inst_name in instance_to_idx:
            net_to_cells[net_name].add(instance_to_idx[inst_name])

    # Create net nodes with consistent indexing
    net_names = sorted(net_to_cells.keys())
    net_name_to_idx = {name: i for i, name in enumerate(net_names)}
    num_nets = len(net_names)
    total_nodes = num_cells + num_nets

    # Star edges: cells <-> net nodes
    src, dst = [], []
    for net_name, cells in net_to_cells.items():
        net_node_idx = num_cells + net_name_to_idx[net_name]
        for cell_idx in cells:
            src += [cell_idx, net_node_idx]
            dst += [net_node_idx, cell_idx]

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    # Node features: real cells [x, y, 0, 0], virtual nets [0, 0, 0, 0]
    cell_features = np.stack([
        x_norm.values,
        y_norm.values,
        np.zeros(num_cells),  # placeholder for macro_region
        np.zeros(num_cells),  # placeholder for RUDY
    ], axis=1).astype(np.float32)

    net_features = np.zeros((num_nets, 4), dtype=np.float32)
    all_features = np.concatenate([cell_features, net_features], axis=0)

    # Labels: zeros for all (we'll extract real congestion separately, only score real cells)
    all_labels = np.zeros(total_nodes, dtype=np.float32)

    # Mask to distinguish real cells from virtual net nodes
    is_real_cell = torch.zeros(total_nodes, dtype=torch.bool)
    is_real_cell[:num_cells] = True

    data = Data(
        x=torch.tensor(all_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(all_labels, dtype=torch.float32),
        cell_mask=is_real_cell,
        num_nodes=total_nodes,
        num_cells=num_cells,
    )
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved GCD graph to {output_path}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape}")
    
    return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python build_gcd_graph.py <extracted_dir> <output_path>")
        sys.exit(1)
    
    extracted_dir = sys.argv[1]
    output_path = sys.argv[2]
    build_gcd_graph(extracted_dir, output_path)