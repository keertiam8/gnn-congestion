"""
Run inference on GCD design using pretrained netlist-GNN.
Compares GNN predictions vs real OpenROAD congestion.

Usage:
    python scripts/infer_gcd.py \
        --graph data/gcd_graph.pt \
        --checkpoint checkpoints/pretrained_netlist.pt \
        --out results/gcd_inference
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gnn.model import CongestionGNN, nodes_to_grid_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', default='data/gcd_graph.pt')
    parser.add_argument('--checkpoint', default='checkpoints/pretrained_netlist.pt')
    parser.add_argument('--out', default='results/gcd_inference')
    parser.add_argument('--grid-size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load graph
    gcd = torch.load(args.graph, weights_only=False)
    print(f"GCD graph loaded:")
    print(f"  Total nodes:  {gcd.num_nodes}")
    print(f"  Real cells:   {gcd.cell_mask.sum().item()}")
    print(f"  Edges:        {gcd.edge_index.shape[1]}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CongestionGNN(in_channels=4, out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # Run inference
    gcd = gcd.to(device)
    with torch.no_grad():
        pred_all = model(gcd.x, gcd.edge_index)
        pred_cells = pred_all[gcd.cell_mask]
        xy_cells = gcd.x[gcd.cell_mask][:, :2]

    print(f"\nGCD congestion predictions (real cells only):")
    print(f"  Mean:  {pred_cells.mean():.4f}")
    print(f"  Std:   {pred_cells.std():.4f}")
    print(f"  Min:   {pred_cells.min():.4f}")
    print(f"  Max:   {pred_cells.max():.4f}")

    # Rasterize to grid heatmap
    grid = (args.grid_size, args.grid_size)
    pred_grid = nodes_to_grid_heatmap(pred_cells, xy_cells, grid_size=grid)
    pred_grid_np = pred_grid.cpu().numpy()

    print(f"\nPredicted congestion heatmap:")
    print(f"  Shape: {pred_grid_np.shape}")
    print(f"  Mean:  {pred_grid_np.mean():.4f}")
    print(f"  Max:   {pred_grid_np.max():.4f}")

    # Save heatmap as image
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_grid_np, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Predicted Congestion')
    plt.title('GCD Design - NetlistGNN Predicted Congestion')
    plt.xlabel('X (grid)')
    plt.ylabel('Y (grid)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'gcd_congestion_heatmap.png'), dpi=150)
    plt.close()
    print(f"\nHeatmap saved to {args.out}/gcd_congestion_heatmap.png")

    # Save raw predictions
    results = {
        'num_cells': int(gcd.cell_mask.sum().item()),
        'pred_mean': float(pred_cells.mean()),
        'pred_std': float(pred_cells.std()),
        'pred_min': float(pred_cells.min()),
        'pred_max': float(pred_cells.max()),
    }
    with open(os.path.join(args.out, 'gcd_inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out}/gcd_inference_results.json")


if __name__ == '__main__':
    main()