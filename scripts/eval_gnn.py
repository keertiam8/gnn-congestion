"""
Evaluate CongestionGNN against the same NRMSE/SSIM metrics eval_baseline.py
uses for the pretrained GPDL model, on the same preprocessed samples.

This is the real number to compare against results/baseline/baseline_metrics.json
-- not std/mean proxy stats. Your GNN needs to beat GPDL's mean_nrmse/mean_ssim
before it's a stronger foundation than the off-the-shelf baseline.

Usage:
    python scripts/eval_gnn.py \
        --graphs /content/circuitnet_graphs \
        --checkpoint checkpoints/pretrained.pt
"""

import argparse
import json
import glob
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gnn.model import CongestionGNN


def nrmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    denom = target.max() - target.min()
    return (rmse / denom.clamp(min=1e-9)).item()


def ssim(pred, target, data_range=1.0, c1=(0.01) ** 2, c2=(0.03) ** 2):
    pred, target = pred.double(), target.double()
    mu_p, mu_t = pred.mean(), target.mean()
    var_p, var_t = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - mu_p) * (target - mu_t)).mean()
    c1, c2 = (c1 * data_range ** 2), (c2 * data_range ** 2)
    num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    den = (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
    return (num / den).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", required=True, help="dir of preprocessed .pt graphs from preprocess_circuitnet.py")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--out", default="results/gnn_eval")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.graphs, "*.pt")))
    if args.num_samples > 0:
        paths = paths[: args.num_samples]
    if not paths:
        raise FileNotFoundError(f"No .pt graphs found in {args.graphs}")

    sample0 = torch.load(paths[0], weights_only=False)
    in_channels = sample0.x.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CongestionGNN(in_channels=in_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"Loaded GNN checkpoint from {args.checkpoint}")

    os.makedirs(args.out, exist_ok=True)
    per_sample = []

    with torch.no_grad():
        for path in paths:
            data = torch.load(path, weights_only=False).to(device)
            pred_nodes = model(data.x, data.edge_index)

            H, W = data.grid_shape
            pred = pred_nodes.view(H, W)
            target = data.y.view(H, W)

            data_range = (target.max() - target.min()).clamp(min=1e-9).item()
            score = {
                "sample": os.path.basename(path),
                "nrmse": nrmse(pred, target),
                "ssim": ssim(pred, target, data_range=data_range),
                "mse": torch.mean((pred - target) ** 2).item(),
            }
            per_sample.append(score)
            print(f"{score['sample']:30s} nrmse={score['nrmse']:.4f}  ssim={score['ssim']:.4f}  mse={score['mse']:.6f}")

    summary = {
        "num_samples": len(per_sample),
        "mean_nrmse": float(np.mean([s["nrmse"] for s in per_sample])),
        "mean_ssim": float(np.mean([s["ssim"] for s in per_sample])),
        "mean_mse": float(np.mean([s["mse"] for s in per_sample])),
        "per_sample": per_sample,
    }
    with open(os.path.join(args.out, "gnn_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GNN summary (pretrained, no per-design fine-tuning) ===")
    print(f"samples: {summary['num_samples']}")
    print(f"mean NRMSE: {summary['mean_nrmse']:.4f}  (lower is better)")
    print(f"mean SSIM:  {summary['mean_ssim']:.4f}  (higher is better)")
    print(f"mean MSE:   {summary['mean_mse']:.6f}")
    print(f"Results written to {args.out}/gnn_metrics.json")
    print("\nCompare against results/baseline/baseline_metrics.json (GPDL baseline).")


if __name__ == "__main__":
    main()
