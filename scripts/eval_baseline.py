"""
Baseline evaluation: run CircuitNet's pretrained GPDL model (checkpoints/
congestion.pth) on packaged CircuitNet congestion samples and report
NRMSE / SSIM.

This establishes the "no fine-tuning, off-the-shelf CircuitNet weights"
baseline number that the closed-loop, per-design fine-tuned GNN needs to beat.
It does NOT train anything -- it's pure inference with pretrained weights.

Expects data laid out the way CircuitNet's own
preprocess_scripts/generate_training_set.py produces it:
    <root>/feature/<sample_id>.npy   (H, W, 3) -- macro_region, RUDY, RUDY_pin
    <root>/label/<sample_id>.npy     (H, W, 1) -- combined GR overflow congestion
already resized to 256x256 and min-max normalized per channel; no further
preprocessing needed here.

Usage:
    python scripts/eval_baseline.py --root data/circuitnet_raw/congestion
    python scripts/eval_baseline.py --root data/circuitnet_raw/congestion --num-samples 20 --save-heatmaps
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gnn.gpdl import load_pretrained_gpdl


def load_sample(root, sample_id):
    feature = np.load(os.path.join(root, "feature", sample_id))  # (H, W, 3)
    label = np.load(os.path.join(root, "label", sample_id))  # (H, W, 1)
    x = torch.tensor(feature.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  # (1, 3, H, W)
    y = torch.tensor(label.squeeze(-1), dtype=torch.float32)  # (H, W)
    return x, y


def nrmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    denom = target.max() - target.min()
    return (rmse / denom.clamp(min=1e-9)).item()


def ssim(pred, target, data_range=1.0, c1=(0.01) ** 2, c2=(0.03) ** 2):
    """Single-scale SSIM over the whole map (no windowing) -- adequate for a
    quick baseline comparison; swap for skimage.metrics.structural_similarity
    if a windowed score is needed."""
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
    parser.add_argument("--root", default="data/circuitnet_raw/congestion",
                         help="dir containing feature/ and label/ subfolders (generate_training_set.py output)")
    parser.add_argument("--checkpoint", default="checkpoints/congestion.pth")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--out", default="results/baseline")
    parser.add_argument("--save-heatmaps", action="store_true")
    args = parser.parse_args()

    feature_dir = os.path.join(args.root, "feature")
    label_dir = os.path.join(args.root, "label")
    if not os.path.isdir(feature_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"Expected {feature_dir} and {label_dir} to exist. This baseline needs "
            f"CircuitNet's packaged congestion samples (feature/<id>.npy + label/<id>.npy), "
            f"as produced by generate_training_set.py --task congestion."
        )

    sample_ids = sorted(os.listdir(feature_dir))
    if args.num_samples > 0:
        sample_ids = sample_ids[: args.num_samples]
    if not sample_ids:
        raise FileNotFoundError(f"No samples found under {feature_dir}")

    device = "cpu"
    model = load_pretrained_gpdl(args.checkpoint, device=device)
    print(f"Loaded pretrained GPDL from {args.checkpoint}")

    os.makedirs(args.out, exist_ok=True)
    if args.save_heatmaps:
        import matplotlib.pyplot as plt

    per_sample = []
    for sample_id in sample_ids:
        label_path = os.path.join(label_dir, sample_id)
        if not os.path.exists(label_path):
            print(f"[skip] {sample_id}: no matching label file at {label_path}")
            continue

        x, y = load_sample(args.root, sample_id)

        with torch.no_grad():
            pred = model(x).squeeze(0).squeeze(0)  # (H, W)

        data_range = (y.max() - y.min()).clamp(min=1e-9).item()
        score = {
            "sample": sample_id,
            "nrmse": nrmse(pred, y),
            "ssim": ssim(pred, y, data_range=data_range),
            "mse": torch.mean((pred - y) ** 2).item(),
        }
        per_sample.append(score)
        print(f"{sample_id:30s} nrmse={score['nrmse']:.4f}  ssim={score['ssim']:.4f}  mse={score['mse']:.6f}")

        if args.save_heatmaps:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(y.numpy(), cmap="viridis")
            axes[0].set_title("ground truth")
            axes[1].imshow(pred.numpy(), cmap="viridis")
            axes[1].set_title("GPDL prediction")
            for ax in axes:
                ax.axis("off")
            fig.savefig(os.path.join(args.out, f"{os.path.splitext(sample_id)[0]}.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "num_samples": len(per_sample),
        "mean_nrmse": float(np.mean([s["nrmse"] for s in per_sample])),
        "mean_ssim": float(np.mean([s["ssim"] for s in per_sample])),
        "mean_mse": float(np.mean([s["mse"] for s in per_sample])),
        "per_sample": per_sample,
    }
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Baseline summary (pretrained GPDL, no fine-tuning) ===")
    print(f"samples: {summary['num_samples']}")
    print(f"mean NRMSE: {summary['mean_nrmse']:.4f}  (lower is better)")
    print(f"mean SSIM:  {summary['mean_ssim']:.4f}  (higher is better)")
    print(f"mean MSE:   {summary['mean_mse']:.6f}")
    print(f"Results written to {args.out}/baseline_metrics.json")


if __name__ == "__main__":
    main()
