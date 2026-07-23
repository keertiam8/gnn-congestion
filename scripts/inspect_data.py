"""
Quick inspection of packaged CircuitNet congestion-subset files.

CircuitNet's own preprocess_scripts/generate_training_set.py --task congestion
packages samples as:
    <root>/feature/<sample_id>.npy   (H, W, 3) -- macro_region, RUDY, RUDY_pin
    <root>/label/<sample_id>.npy     (H, W, 1) -- combined GR overflow congestion
already resized to 256x256 and min-max normalized per channel.

Run this after generating/downloading the packaged training set to confirm
shapes and value ranges before running preprocess_circuitnet.py or
eval_baseline.py.

Usage:
    python scripts/inspect_data.py --root data/circuitnet_raw/congestion
"""

import argparse
import os
import numpy as np


def inspect_pair(root, sample_id):
    print(f"\n=== {sample_id} ===")
    for subdir in ("feature", "label"):
        path = os.path.join(root, subdir, sample_id)
        if not os.path.exists(path):
            print(f"  {subdir}: MISSING ({path})")
            continue
        arr = np.load(path)
        print(
            f"  {subdir:10s} shape={str(arr.shape):15s} "
            f"dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dir containing feature/ and label/ subfolders")
    parser.add_argument("--num-samples", type=int, default=3, help="how many samples to inspect")
    args = parser.parse_args()

    feature_dir = os.path.join(args.root, "feature")
    label_dir = os.path.join(args.root, "label")

    if not os.path.isdir(feature_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"Expected {feature_dir} and {label_dir} to exist. Run "
            f"generate_training_set.py --task congestion first, or point --root "
            f"at wherever that output was copied to."
        )

    sample_ids = sorted(os.listdir(feature_dir))
    if not sample_ids:
        print(f"No files found under {feature_dir}.")
        return

    print(f"Found {len(sample_ids)} samples. Inspecting first {args.num_samples}...")
    for sample_id in sample_ids[: args.num_samples]:
        inspect_pair(args.root, sample_id)


if __name__ == "__main__":
    main()
