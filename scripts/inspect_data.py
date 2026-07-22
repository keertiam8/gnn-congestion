"""
Quick inspection of raw CircuitNet congestion-subset files.

CircuitNet (N28 congestion task) ships per-sample .npy feature maps, e.g.:
    <sample_id>/macro_region.npy
    <sample_id>/RUDY.npy
    <sample_id>/RUDY_pin.npy
    <sample_id>/congestion_label.npy   (or similar names -- verify below)

Run this after downloading + extracting CircuitNet to confirm actual file
names, shapes and value ranges before running preprocess_circuitnet.py.

Usage:
    python scripts/inspect_data.py --root data/circuitnet_raw
"""

import argparse
import os
import numpy as np


def inspect_sample(sample_dir):
    print(f"\n=== {sample_dir} ===")
    for fname in sorted(os.listdir(sample_dir)):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(sample_dir, fname)
        arr = np.load(path)
        print(
            f"  {fname:30s} shape={str(arr.shape):15s} "
            f"dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root dir containing CircuitNet samples")
    parser.add_argument("--num-samples", type=int, default=3, help="how many samples to inspect")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(
            f"{args.root} does not exist. Download CircuitNet first "
            f"(see scripts/README.md) and point --root at the extracted folder."
        )

    entries = sorted(os.listdir(args.root))
    sample_dirs = [os.path.join(args.root, e) for e in entries if os.path.isdir(os.path.join(args.root, e))]

    if not sample_dirs:
        # maybe .npy files are directly under root instead of per-sample subdirs
        print(f"No subdirectories found under {args.root}. Files directly present:")
        for fname in sorted(entries)[:20]:
            print(" ", fname)
        return

    print(f"Found {len(sample_dirs)} sample directories. Inspecting first {args.num_samples}...")
    for d in sample_dirs[: args.num_samples]:
        inspect_sample(d)


if __name__ == "__main__":
    main()
