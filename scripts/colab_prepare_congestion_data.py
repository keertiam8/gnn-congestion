"""
One-shot Colab script: download a handful of CircuitNet congestion samples,
pack them into GPDL's training-set format, and run the pretrained-weight
baseline evaluation against them -- all in one go.

Run this from inside a clone of the gnn-congestion repo (so it has access to
scripts/eval_baseline.py, gnn/gpdl.py, and checkpoints/congestion.pth):

    !git clone https://github.com/keertiam8/gnn-congestion.git
    %cd gnn-congestion
    !python scripts/colab_prepare_congestion_data.py --num-samples 20

Only ONE archive is ever resident on disk at a time -- peak usage stays
around one archive's size instead of the sum of all three.

Output:
    data/circuitnet_raw/congestion/feature/<sample_id>.npy   (256, 256, 2)
    data/circuitnet_raw/congestion/label/<sample_id>.npy     (256, 256, 1)
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# CONFIG
# 2 input features only: macro_region + RUDY (rudy_pin removed)
# ---------------------------------------------------------------------------
ARCHIVES = {
    "macro_region": {
        "source": "14n9khpSK56NUZrGUNPPRYmZkAbCOstKq",
        "keywords": ["macro_region/"],
    },
    "rudy": {
        "source": "1KUocSofLvyAFKXu8AXt4j3TPJsiaCjS6",
        "keywords": ["RUDY/"],
    },
    "congestion": {
        "source": "1aFFT6A32ybvKsTcFZGEQuARa8ipurke_",
        "keywords": [
            "congestion_GR_horizontal_overflow",
            "congestion_GR_vertical_overflow",
        ],
    },
}

# macro_region is smallest (~6MB vs 2.73GB for rudy) so use it to pick sample IDs
ID_SOURCE_KEY = "macro_region"

DOWNLOAD_DIR = "/content/circuitnet_downloads"
EXTRACT_DIR  = "/content/circuitnet_downloads/extracted"
PACKED_DIR   = "data/circuitnet_raw"


def is_url(source):
    return source.startswith("http://") or source.startswith("https://")


def is_drive_id(source):
    return not is_url(source) and not os.path.exists(source) and "/" not in source


def download_archive(source, dest_path, retries=2):
    for attempt in range(1, retries + 2):
        if is_drive_id(source):
            import gdown
            gdown.download(id=source, output=dest_path, quiet=False)
        elif is_url(source):
            import requests
            with requests.get(source, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
        else:
            if os.path.abspath(source) != os.path.abspath(dest_path):
                shutil.copy(source, dest_path)

        ok, reason = _looks_like_valid_targz(dest_path)
        if ok:
            return dest_path

        print(f"[attempt {attempt}] download looks bad ({reason}), retrying..."
              if attempt <= retries
              else f"download still bad after {retries} retries ({reason})")

    with open(dest_path, "rb") as f:
        preview = f.read(300)
    raise RuntimeError(
        f"{dest_path} failed to download as a valid tar.gz after {retries + 1} attempts "
        f"({reason}).\nFirst 300 bytes: {preview}"
    )


def _strip_gzip_layers_to_disk(path, max_layers=5):
    """Peel off nested gzip layers to a real temp file on disk."""
    current = path
    for layer_num in range(max_layers):
        with open(current, "rb") as f:
            magic = f.peek(2)[:2]
        if magic != b"\x1f\x8b":
            return current

        next_path = f"{path}.layer{layer_num}"
        with gzip.open(current, "rb") as src, open(next_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        if current != path:
            os.remove(current)
        current = next_path

    raise RuntimeError(f"{path}: still gzip-compressed after {max_layers} layers")


def _open_tar(path):
    return tarfile.open(_strip_gzip_layers_to_disk(path), mode="r:")


def _looks_like_valid_targz(path):
    try:
        with _open_tar(path) as tar:
            tar.getmembers()
        return True, None
    except Exception as e:
        return False, str(e)


def extract_matching(tar_path, keywords, sample_ids, extract_dir):
    extracted = 0
    with _open_tar(tar_path) as tar:
        for member in tar.getmembers():
            if any(kw in member.name for kw in keywords) and (
                sample_ids is None or os.path.basename(member.name) in sample_ids
            ):
                tar.extract(member, extract_dir)
                extracted += 1
    return extracted


def collect_sample_ids(tar_path, keywords, num_samples):
    with _open_tar(tar_path) as tar:
        names = [n for n in tar.getnames() if keywords[0] in n]
        names = [n for n in names if not tar.getmember(n).isdir()]
    return [os.path.basename(n) for n in names[:num_samples]]


def resize(a):
    return ndimage.zoom(a, (256 / a.shape[0], 256 / a.shape[1]), order=3)


def std(a):
    return a if a.max() == 0 else (a - a.min()) / (a.max() - a.min())


def pack_congestion(extract_dir, packed_dir, sample_ids):
    """
    Pack 2-channel features (macro_region + RUDY) and 1-channel labels
    into the format expected by preprocess_circuitnet.py.

    feature shape: (256, 256, 2)  ← macro_region + RUDY
    label shape:   (256, 256, 1)  ← horizontal + vertical overflow combined
    """
    feature_dir = os.path.join(packed_dir, "congestion", "feature")
    label_dir   = os.path.join(packed_dir, "congestion", "label")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir,   exist_ok=True)

    num_ok, num_fail = 0, 0

    for sid in sample_ids:
        try:
            # Load 2 input features (no rudy_pin)
            macro = np.load(os.path.join(extract_dir, "macro_region", sid))
            rudy  = np.load(os.path.join(extract_dir, "RUDY", sid))

            # Load 2 label channels
            h = np.load(os.path.join(
                extract_dir, "overflow_based",
                "congestion_GR_horizontal_overflow", sid,
            ))
            v = np.load(os.path.join(
                extract_dir, "overflow_based",
                "congestion_GR_vertical_overflow", sid,
            ))

        except FileNotFoundError as e:
            print(f"[skip] {sid}: {e}")
            num_fail += 1
            continue

        # Stack 2 features → (256, 256, 2)
        feature = np.stack(
            [std(resize(macro)), std(resize(rudy))],
            axis=-1
        )

        # Combine h + v overflow → (256, 256, 1)
        label = std(resize(h) + resize(v))[..., None]

        np.save(os.path.join(feature_dir, sid), feature.astype(np.float32))
        np.save(os.path.join(label_dir,   sid), label.astype(np.float32))
        num_ok += 1

    print(f"Packed {num_ok} samples ({num_fail} skipped) into {packed_dir}/congestion")
    return num_ok


def run_baseline_eval(congestion_dir, checkpoint="checkpoints/congestion.pth"):
    if not os.path.exists("scripts/eval_baseline.py"):
        print("\n[skipped baseline eval] scripts/eval_baseline.py not found.")
        return
    if not os.path.exists(checkpoint):
        print(f"\n[skipped baseline eval] {checkpoint} not found.")
        return

    print("\n=== Running pretrained-weight baseline evaluation ===")
    subprocess.run(
        [sys.executable, "scripts/eval_baseline.py",
         "--root", congestion_dir,
         "--checkpoint", checkpoint,
         "--save-heatmaps"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--keep-extracted", action="store_true",
                        help=f"don't delete {EXTRACT_DIR} afterward")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip running eval_baseline.py at the end")
    parser.add_argument("--checkpoint", default="checkpoints/congestion.pth")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    sample_ids = None

    for name, cfg in ARCHIVES.items():
        if cfg["source"] == "FILL_ME_IN":
            raise ValueError(
                f"Set ARCHIVES['{name}']['source'] before running this script."
            )

        tar_path = f"{DOWNLOAD_DIR}/{name}.tar.gz"
        print(f"\n=== {name} ===")
        print("downloading...")
        download_archive(cfg["source"], tar_path)

        if sample_ids is None and name == ID_SOURCE_KEY:
            sample_ids = collect_sample_ids(
                tar_path, cfg["keywords"], args.num_samples
            )
            print(f"picked {len(sample_ids)} sample IDs from {name}")

        if sample_ids is None:
            raise RuntimeError(
                f"ID_SOURCE_KEY='{ID_SOURCE_KEY}' must be processed before "
                f"archives that depend on its sample_ids."
            )

        n = extract_matching(tar_path, cfg["keywords"], sample_ids, EXTRACT_DIR)
        print(f"extracted {n} files from {name}")

        os.remove(tar_path)
        print(f"deleted {tar_path} to free space")

    pack_congestion(EXTRACT_DIR, PACKED_DIR, sample_ids)

    if not args.keep_extracted:
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        print("cleaned up raw extracted files (kept only the packed feature/label npy)")

    congestion_dir = os.path.join(PACKED_DIR, "congestion")
    print(f"\nData ready at {congestion_dir}")

    if not args.no_eval:
        run_baseline_eval(congestion_dir, checkpoint=args.checkpoint)
        print(
            "\nDone. Baseline results at results/baseline/baseline_metrics.json "
            "and heatmap PNGs at results/baseline/*.png"
        )
    else:
        print("\nDone (baseline eval skipped, --no-eval was set).")


if __name__ == "__main__":
    main()