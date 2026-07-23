"""
One-shot Colab script: download a handful of CircuitNet congestion samples,
pack them into GPDL's training-set format, and zip the result for download.

Fill in ARCHIVES below with your actual sources (Google Drive file IDs, direct
URLs, or local paths if you've already downloaded them), then run:

    !python colab_prepare_congestion_data.py --num-samples 20

Only ONE archive is ever resident on disk at a time (or, if streaming works
for your source, NONE are fully written to disk) -- so peak usage stays
around one archive's size instead of the sum of all five.

Output: /content/circuitnet/congestion_samples.zip containing
    congestion/feature/<sample_id>.npy   (256, 256, 3)
    congestion/label/<sample_id>.npy     (256, 256, 1)
which matches what scripts/eval_baseline.py, scripts/preprocess_circuitnet.py,
and scripts/inspect_data.py expect (point --root at the unzipped congestion/ dir).
"""

import argparse
import os
import shutil
import tarfile

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# CONFIG -- fill these in with your actual archive sources.
# Each value is either:
#   - a Google Drive file ID (string) -- downloaded via gdown
#   - a direct https URL (string starting with http) -- streamed via requests
#   - a local path to an already-downloaded .tar.gz
# ---------------------------------------------------------------------------
ARCHIVES = {
    "macro_region": {
        "source": "FILL_ME_IN",
        "keywords": ["macro_region"],
    },
    "rudy": {
        "source": "FILL_ME_IN",
        "keywords": ["RUDY/RUDY", "RUDY/RUDY_pin"],
    },
    "congestion": {
        "source": "FILL_ME_IN",
        "keywords": [
            "congestion_GR_horizontal_overflow",
            "congestion_GR_vertical_overflow",
        ],
    },
}
# the archive whose keyword list is used to pick the sample_id subset
# (pick your smallest/fastest-to-list archive here)
ID_SOURCE_KEY = "macro_region"

EXTRACT_DIR = "/content/circuitnet/extracted"
PACKED_DIR = "/content/circuitnet/training_set"
ROOT_KEY = "routability_features_decompressed"  # top-level folder inside CircuitNet archives


def is_url(source):
    return source.startswith("http://") or source.startswith("https://")


def is_drive_id(source):
    return not is_url(source) and not os.path.exists(source) and "/" not in source


def download_archive(source, dest_path):
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
        # already a local file
        if os.path.abspath(source) != os.path.abspath(dest_path):
            shutil.copy(source, dest_path)
    return dest_path


def extract_matching(tar_path, keywords, sample_ids, extract_dir):
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if any(kw in member.name for kw in keywords) and (
                sample_ids is None or os.path.basename(member.name) in sample_ids
            ):
                tar.extract(member, extract_dir)
                extracted += 1
    return extracted


def collect_sample_ids(tar_path, keywords, num_samples):
    with tarfile.open(tar_path, "r:gz") as tar:
        names = [n for n in tar.getnames() if keywords[0] in n and n.endswith(".npy")]
    return [os.path.basename(n) for n in names[:num_samples]]


def resize(a):
    return ndimage.zoom(a, (256 / a.shape[0], 256 / a.shape[1]), order=3)


def std(a):
    return a if a.max() == 0 else (a - a.min()) / (a.max() - a.min())


def pack_congestion(extract_dir, packed_dir, sample_ids):
    feature_dir = os.path.join(packed_dir, "congestion", "feature")
    label_dir = os.path.join(packed_dir, "congestion", "label")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    root = os.path.join(extract_dir, ROOT_KEY)
    num_ok, num_fail = 0, 0

    for sid in sample_ids:
        try:
            macro = np.load(os.path.join(root, "macro_region", sid))
            rudy = np.load(os.path.join(root, "RUDY", "RUDY", sid))
            rudy_pin = np.load(os.path.join(root, "RUDY", "RUDY_pin", sid))
            h = np.load(os.path.join(
                root, "congestion", "congestion_global_routing", "overflow_based",
                "congestion_GR_horizontal_overflow", sid,
            ))
            v = np.load(os.path.join(
                root, "congestion", "congestion_global_routing", "overflow_based",
                "congestion_GR_vertical_overflow", sid,
            ))
        except FileNotFoundError as e:
            print(f"[skip] {sid}: {e}")
            num_fail += 1
            continue

        feature = np.stack([std(resize(macro)), std(resize(rudy)), std(resize(rudy_pin))], axis=-1)
        label = std(resize(h) + resize(v))[..., None]

        np.save(os.path.join(feature_dir, sid), feature.astype(np.float32))
        np.save(os.path.join(label_dir, sid), label.astype(np.float32))
        num_ok += 1

    print(f"Packed {num_ok} samples ({num_fail} skipped) into {packed_dir}/congestion")
    return num_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--zip-out", default="/content/circuitnet/congestion_samples.zip")
    parser.add_argument("--keep-extracted", action="store_true", help="don't delete /content/circuitnet/extracted afterward")
    args = parser.parse_args()

    os.makedirs("/content/circuitnet", exist_ok=True)
    sample_ids = None

    for name, cfg in ARCHIVES.items():
        if cfg["source"] == "FILL_ME_IN":
            raise ValueError(f"Set ARCHIVES['{name}']['source'] before running this script.")

        tar_path = f"/content/circuitnet/{name}.tar.gz"
        print(f"\n=== {name} ===")
        print("downloading...")
        download_archive(cfg["source"], tar_path)

        if sample_ids is None and name == ID_SOURCE_KEY:
            sample_ids = collect_sample_ids(tar_path, cfg["keywords"], args.num_samples)
            print(f"picked {len(sample_ids)} sample IDs from {name}")

        if sample_ids is None:
            raise RuntimeError(
                f"ID_SOURCE_KEY='{ID_SOURCE_KEY}' must be processed before archives that "
                f"depend on its sample_ids -- check ARCHIVES ordering."
            )

        n = extract_matching(tar_path, cfg["keywords"], sample_ids, EXTRACT_DIR)
        print(f"extracted {n} files from {name}")

        os.remove(tar_path)
        print(f"deleted {tar_path} to free space")

    pack_congestion(EXTRACT_DIR, PACKED_DIR, sample_ids)

    if not args.keep_extracted:
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        print("cleaned up raw extracted files (kept only the packed feature/label npy)")

    shutil.make_archive(args.zip_out.rsplit(".zip", 1)[0], "zip", PACKED_DIR)
    print(f"\nDone. Zipped to {args.zip_out}")

    try:
        from google.colab import files

        files.download(args.zip_out)
    except ImportError:
        print("(not running in Colab -- skipped auto-download, zip is at the path above)")


if __name__ == "__main__":
    main()
