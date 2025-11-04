#!/usr/bin/env python3
"""
Build FAISS inner‑product indexes from JSON embeddings produced by our new pipeline.

Inputs (default, configurable via args):
  code/embeddingGeneration/flava_<dataset>_image_embeddings.json
  code/embeddingGeneration/flava_<dataset>_text_embeddings.json

Outputs:
  code/embeddingGeneration/<prefix>_<dataset>_faiss_img.index
  code/embeddingGeneration/<prefix>_<dataset>_faiss_txt.index
  code/embeddingGeneration/<prefix>_<dataset>_img_names.json
  code/embeddingGeneration/<prefix>_<dataset>_txt_meta.json   (image_name + caption)

# From repo root: /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark
python code/embeddingGeneration/new_faiss_flava.py \
  --dataset flickr \
  --prefix flava \
  --in-dir /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/embeddingGeneration

# After generating embeddings with flava_embeddings.py
python code/embeddingGeneration/build_faiss_from_json.py \
  --dataset coco \
  --prefix flava \
  --in-dir /mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/embeddingGeneration


"""

import os, sys, json, argparse
from pathlib import Path
import numpy as np
import faiss

# Allow: from Benchmark.config.config_utils import load_config
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


def load_image_embeddings(path: Path):
    """
    Expects a list of {"image_name": str, "embedding": [..]}.
    Returns (names, vectors_float32).
    """
    data = json.loads(path.read_text())
    names = []
    vecs = []
    for obj in data:
        names.append(obj["image_name"])
        vecs.append(obj["embedding"])
    X = np.asarray(vecs, dtype="float32")
    return names, X


def load_text_embeddings(path: Path):
    """
    Expects a list of {"image_name": str, "caption": str, "embedding": [..]}.
    Returns (meta, vectors_float32) where meta[i] = {"image_name":..., "caption":...}.
    """
    data = json.loads(path.read_text())
    meta = []
    vecs = []
    for obj in data:
        meta.append({"image_name": obj["image_name"], "caption": obj.get("caption", "")})
        vecs.append(obj["embedding"])
    X = np.asarray(vecs, dtype="float32")
    return meta, X


def build_ip_index(X: np.ndarray) -> faiss.Index:
    """
    Build an inner‑product FAISS index. Assumes embeddings are already L2‑normalized.
    """
    dim = int(X.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IP indexes from JSON embeddings.")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Dataset key from config (e.g., coco, flickr). Defaults to selected_config.dataset.")
    parser.add_argument("--prefix", default="flava",
                        help="Model prefix used in file names (e.g., flava, clip, minilm).")
    parser.add_argument("--in-dir", default="/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/code/embeddingGeneration",
                        help="Directory where JSON embeddings live.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write FAISS indexes. Defaults to --in-dir.")
    args = parser.parse_args()

    cfg = load_config()
    dataset = args.dataset or cfg["selected_config"]["dataset"]

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img_json = in_dir / f"{args.prefix}_{dataset}_image_embeddings.json"
    txt_json = in_dir / f"{args.prefix}_{dataset}_text_embeddings.json"

    if not img_json.exists():
        raise FileNotFoundError(f"Image embeddings not found: {img_json}")
    if not txt_json.exists():
        raise FileNotFoundError(f"Text embeddings not found:  {txt_json}")

    # Load
    img_names, X_img = load_image_embeddings(img_json)
    txt_meta, X_txt  = load_text_embeddings(txt_json)

    if X_img.ndim != 2 or X_txt.ndim != 2:
        raise ValueError("Embeddings must be 2‑D arrays.")

    # Build indexes
    idx_img = build_ip_index(X_img)
    idx_txt = build_ip_index(X_txt)

    # Write indexes
    img_index_path = out_dir / f"{args.prefix}_{dataset}_faiss_img.index"
    txt_index_path = out_dir / f"{args.prefix}_{dataset}_faiss_txt.index"
    faiss.write_index(idx_img, str(img_index_path))
    faiss.write_index(idx_txt, str(txt_index_path))

    # Write simple metadata for lookups
    (out_dir / f"{args.prefix}_{dataset}_img_names.json").write_text(
        json.dumps(img_names, indent=2)
    )
    (out_dir / f"{args.prefix}_{dataset}_txt_meta.json").write_text(
        json.dumps(txt_meta, indent=2)
    )

    print(f"✓ Built image index: {img_index_path}  ({len(img_names)} vectors, dim={X_img.shape[1]})")
    print(f"✓ Built text  index: {txt_index_path}  ({len(txt_meta)} vectors, dim={X_txt.shape[1]})")


if __name__ == "__main__":
    main()
