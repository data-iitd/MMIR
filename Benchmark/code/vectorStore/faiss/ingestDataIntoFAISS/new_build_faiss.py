#!/usr/bin/env python3
"""
Build FAISS inner-product indexes from JSON embeddings.

It now works even if you have *only* image *or* text embeddings.

Inputs (default, configurable via args):
  <in-dir>/<model>_<dataset>_image_embeddings.json   (optional)
  <in-dir>/<model>_<dataset>_text_embeddings.json    (optional)

Outputs – by default into config['vector_store']['faiss']['index_dir']:
  <model>_<dataset>_faiss_img.index       (if images present)
  <model>_<dataset>_faiss_txt.index       (if text present)
  <model>_<dataset>_img_names.json        (if images present)
  <model>_<dataset>_txt_meta.json         (if text present)
"""

import sys, json, argparse, warnings
from pathlib import Path

import faiss
import numpy as np

# project import
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config


# ───────────────────────── helpers ───────────────────────── #

def load_image_embeddings(path: Path):
    data  = json.loads(path.read_text())
    names = [o["image_name"] for o in data]
    vecs  = np.asarray([o["embedding"] for o in data], dtype="float32")
    return names, vecs


def load_text_embeddings(path: Path):
    data  = json.loads(path.read_text())
    meta  = [{"image_name": o["image_name"], "caption": o.get("caption", "")}
             for o in data]
    vecs  = np.asarray([o["embedding"] for o in data], dtype="float32")
    return meta, vecs


def build_ip_index(x: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(int(x.shape[1]))
    idx.add(x)
    return idx


# ─────────────────────────── main ─────────────────────────── #

def main():
    cfg = load_config()

    p = argparse.ArgumentParser(
        description="Build FAISS IP indexes from JSON embeddings (image, text, or both)."
    )
    p.add_argument("--dataset", "-d", default=None,
                   help="Dataset key (e.g. coco, flickr). Defaults to selected_config.dataset.")
    p.add_argument("--model", default="flava",
                   help="Model tag in file names (e.g. flava, uniir, minilm).")
    p.add_argument("--in-dir", default=cfg["paths"]["project_root"] + "data/embeddings",
                   help="Directory containing the JSON embeddings.")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (defaults to faiss.index_dir in config).")
    args = p.parse_args()

    dataset   = args.dataset or cfg["selected_config"]["dataset"]
    in_dir    = Path(args.in_dir)

    # — default to config value unless user overrides —
    default_out = Path(cfg["vector_store"]["faiss"]["index_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    img_json = in_dir / f"{args.model}_{dataset}_image_embeddings.json"
    txt_json = in_dir / f"{args.model}_{dataset}_text_embeddings.json"

    img_exists = img_json.exists()
    txt_exists = txt_json.exists()

    if not img_exists and not txt_exists:
        raise FileNotFoundError(
            f"Neither image nor text embeddings found for model '{args.model}' "
            f"and dataset '{dataset}' in {in_dir}"
        )

    # ── Image embeddings ─────────────────────────────────── #
    if img_exists:
        names_img, x_img = load_image_embeddings(img_json)
        if x_img.ndim != 2:
            raise ValueError("Image embeddings must be a 2-D array.")
        idx_img = build_ip_index(x_img)

        img_idx_path = out_dir / f"{args.model}_{dataset}_faiss_img.index"
        faiss.write_index(idx_img, str(img_idx_path))
        (out_dir / f"{args.model}_{dataset}_img_names.json").write_text(
            json.dumps(names_img, indent=2)
        )
        print(f"✓ Built IMAGE index → {img_idx_path}  "
              f"({len(names_img)} vecs, dim={x_img.shape[1]})")
    else:
        warnings.warn("No image embedding file found – skipping image index.")

    # ── Text embeddings ──────────────────────────────────── #
    if txt_exists:
        meta_txt, x_txt = load_text_embeddings(txt_json)
        if x_txt.ndim != 2:
            raise ValueError("Text embeddings must be a 2-D array.")
        idx_txt = build_ip_index(x_txt)

        txt_idx_path = out_dir / f"{args.model}_{dataset}_faiss_txt.index"
        faiss.write_index(idx_txt, str(txt_idx_path))
        (out_dir / f"{args.model}_{dataset}_txt_meta.json").write_text(
            json.dumps(meta_txt, indent=2)
        )
        print(f"✓ Built TEXT  index → {txt_idx_path}  "
              f"({len(meta_txt)} vecs, dim={x_txt.shape[1]})")
    else:
        warnings.warn("No text embedding file found – skipping text index.")


if __name__ == "__main__":
    main()
