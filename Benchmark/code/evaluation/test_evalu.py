import os, json, requests
from tqdm import tqdm

# ---- network env (no proxy for localhost) ----
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
proxies = {'http': '', 'https': ''}

# ---- CONFIG ----
BASE_URL = "http://localhost:5052"
DATASET = "flickr"  # "coco" if you run the coco server
ENDPOINT = f"{BASE_URL}/preflmr_image_{DATASET}_colbert"

ANNOTATION_FILE = "/mnt/storage/RSystemsBenchmarking/data/datasets/vision/flickr30k/annotations/test.json"
TOP_K = 10
MAX_SAMPLES = None   # None ⇒ evaluate all
CAPTION_IDX = 4      # use the 5th caption per your original script
VARIANTS = ["joint"]  # try "all" via one call if you prefer (see note below)

# If your GT "image" is a relative path like "flickr30k-images/123.jpg"
# and server returns the same, keep this False. If needed, flip to True to compare by basename only.
NORMALIZE_TO_BASENAME = False


def normalize(name: str) -> str:
    if not NORMALIZE_TO_BASENAME:
        return name
    import os as _os
    return _os.path.basename(name)


def evaluate_variant(service_url: str, variant: str):
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    r1 = r5 = r10 = 0
    total = 0

    it = data if MAX_SAMPLES is None else data[:MAX_SAMPLES]
    for entry in tqdm(it, desc=f"Evaluating variant={variant}"):
        image_gt = normalize(entry["image"])

        caps = entry.get("caption", []) or []
        if len(caps) == 0:
            # fall back to empty query (rare)
            query_caption = ""
        else:
            # clamp CAPTION_IDX if fewer than 5 captions exist
            idx = min(CAPTION_IDX, len(caps) - 1)
            query_caption = caps[idx].strip()

        try:
            # Server ignores 'methods'; you can drop it.
            params = {"q": query_caption, "k": TOP_K, "variant": variant}
            resp = requests.get(service_url, params=params, proxies=proxies, timeout=60)
            resp.raise_for_status()

            payload = resp.json()
            # If you instead call variant="all", payload looks like:
            # {"variants": {"joint": {...}, "image_only": {...}, "caption_only": {...}}}
            results = payload["list_of_top_k"]
        except Exception as e:
            print(f"[{variant}] Error for query: {query_caption!r} -> {e}")
            continue

        retrieved = [normalize(r["image_path"]) for r in results]

        total += 1
        if image_gt in retrieved[:1]:  r1  += 1
        if image_gt in retrieved[:5]:  r5  += 1
        if image_gt in retrieved[:10]: r10 += 1

    if total == 0:
        print(f"[{variant}] No samples evaluated.")
        return

    print(f"\n[{variant}] Evaluated on {total} samples")
    print(f"[{variant}] R@1 : {r1/total:.4f}")
    print(f"[{variant}] R@5 : {r5/total:.4f}")
    print(f"[{variant}] R@10: {r10/total:.4f}")


def main():
    # sanity check
    try:
        ping = requests.get(f"{BASE_URL}/available", proxies=proxies, timeout=10).json()
        print("Server /available:", ping)
    except Exception as e:
        print("Warning: Could not reach /available:", e)

    for v in VARIANTS:
        evaluate_variant(ENDPOINT, v)


if __name__ == "__main__":
    main()
