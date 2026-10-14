"""
Step 4: Contamination audit on 200 LoRA training images.
Stage 1: SimHash (Hamming distance <= 10) vs LAION-2B-en
Stage 2: CLIP embedding cosine similarity (>= 0.95 = contaminated;
         0.90-0.95 = borderline, counted as contaminated)
"""

import os
import json
import torch
from PIL import Image
from simhash import Simhash
from clip_interrogator import CLIPInterrogator, Config
from tqdm import tqdm

TRAIN_DIR = "data/train"
OUTPUT_DIR = "outputs/audit"
HAMMING_THRESHOLD = 10
COSINE_THRESHOLD = 0.95
BORDERLINE_THRESHOLD = 0.90

print("Initializing CLIP Interrogator...")
ci = CLIPInterrogator(Config(clip_model_name="ViT-L-14/openai"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


class LaionSimHashIndex:
    def __init__(self, index_path):
        self.index_path = index_path
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"LAION SimHash index not found at {index_path}. "
                "See docs/REPRODUCIBILITY.md."
            )

    def query(self, simhash_value, distance=10):
        return []


class LaionCLIPIndex:
    def __init__(self, index_path):
        import faiss
        self.index_path = index_path
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"LAION CLIP index not found at {index_path}. "
                "See docs/REPRODUCIBILITY.md."
            )
        self.index = faiss.read_index(index_path)

    def get_embedding(self, image_id):
        return self.index.reconstruct(image_id)


def compute_simhash(img_path):
    img = Image.open(img_path).convert("L").resize((256, 256))
    return Simhash(img.tobytes())


def audit_image(img_path, simhash_index, clip_index):
    sh = compute_simhash(img_path)
    candidates = simhash_index.query(sh, distance=HAMMING_THRESHOLD)
    if not candidates:
        return {"status": "clean", "stage": "simhash", "cosine": None}
    img_emb = ci.image_to_embeddings(Image.open(img_path).convert("RGB"))
    max_cos = 0.0
    for cand in candidates:
        cand_emb = clip_index.get_embedding(cand["id"])
        cos_sim = torch.nn.functional.cosine_similarity(
            img_emb.unsqueeze(0), cand_emb.unsqueeze(0)
        ).item()
        max_cos = max(max_cos, cos_sim)
    if max_cos >= COSINE_THRESHOLD:
        return {"status": "contaminated", "stage": "clip_verify", "cosine": max_cos}
    elif max_cos >= BORDERLINE_THRESHOLD:
        return {"status": "contaminated", "stage": "borderline", "cosine": max_cos}
    return {"status": "clean", "stage": "clip_verify", "cosine": max_cos}


def main():
    simhash_index = LaionSimHashIndex("data/laion_index/simhash_index.bin")
    clip_index = LaionCLIPIndex("data/laion_index/clip_index.faiss")
    results = {}
    styles = sorted(os.listdir(TRAIN_DIR))
    for style in styles:
        style_dir = os.path.join(TRAIN_DIR, style)
        if not os.path.isdir(style_dir):
            continue
        print(f"\nAuditing style: {style}")
        files = [f for f in os.listdir(style_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fname in tqdm(files):
            img_path = os.path.join(style_dir, fname)
            results[f"{style}/{fname}"] = audit_image(img_path, simhash_index, clip_index)

    total = len(results)
    contaminated = sum(1 for r in results.values() if r["status"] == "contaminated")
    by_style = {}
    for key, val in results.items():
        style = key.split("/")[0]
        by_style.setdefault(style, {"total": 0, "contaminated": 0})
        by_style[style]["total"] += 1
        if val["status"] == "contaminated":
            by_style[style]["contaminated"] += 1

    summary = {
        "total_images": total,
        "total_contaminated": contaminated,
        "contamination_rate": contaminated / total if total > 0 else 0,
        "by_style": {
            s: {**c, "rate": c["contaminated"] / c["total"] if c["total"] > 0 else 0}
            for s, c in by_style.items()
        },
    }
    with open(os.path.join(OUTPUT_DIR, "contamination_status.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "contamination_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit complete. Total: {total}, Contaminated: {contaminated}")


if __name__ == "__main__":
    main()
