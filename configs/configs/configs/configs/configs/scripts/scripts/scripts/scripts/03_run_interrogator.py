"""
Step 3: Run CLIP Interrogator (global + 2x2 regional) on all images.
"""

import os
import json
from PIL import Image
from clip_interrogator import CLIPInterrogator, Config
from tqdm import tqdm

IMAGE_DIR = "outputs/images"
OUTPUT_DIR = "outputs/interrogator"
CLIP_MODEL = "ViT-L-14/openai"

print(f"Initializing CLIP Interrogator with {CLIP_MODEL}...")
ci = CLIPInterrogator(Config(clip_model_name=CLIP_MODEL))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def interrogate_regional(image):
    w, h = image.size
    quadrants = {
        "top_left": image.crop((0, 0, w // 2, h // 2)),
        "top_right": image.crop((w // 2, 0, w, h // 2)),
        "bottom_left": image.crop((0, h // 2, w // 2, h)),
        "bottom_right": image.crop((w // 2, h // 2, w, h)),
    }
    regional = {k: ci.interrogate(v) for k, v in quadrants.items()}
    merged = " ".join(regional.values())
    return regional, merged


results = {}
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
print(f"Processing {len(image_files)} images...")

for fname in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, fname)
    image = Image.open(img_path).convert("RGB")
    global_prompt = ci.interrogate(image)
    regional, merged = interrogate_regional(image)
    results[fname] = {"global": global_prompt, "regional": regional, "merged_regional": merged}

with open(os.path.join(OUTPUT_DIR, "prompts.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Interrogation complete. Saved to {OUTPUT_DIR}/prompts.json")
