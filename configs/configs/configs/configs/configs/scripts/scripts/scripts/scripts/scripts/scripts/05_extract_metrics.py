"""
Step 5: Extract recall, precision, F1, coverage against ground truth.
"""

import os
import json
import pandas as pd
from collections import defaultdict

INTERROGATOR_DIR = "outputs/interrogator"
OUTPUT_DIR = "outputs/metrics"
GROUND_TRUTH_PATH = "data/ground_truth_labels.csv"

STYLE_KEYWORDS = {
    "ink_wash": ["ink wash", "chinese ink", "brush painting", "sumi-e", "ink painting"],
    "ukiyoe": ["ukiyo-e", "ukiyoe", "japanese print", "woodblock print"],
    "cyberpunk": ["cyberpunk", "neon", "futuristic city", "sci-fi"],
    "baroque": ["baroque", "chiaroscuro", "dramatic lighting", "ornate"],
    "minimalist": ["minimalist", "minimalism", "simple", "clean lines"],
}


def extract_style_labels(prompt):
    prompt_lower = prompt.lower()
    detected = set()
    for style, keywords in STYLE_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                detected.add(style)
                break
    return detected


def compute_metrics(detected, ground_truth):
    detected, gt = set(detected), set(ground_truth)
    if not gt:
        return None
    recall = len(detected & gt) / len(gt)
    precision = len(detected & gt) / len(detected) if detected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = len(detected & gt) / len(gt)
    return {"recall": recall, "precision": precision, "f1": f1, "coverage": coverage}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(INTERROGATOR_DIR, "prompts.json")) as f:
        prompts = json.load(f)
    gt_df = pd.read_csv(GROUND_TRUTH_PATH)
    with open("outputs/images/metadata.json") as f:
        metadata = {m["filename"]: m for m in json.load(f)}

    rows = []
    for _, gt_row in gt_df.iterrows():
        img_name = gt_row["filename"]
        if img_name not in prompts or img_name not in metadata:
            continue
        meta = metadata[img_name]
        gt_styles = [s.strip() for s in gt_row["ground_truth_styles"].split(";")]
        global_detected = extract_style_labels(prompts[img_name]["global"])
        regional_detected = extract_style_labels(prompts[img_name]["merged_regional"])
        gm = compute_metrics(global_detected, gt_styles)
        rm = compute_metrics(regional_detected, gt_styles)
        if gm is None or rm is None:
            continue
        if meta["type"] in ("primary_hybrid", "generalization_hybrid"):
            if meta["weight_a"] > meta["weight_b"]:
                dominant, secondary = meta["style_a"], meta["style_b"]
            elif meta["weight_b"] > meta["weight_a"]:
                dominant, secondary = meta["style_b"], meta["style_a"]
            else:
                dominant, secondary = None, None
        else:
            dominant, secondary = meta.get("style"), None
        rows.append({
            "filename": img_name, "type": meta["type"],
            "style_a": meta.get("style_a"), "style_b": meta.get("style_b"),
            "weight_a": meta.get("weight_a"), "weight_b": meta.get("weight_b"),
            "dominant": dominant, "secondary": secondary,
            "global_recall": gm["recall"], "global_precision": gm["precision"],
            "global_f1": gm["f1"], "global_coverage": gm["coverage"],
            "regional_recall": rm["recall"], "regional_precision": rm["precision"],
            "regional_f1": rm["f1"], "regional_coverage": rm["coverage"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "per_image_metrics.csv"), index=False)
    print(f"Metrics extracted for {len(df)} images.")


if __name__ == "__main__":
    main()
