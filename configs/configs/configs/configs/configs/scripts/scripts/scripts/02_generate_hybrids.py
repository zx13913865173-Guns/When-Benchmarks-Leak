"""
Step 2: Generate hybrid images using SDXL Base 1.0 + LoRA adapters.

Dataset composition (total = 140):
  - Primary fusion pairs: 4 pairs x 4 ratio conditions x 5 reps = 80
  - Pure-style controls: 4 styles x 5 images = 20
  - Generalization pairs: 2 pairs x 4 ratio conditions x 5 reps = 40

Weight ratios: (0.3,0.7), (0.5,0.5), (0.7,0.3), (0.5,0.5)
"""

import os
import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_DIR = "outputs/lora"
OUTPUT_DIR = "outputs/images"
SEED = 42
CFG_SCALE = 7.5
NUM_STEPS = 30
RESOLUTION = 1024
NEGATIVE_PROMPT = "low quality, blurry, distorted, ugly, bad anatomy"

PRIMARY_PAIRS = [
    ("ink_wash", "cyberpunk"),
    ("ukiyoe", "baroque"),
    ("ink_wash", "minimalist"),
    ("cyberpunk", "baroque"),
]

GENERALIZATION_PAIRS = [
    ("ink_wash", "minimalist"),
    ("ukiyoe", "minimalist"),
]

WEIGHT_RATIOS = [
    (0.3, 0.7),
    (0.5, 0.5),
    (0.7, 0.3),
    (0.5, 0.5),
]

REPETITIONS = 5

print("Loading SDXL Base 1.0...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)


def load_lora_adapters(style_a, style_b):
    pipe.unload_lora_weights()
    pipe.load_lora_weights(f"{LORA_DIR}/{style_a}/{style_a}_lora.safetensors", adapter_name=style_a)
    pipe.load_lora_weights(f"{LORA_DIR}/{style_b}/{style_b}_lora.safetensors", adapter_name=style_b)
    return style_a, style_b


def generate_image(weight_a, weight_b, style_a, style_b, seed):
    pipe.set_adapters([style_a, style_b], adapter_weights=[weight_a, weight_b])
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt="hybrid artwork, masterpiece, high quality, detailed",
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=CFG_SCALE,
        width=RESOLUTION,
        height=RESOLUTION,
        generator=generator,
    ).images[0]
    return image


os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = []

print("\n=== Primary fusion images ===")
for pair_idx, (style_a, style_b) in enumerate(PRIMARY_PAIRS):
    load_lora_adapters(style_a, style_b)
    for ratio_idx, (w_a, w_b) in enumerate(WEIGHT_RATIOS):
        for rep in range(REPETITIONS):
            seed = SEED + pair_idx * 1000 + ratio_idx * 100 + rep
            image = generate_image(w_a, w_b, style_a, style_b, seed)
            fname = f"primary_{style_a}_{style_b}_r{ratio_idx}_rep{rep}.png"
            image.save(os.path.join(OUTPUT_DIR, fname))
            metadata.append({"filename": fname, "type": "primary_hybrid",
                             "style_a": style_a, "style_b": style_b,
                             "weight_a": w_a, "weight_b": w_b,
                             "ratio_index": ratio_idx, "repetition": rep, "seed": seed})
            print(f"  Saved {fname}")

print("\n=== Pure-style controls ===")
for style in ["ink_wash", "ukiyoe", "cyberpunk", "baroque"]:
    pipe.unload_lora_weights()
    pipe.load_lora_weights(f"{LORA_DIR}/{style}/{style}_lora.safetensors", adapter_name=style)
    pipe.set_adapters([style], adapter_weights=[1.0])
    for rep in range(REPETITIONS):
        seed = SEED + 9000 + rep
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt=f"{style} artwork, masterpiece, high quality",
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG_SCALE,
            width=RESOLUTION, height=RESOLUTION, generator=generator,
        ).images[0]
        fname = f"pure_{style}_rep{rep}.png"
        image.save(os.path.join(OUTPUT_DIR, fname))
        metadata.append({"filename": fname, "type": "pure_style",
                         "style": style, "repetition": rep, "seed": seed})
        print(f"  Saved {fname}")

print("\n=== Generalization fusion images ===")
for pair_idx, (style_a, style_b) in enumerate(GENERALIZATION_PAIRS):
    load_lora_adapters(style_a, style_b)
    for ratio_idx, (w_a, w_b) in enumerate(WEIGHT_RATIOS):
        for rep in range(REPETITIONS):
            seed = SEED + 5000 + pair_idx * 1000 + ratio_idx * 100 + rep
            image = generate_image(w_a, w_b, style_a, style_b, seed)
            fname = f"gen_{style_a}_{style_b}_r{ratio_idx}_rep{rep}.png"
            image.save(os.path.join(OUTPUT_DIR, fname))
            metadata.append({"filename": fname, "type": "generalization_hybrid",
                             "style_a": style_a, "style_b": style_b,
                             "weight_a": w_a, "weight_b": w_b,
                             "ratio_index": ratio_idx, "repetition": rep, "seed": seed})
            print(f"  Saved {fname}")

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nGenerated {len(metadata)} images.")
