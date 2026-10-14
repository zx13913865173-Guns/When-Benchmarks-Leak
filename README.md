# When Benchmarks Leak: Data Contamination and Its Limited Role in Cross-Cultural Style Decomposition Using CLIP Interrogator

Complete reproduction package for the manuscript submitted to [Journal Name].

## Overview

This repository contains all code, configurations, and data to reproduce the results reported in the paper:

1. LoRA training for four cultural styles (Chinese ink wash, Japanese ukiyo-e, cyberpunk, Baroque)
2. Hybrid image generation using SDXL Base 1.0
3. CLIP Interrogator application (global and 2×2 regional)
4. Training-data contamination audit (SimHash + CLIP embedding)
5. Statistical analysis (mixed-effects models with FDR correction)
6. Publication-quality figures

## Hardware Requirements

- GPU: NVIDIA RTX 4070 Super 12GB (or higher)
- RAM: 32 GB
- Storage: 500 GB (for LoRA training, image generation, LAION index cache)
- OS: Windows 11 / Ubuntu 22.04

## Software Requirements

- Python 3.10.11
- PyTorch 2.1.0+cu121
- Diffusers 0.25.0
- clip-interrogator 0.6.0
- R 4.3.2 with lme4, lmerTest, effectsize

## Reproduction Steps

### Step 0: Environment Setup

```bash
pip install -r requirements.txt
Rscript -e 'install.packages(c("lme4", "lmerTest", "effectsize", "emmeans", "ggplot2"))'
