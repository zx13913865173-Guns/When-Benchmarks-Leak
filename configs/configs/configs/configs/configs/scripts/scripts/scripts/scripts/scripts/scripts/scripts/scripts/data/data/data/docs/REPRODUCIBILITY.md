# Reproducibility Guide

## 1. Data Preparation

### 1.1 LoRA Training Images
Download 50 public-domain images per style:
- Chinese ink wash: WikiArt, The Met Open Access
- Japanese ukiyo-e: The Met Open Access, Rijksmuseum
- Cyberpunk: CC0 digital art archives
- Baroque: WikiArt, Rijksmuseum
- Minimalist: WikiArt, The Met Open Access

Place in `data/train/<style>/001.jpg` through `050.jpg`.

### 1.2 Ground Truth Labels
`data/ground_truth_labels.csv` columns: filename, ground_truth_styles, dominant, secondary.

Ground truth derived from three sources:
1. LoRA weight ratios
2. Generation prompt descriptors
3. Three independent art expert annotations (Fleiss Kappa = 0.81)

Only labels with agreement across at least two sources included.

## 2. LAION Index
Not included due to size (multiple TB). Options:

### Option A: Local Build
```bash
pip install clip-retrieval
clip-retrieval index --input laion_metadata.parquet --output data/laion_index
