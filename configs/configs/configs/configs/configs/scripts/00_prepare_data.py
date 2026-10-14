"""
Step 0: Prepare training data directory structure.

Download 50 public-domain images per style from:
  - WikiArt: https://www.wikiart.org/
  - The Met Open Access: https://www.metmuseum.org/art/collection
  - Rijksmuseum: https://www.rijksmuseum.nl/en/collection

Place them in data/train/<style>/001.jpg through 050.jpg.
"""

import os

STYLES = ["ink_wash", "ukiyoe", "cyberpunk", "baroque", "minimalist"]

for style in STYLES:
    os.makedirs(f"data/train/{style}", exist_ok=True)
    print(f"Created data/train/{style}/")

print("\nDownload 50 public-domain images per style and place in directories.")
