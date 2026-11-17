
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASK_DIRS = {
    "COVID": PROJECT_ROOT / "data/raw/archive/COVID-19_Radiography_Dataset/COVID/masks",
    "Viral Pneumonia": PROJECT_ROOT / "data/raw/archive/COVID-19_Radiography_Dataset/Viral Pneumonia/masks",
    "Normal": PROJECT_ROOT / "data/raw/archive/COVID-19_Radiography_Dataset/Normal/masks",
}

for class_name, mask_dir in MASK_DIRS.items():
    print(f"\n{'=' * 60}")
    print(f"CLASS: {class_name}")
    print(f"{'=' * 60}")

    files = sorted(mask_dir.glob("*.png"))[:3]

    for file_path in files:
        mask = np.array(Image.open(file_path).convert("L"))

        unique_values = np.unique(mask)
        foreground_pixels = np.count_nonzero(mask > 0)
        total_pixels = mask.size

        print(f"\nFile: {file_path.name}")
        print(f"Dimensions: {mask.shape}")
        print(f"Unique pixel values: {unique_values[:20]}")
        print(
            f"Foreground percentage: "
            f"{foreground_pixels / total_pixels * 100:.2f}%"
        )

# to run this script, use the following command:
# python scripts/81_inspect_covid_masks.py