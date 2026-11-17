
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path("data/processed/segmentation")

for split in ["train", "val"]:
    mask_dir = ROOT / split / "masks"

    rows = []

    for path in sorted(mask_dir.glob("*_mask.png")):
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) >= 128

        _, count = ndimage.label(mask)

        rows.append({
            "source": path.name.split("_")[0],
            "components": count,
        })

    df = pd.DataFrame(rows)

    print(f"\n{split.upper()} COMPONENT DISTRIBUTION")
    print("=" * 55)

    for source, group in df.groupby("source"):
        print(f"\n{source} ({len(group)} masks)")

        counts = Counter(group["components"])

        for component_count, frequency in sorted(counts.items()):
            percentage = frequency / len(group) * 100

            print(
                f"{component_count} components: "
                f"{frequency} masks ({percentage:.2f}%)"
            )



# to run script, use the following command:
# python scripts/18_check_mask_components.py