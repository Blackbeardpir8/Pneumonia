
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/processed/segmentation/val/images")
CSV_PATH = Path("results/segmentation/validation_metrics.csv")

df = pd.read_csv(CSV_PATH)

rows = []

for _, row in df.iterrows():
    image_path = ROOT / row["image"]

    with Image.open(image_path) as image:
        image_array = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        )

    rows.append({
        "source": row["image"].split("_")[0],
        "brightness": image_array.mean(),
        "contrast": image_array.std(),
        "min": image_array.min(),
        "max": image_array.max(),
    })

stats = pd.DataFrame(rows)

print("\nIMAGE STATISTICS BY DATASET")
print("=" * 65)

print(
    stats.groupby("source")[
        ["brightness", "contrast", "min", "max"]
    ].agg(["mean", "std"]).round(2).to_string()
)

print("\nTotal images:", len(stats))


# to run 