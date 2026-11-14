
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/processed/segmentation")

for split in ["train", "val"]:
    mask_dir = ROOT / split / "masks"
    rows = []

    for mask_path in sorted(mask_dir.glob("*_mask.png")):
        with Image.open(mask_path) as image:
            mask = np.asarray(
                image.convert("L")
            ) >= 128

        rows.append({
            "source": mask_path.name.split("_")[0],
            "foreground_area": mask.mean(),
        })

    df = pd.DataFrame(rows)

    print(f"\n{split.upper()} MASK STATISTICS")
    print("=" * 55)

    print(
        df.groupby("source")["foreground_area"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
        .to_string()
    )

    print("Total masks:", len(df))


# to run 