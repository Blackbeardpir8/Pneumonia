
import pandas as pd
import numpy as np
from PIL import Image

manifest_path = "data/processed/segmentation_covid/manifest.csv"

df = pd.read_csv(manifest_path)

print("Total records:", len(df))

print("\nSplit distribution:")
print(df.groupby("split").size())

print("\nChecking first 10 image-mask pairs:")

for _, row in df.head(10).iterrows():

    image = np.array(
        Image.open(row["processed_image_path"])
    )

    mask = np.array(
        Image.open(row["processed_mask_path"])
    )

    print(
        row["class_name"],
        "| Image:", image.shape,
        "| Mask:", mask.shape,
        "| Mask values:", np.unique(mask).tolist(),
        "| Mask coverage:",
        round(np.mean(mask > 0) * 100, 2),
        "%"
    )

print("\nVerification completed.")

# to run this script, use the following command:
# python scripts/67_verify_covid_segmentation_data.py