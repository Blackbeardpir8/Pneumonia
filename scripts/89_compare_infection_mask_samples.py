
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

CSV_PATH = (
    ROOT / "results/infection_mask_investigation/"
    "infection_area_quality_audit.csv"
)

OUTPUT_DIR = ROOT / "results/infection_mask_investigation"

df = pd.read_csv(CSV_PATH)
df = df[df["split"] == "Train"].copy()

identical = df[df["identical_masks"] == True]

different = df[df["identical_masks"] == False].copy()

# Select representative examples.
selected = []

selected.extend(
    identical.sort_values(
        "infection_to_lung_percentage"
    ).head(2).to_dict("records")
)

selected.extend(
    different.nsmallest(
        2, "infection_to_lung_percentage"
    ).to_dict("records")
)

selected.extend(
    different.nlargest(
        2, "infection_to_lung_percentage"
    ).to_dict("records")
)

fig, axes = plt.subplots(
    len(selected), 4,
    figsize=(16, 4 * len(selected))
)

for row, sample in enumerate(selected):

    filename = sample["filename"]

    base = DATASET / "Train" / "COVID-19"

    image = np.array(
        Image.open(
            base / "images" / f"{filename}.png"
        ).convert("L")
    )

    infection = np.array(
        Image.open(
            base / "infection masks" / f"{filename}.png"
        ).convert("L")
    ) > 0

    lung = np.array(
        Image.open(
            base / "lung masks" / f"{filename}.png"
        ).convert("L")
    ) > 0

    overlay = np.stack([image, image, image], axis=-1).astype(
        np.float32
    )

    overlay[lung] = (
        overlay[lung] * 0.55
        + np.array([0, 255, 0]) * 0.45
    )

    overlay[infection] = (
        overlay[infection] * 0.35
        + np.array([255, 0, 0]) * 0.65
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    ratio = sample["infection_to_lung_percentage"]
    is_identical = sample["identical_masks"]

    label = (
        f"{filename} | Infection/Lung: {ratio:.2f}%\n"
        f"Identical masks: {is_identical}"
    )

    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 0].set_title(f"{label}\nOriginal")

    axes[row, 1].imshow(infection, cmap="gray")
    axes[row, 1].set_title("Infection Mask")

    axes[row, 2].imshow(lung, cmap="gray")
    axes[row, 2].set_title("Lung Mask")

    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title("Red: infection | Green: lung")

    for col in range(4):
        axes[row, col].axis("off")

plt.tight_layout()

output_path = (
    OUTPUT_DIR / "infection_mask_quality_comparison.png"
)

plt.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close()

print("Comparison generated:", output_path)

print("\nSelected samples:")

for sample in selected:
    print(
        sample["filename"],
        "| Infection/Lung:",
        round(sample["infection_to_lung_percentage"], 2),
        "%",
        "| Identical:",
        sample["identical_masks"],
    )
