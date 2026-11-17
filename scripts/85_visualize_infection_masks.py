
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
    "infection_segmentation_audit.csv"
)

OUTPUT_DIR = ROOT / "results/infection_mask_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Select representative COVID samples by infection-mask area.
covid = df[
    (df["class"] == "COVID-19") &
    (df["split"] == "Train")
].sort_values("infection_foreground_percent")

selected = [
    covid.iloc[0],
    covid.iloc[len(covid) // 2],
    covid.iloc[-1],
]

# Add one Non-COVID and one Normal sample.
for class_name in ["Non-COVID", "Normal"]:
    sample = df[
        (df["class"] == class_name) &
        (df["split"] == "Train")
    ].iloc[0]

    selected.append(sample)

fig, axes = plt.subplots(
    len(selected), 4,
    figsize=(16, 4 * len(selected))
)

for row, sample in enumerate(selected):

    split = sample["split"]
    class_name = sample["class"]
    filename = sample["filename"]

    base = DATASET / split / class_name

    # Find files using their common filename stem.
    def find_file(folder):
        matches = [
            p for p in folder.iterdir()
            if p.is_file() and p.stem == filename
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one file for {filename} in {folder}, "
                f"found {len(matches)}"
            )

        return matches[0]

    image_path = find_file(base / "images")
    infection_path = find_file(base / "infection masks")
    lung_path = find_file(base / "lung masks")

    image = np.array(
        Image.open(image_path).convert("L")
    )

    infection = np.array(
        Image.open(infection_path).convert("L")
    )

    lung = np.array(
        Image.open(lung_path).convert("L")
    )

    infection_binary = infection > 0
    lung_binary = lung > 0

    # RGB overlay: green = lung, red = infection.
    overlay = np.stack([image, image, image], axis=-1).astype(
        np.float32
    )

    overlay[lung_binary] = (
        overlay[lung_binary] * 0.55
        + np.array([0, 255, 0]) * 0.45
    )

    overlay[infection_binary] = (
        overlay[infection_binary] * 0.35
        + np.array([255, 0, 0]) * 0.65
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    label = (
        f"{class_name} | {filename}\n"
        f"Infection area: {sample['infection_foreground_percent']:.2f}%"
    )

    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 0].set_title(f"{label}\nOriginal")

    axes[row, 1].imshow(infection, cmap="gray")
    axes[row, 1].set_title("Infection Mask")

    axes[row, 2].imshow(lung, cmap="gray")
    axes[row, 2].set_title("Lung Mask")

    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title("Overlay (Red: infection, Green: lung)")

    for col in range(4):
        axes[row, col].axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "infection_mask_visual_inspection.png"

plt.savefig(
    output_path,
    dpi=160,
    bbox_inches="tight"
)

plt.close()

print("Visualization generated successfully.")
print("Saved at:", output_path)
print("\nSelected samples:")

for sample in selected:
    print(
        sample["class"],
        sample["filename"],
        f"{sample['infection_foreground_percent']:.2f}%"
    )

# to run this script, use the following command:
# python scripts/85_visualize_infection_masks.py