
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data/raw/archive/COVID-19_Radiography_Dataset"

SAMPLES = {
    "COVID": "COVID-1.png",
    "Viral Pneumonia": "Viral Pneumonia-1.png",
    "Normal": "Normal-1.png",
}

OUTPUT_DIR = ROOT / "results/infection_mask_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, (class_name, filename) in enumerate(SAMPLES.items()):

    image_path = DATASET / class_name / "images" / filename
    mask_path = DATASET / class_name / "masks" / filename

    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    # Resize mask to match original X-ray dimensions.
    mask = mask.resize(image.size, Image.Resampling.NEAREST)

    image = np.array(image)
    mask = np.array(mask)

    # Create RGB overlay.
    overlay = np.stack([image, image, image], axis=-1)

    foreground = mask > 0

    overlay[foreground] = (
        overlay[foreground] * 0.5
        + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 0].set_title(f"{class_name} - Original")

    axes[row, 1].imshow(mask, cmap="gray")
    axes[row, 1].set_title("Provided Mask")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("Mask Overlay")

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "covid_mask_visual_inspection.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print("Visualization generated successfully.")
print(f"Saved at: {output_path}")
