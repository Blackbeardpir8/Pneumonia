
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path("data/raw/archive/COVID-19_Radiography_Dataset")
OUTPUT = Path("results/segmentation/covid_mask_audit/visual_samples")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "COVID": "COVID-1.png",
    "Normal": "Normal-1.png",
    "Viral Pneumonia": "Viral Pneumonia-1.png",
    "Lung_Opacity": "Lung_Opacity-1.png",
}

for class_name, filename in SAMPLES.items():

    image_path = BASE / class_name / "images" / filename
    mask_path = BASE / class_name / "masks" / filename

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image)
    mask_array = np.array(mask)

    # Resize the mask to the image dimensions for visualization only.
    mask_resized = np.array(
        Image.fromarray(mask_array).resize(
            image.size,
            Image.Resampling.NEAREST
        )
    )

    overlay = image_array.copy()
    foreground = mask_resized > 0

    overlay[foreground] = (
        0.5 * overlay[foreground] +
        0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_array)
    axes[0].set_title(f"{class_name} - Original X-ray")

    axes[1].imshow(mask_array, cmap="gray")
    axes[1].set_title("Original Ground-truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Nearest-neighbor resized overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_file = OUTPUT / f"{class_name.replace(' ', '_')}_alignment.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print("Saved:", output_file)

print("\nVisual alignment samples generated.")
