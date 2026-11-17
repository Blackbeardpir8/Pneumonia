"""
58_visualize_low_mask_segmentation_audit.py

Creates class-wise contact sheets and individual mask overlays for the
20 validation images listed in low_mask_audit_samples.csv.
Uses the balanced U-Net and preprocessing from script 56.
No training or checkpoint modification.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
UNET_PATH = ROOT / "saved_models" / "unet_lung_segmentation_balanced.keras"
CSV_PATH = ROOT / "results/classification/lung_only_classifier_ablation/low_mask_audit_samples.csv"
OUT = ROOT / "results/classification/lung_only_classifier_ablation/low_mask_visual_audit"
SIZE = 224
THRESHOLD = 0.30
CLASSES = ["Normal", "Viral", "Bacterial", "COVID"]

def read_image(path, channels):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])
    image = tf.image.resize(image, [SIZE, SIZE], method="bilinear")
    image = tf.cast(image, tf.float32)
    if channels == 1:
        image = image / 255.0
    return image.numpy()

def overlay_mask(rgb, mask):
    base = np.clip(rgb / 255.0, 0, 1)
    result = base.copy()
    tint = np.zeros_like(base)
    tint[..., 1] = 1.0
    result[mask] = 0.55 * result[mask] + 0.45 * tint[mask]
    return np.clip(result, 0, 1)

def main():
    if not UNET_PATH.is_file():
        raise FileNotFoundError(UNET_PATH)
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    required = {"actual_class", "processed_path", "lung_mask_area_percent"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(required - set(df.columns))}")

    missing = [p for p in df.processed_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image: {missing[0]}")

    print("Loading balanced U-Net...")
    unet = tf.keras.models.load_model(UNET_PATH, compile=False)
    measurements = []

    for class_name in CLASSES:
        subset = df[df.actual_class == class_name].head(5)
        if subset.empty:
            print(f"WARNING: no samples for {class_name}")
            continue

        class_dir = OUT / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(subset), 3, figsize=(12, 3.5 * len(subset)), squeeze=False)

        for i, (_, row) in enumerate(subset.iterrows()):
            path = Path(row.processed_path)
            rgb = read_image(path, 3)
            gray = read_image(path, 1)
            pred = unet(np.expand_dims(gray, 0).astype(np.float32), training=False).numpy()
            if pred.ndim != 4 or pred.shape[-1] < 1:
                raise ValueError(f"Unexpected U-Net output shape: {pred.shape}")
            mask = pred[0, ..., 0] >= THRESHOLD
            overlay = overlay_mask(rgb, mask)
            area = float(mask.mean() * 100)

            panels = [
                (np.clip(rgb / 255.0, 0, 1), "Original X-ray", None),
                (mask, f"U-Net mask ({area:.2f}%)", "gray"),
                (overlay, "Mask overlay", None),
            ]
            for ax, (image, title, cmap) in zip(axes[i], panels):
                ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                ax.set_title(title)
                ax.axis("off")

            single, single_axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, (image, title, cmap) in zip(single_axes, panels):
                ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                ax.set_title(title)
                ax.axis("off")
            single.suptitle(f"{class_name}: {path.stem}", fontsize=9)
            single.tight_layout()
            single.savefig(class_dir / f"{path.stem}_mask_audit.png", dpi=150, bbox_inches="tight")
            plt.close(single)

            measurements.append({
                "actual_class": class_name,
                "processed_path": str(path),
                "csv_mask_area_percent": float(row.lung_mask_area_percent),
                "recomputed_mask_area_percent": area,
                "mask_pixel_count": int(mask.sum()),
                "empty_mask": bool(mask.sum() == 0),
            })
            print(f"{class_name}: {path.name} | mask area={area:.2f}%")

        fig.suptitle(f"{class_name} — Low-area U-Net mask audit")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT / f"{class_name}_contact_sheet.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(measurements).to_csv(OUT / "mask_audit_measurements.csv", index=False)
    print(f"\nAudit complete. Images: {len(measurements)}")
    print(f"Output directory: {OUT}")

if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/58_visualize_low_mask_segmentation_audit.py