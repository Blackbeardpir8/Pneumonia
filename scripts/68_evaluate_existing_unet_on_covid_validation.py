
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from aiml.config import SAVED_MODELS_DIR

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT /
    "data/processed/segmentation_covid/manifest.csv"
)

MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_baseline"
)

THRESHOLD = 0.30
BATCH_SIZE = 16


# ============================================================
# METRICS
# ============================================================

def calculate_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)

    return (
        (2.0 * intersection + 1e-6) /
        (denominator + 1e-6)
    )


def calculate_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (
        (intersection + 1e-6) /
        (union + 1e-6)
    )


# ============================================================
# LOAD IMAGE AND MASK
# ============================================================

def load_grayscale_image(path):
    image = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False,
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("EXISTING U-NET: COVID DOMAIN VALIDATION")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)

    # Use validation split only.
    val_df = manifest[
        manifest["split"] == "val"
    ].copy()

    val_df = val_df.reset_index(drop=True)

    print("Validation images:", len(val_df))

    print("\nLoading existing balanced U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    records = []

    images = []
    masks = []
    batch_rows = []

    def evaluate_batch():

        if not images:
            return

        X = np.asarray(images, dtype=np.float32)
        y_true_batch = np.asarray(masks, dtype=np.float32)

        predictions = model.predict(
            X,
            verbose=0,
        )

        for i, row in enumerate(batch_rows):

            y_true = y_true_batch[i, :, :, 0]

            y_pred = (
                predictions[i, :, :, 0] >= THRESHOLD
            ).astype(np.float32)

            dice = calculate_dice(y_true, y_pred)
            iou = calculate_iou(y_true, y_pred)

            records.append({
                "class_name": row["class_name"],
                "original_file_name": row["original_file_name"],
                "dice": float(dice),
                "iou": float(iou),
                "true_mask_area_percent": float(
                    np.mean(y_true) * 100
                ),
                "predicted_mask_area_percent": float(
                    np.mean(y_pred) * 100
                ),
            })

        images.clear()
        masks.clear()
        batch_rows.clear()

    # Process validation images in batches.
    for index, row in val_df.iterrows():

        image_path = Path(row["processed_image_path"])
        mask_path = Path(row["processed_mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = load_grayscale_image(image_path)
        mask = load_grayscale_image(mask_path)

        mask = (mask >= 0.5).astype(np.float32)

        images.append(image)
        masks.append(mask)
        batch_rows.append(row)

        if len(images) == BATCH_SIZE:
            evaluate_batch()

        if (index + 1) % 250 == 0:
            print(
                f"Loaded {index + 1}/{len(val_df)} images"
            )

    evaluate_batch()

    # ========================================================
    # SAVE PER-IMAGE RESULTS
    # ========================================================

    results_df = pd.DataFrame(records)

    results_path = OUTPUT_DIR / "per_image_metrics.csv"
    results_df.to_csv(results_path, index=False)

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    overall = {
        "model": str(MODEL_PATH),
        "dataset": "COVID-19 Radiography validation",
        "split": "val",
        "images_evaluated": len(results_df),
        "threshold": THRESHOLD,
        "mean_dice": float(results_df["dice"].mean()),
        "median_dice": float(results_df["dice"].median()),
        "std_dice": float(results_df["dice"].std()),
        "mean_iou": float(results_df["iou"].mean()),
        "median_iou": float(results_df["iou"].median()),
        "images_dice_below_0_5": int(
            (results_df["dice"] < 0.5).sum()
        ),
    }

    with open(
        OUTPUT_DIR / "overall_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(overall, file, indent=4)

    # ========================================================
    # CLASS-WISE SUMMARY
    # ========================================================

    class_summary = (
        results_df
        .groupby("class_name")
        .agg(
            image_count=("dice", "count"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            mean_iou=("iou", "mean"),
            images_dice_below_0_5=(
                "dice",
                lambda x: int((x < 0.5).sum()),
            ),
        )
        .reset_index()
    )

    class_summary.to_csv(
        OUTPUT_DIR / "class_wise_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("COVID DOMAIN BASELINE RESULTS")
    print("=" * 65)

    for key, value in overall.items():
        print(f"{key}: {value}")

    print("\nCLASS-WISE RESULTS")
    print(class_summary.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
