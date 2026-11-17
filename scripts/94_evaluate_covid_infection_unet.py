
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "test_evaluation"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
THRESHOLD = 0.5

EPSILON = 1e-7


# ============================================================
# 2. COLLECT TEST IMAGE-MASK PAIRS
# ============================================================

def collect_test_pairs():

    test_dir = DATASET_ROOT / "Test" / "COVID-19"

    image_dir = test_dir / "images"
    mask_dir = test_dir / "infection masks"

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file()
    }

    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file()
    }

    missing = set(images) - set(masks)
    extra = set(masks) - set(images)

    if missing or extra:
        raise ValueError(
            f"Test image-mask mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    pairs = [
        (name, str(images[name]), str(masks[name]))
        for name in sorted(images)
    ]

    if not pairs:
        raise ValueError("No test images found.")

    print("Test image-mask pairs:", len(pairs))

    return pairs


# ============================================================
# 3. PREPROCESS TEST DATA
# ============================================================

def load_image_and_mask(image_path, mask_path):

    image = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=1,
        expand_animations=False,
    )

    mask = tf.io.decode_image(
        tf.io.read_file(mask_path),
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask >= 128.0, tf.float32)

    return image, mask


def create_test_dataset(pairs):

    image_paths = [pair[1] for pair in pairs]
    mask_paths = [pair[2] for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    dataset = dataset.map(
        load_image_and_mask,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    intersection = np.sum(y_true * y_pred)

    true_pixels = np.sum(y_true)
    predicted_pixels = np.sum(y_pred)

    union = true_pixels + predicted_pixels - intersection

    true_positive = intersection
    false_positive = predicted_pixels - intersection
    false_negative = true_pixels - intersection

    dice = (
        2.0 * intersection + EPSILON
    ) / (
        true_pixels + predicted_pixels + EPSILON
    )

    iou = (
        intersection + EPSILON
    ) / (
        union + EPSILON
    )

    precision = (
        true_positive + EPSILON
    ) / (
        true_positive + false_positive + EPSILON
    )

    recall = (
        true_positive + EPSILON
    ) / (
        true_positive + false_negative + EPSILON
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_pixels": int(true_pixels),
        "predicted_pixels": int(predicted_pixels),
    }


# ============================================================
# 5. TEST EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("COVID INFECTION U-NET - HELD-OUT TEST EVALUATION")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = collect_test_pairs()

    print("\nLoading saved best model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    test_dataset = create_test_dataset(pairs)

    image_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_pixels = 0
    total_predicted_pixels = 0

    sample_index = 0

    print("\nEvaluating test images...")

    for images, masks in test_dataset:

        predictions = model(
            images,
            training=False,
        ).numpy()

        true_masks = masks.numpy()

        predicted_masks = (
            predictions >= THRESHOLD
        ).astype(np.uint8)

        true_masks = true_masks.astype(np.uint8)

        for i in range(len(predicted_masks)):

            name, _, _ = pairs[sample_index]

            metrics = calculate_metrics(
                true_masks[i],
                predicted_masks[i],
            )

            image_results.append({
                "image": name,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ground_truth_pixels": metrics["true_pixels"],
                "predicted_pixels": metrics["predicted_pixels"],
            })

            total_tp += metrics["true_positive"]
            total_fp += metrics["false_positive"]
            total_fn += metrics["false_negative"]

            total_true_pixels += metrics["true_pixels"]
            total_predicted_pixels += metrics["predicted_pixels"]

            sample_index += 1

        if sample_index % 100 < BATCH_SIZE:
            print(
                f"Processed {sample_index}/{len(pairs)} images"
            )

    # --------------------------------------------------------
    # PER-IMAGE AVERAGE METRICS
    # --------------------------------------------------------

    df = pd.DataFrame(image_results)

    mean_metrics = {
        "mean_dice": float(df["dice"].mean()),
        "mean_iou": float(df["iou"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
    }

    median_metrics = {
        "median_dice": float(df["dice"].median()),
        "median_iou": float(df["iou"].median()),
        "median_precision": float(df["precision"].median()),
        "median_recall": float(df["recall"].median()),
    }

    # --------------------------------------------------------
    # GLOBAL PIXEL-LEVEL METRICS
    # --------------------------------------------------------

    global_dice = (
        2.0 * total_tp + EPSILON
    ) / (
        total_true_pixels + total_predicted_pixels + EPSILON
    )

    global_iou = (
        total_tp + EPSILON
    ) / (
        total_tp + total_fp + total_fn + EPSILON
    )

    global_precision = (
        total_tp + EPSILON
    ) / (
        total_tp + total_fp + EPSILON
    )

    global_recall = (
        total_tp + EPSILON
    ) / (
        total_tp + total_fn + EPSILON
    )

    global_metrics = {
        "global_dice": float(global_dice),
        "global_iou": float(global_iou),
        "global_precision": float(global_precision),
        "global_recall": float(global_recall),
    }

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    csv_path = RESULTS_DIR / "test_image_metrics.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "experiment": "COVID Infection Segmentation",
        "model": MODEL_PATH.name,
        "test_samples": len(df),
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "prediction_threshold": THRESHOLD,
        "mean_per_image_metrics": mean_metrics,
        "median_per_image_metrics": median_metrics,
        "global_pixel_metrics": global_metrics,
        "pixel_counts": {
            "true_positive": int(total_tp),
            "false_positive": int(total_fp),
            "false_negative": int(total_fn),
            "ground_truth_positive": int(total_true_pixels),
            "predicted_positive": int(total_predicted_pixels),
        },
        "note": (
            "Evaluation against COVID-QU-Ex infection annotations. "
            "Metrics do not establish clinical validity."
        ),
    }

    json_path = RESULTS_DIR / "test_metrics_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TEST EVALUATION COMPLETED")
    print("=" * 65)

    print("Test images:", len(df))

    print("\nMEAN PER-IMAGE METRICS")

    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nMEDIAN PER-IMAGE METRICS")

    for key, value in median_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nGLOBAL PIXEL METRICS")

    for key, value in global_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nSaved results:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/94_evaluate_covid_infection_unet.py