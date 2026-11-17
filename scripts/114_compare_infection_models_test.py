
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

TEST_DIR = DATASET_ROOT / "Test" / "COVID-19"

IMAGE_DIR = TEST_DIR / "images"
MASK_DIR = TEST_DIR / "infection masks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "final_loss_function_test_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "original_bce_dice": (
        PROJECT_ROOT
        / "saved_models"
        / "unet_covid_infection_segmentation_best.keras"
    ),
    "weighted_bce_dice": (
        PROJECT_ROOT
        / "saved_models"
        / "infection_loss_experiments"
        / "weighted_bce_dice"
        / "best_model.keras"
    ),
}

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BATCH_SIZE = 8
PREDICTION_THRESHOLD = 0.5

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


# ============================================================
# 2. COLLECT TEST IMAGE-MASK PAIRS
# ============================================================

def collect_test_pairs():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Test mask directory not found:\n{MASK_DIR}"
        )

    images = {
        path.stem: path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    masks = {
        path.stem: path
        for path in MASK_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    missing_masks = set(images) - set(masks)
    extra_masks = set(masks) - set(images)

    if missing_masks or extra_masks:
        raise ValueError(
            "Test image-mask mismatch.\n"
            f"Images without masks: {len(missing_masks)}\n"
            f"Masks without images: {len(extra_masks)}"
        )

    pairs = [
        (images[name], masks[name])
        for name in sorted(images)
    ]

    if not pairs:
        raise ValueError("No test image-mask pairs found.")

    print("Matched test pairs:", len(pairs))

    return pairs


# ============================================================
# 3. IMAGE AND MASK PREPROCESSING
# ============================================================

def load_image_and_mask(image_path, mask_path):

    image_bytes = tf.io.read_file(image_path)
    mask_bytes = tf.io.read_file(mask_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    mask = tf.io.decode_image(
        mask_bytes,
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


# ============================================================
# 4. BUILD TEST DATASET
# ============================================================

def create_test_dataset(pairs):

    image_paths = [
        str(image_path)
        for image_path, _ in pairs
    ]

    mask_paths = [
        str(mask_path)
        for _, mask_path in pairs
    ]

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
# 5. CALCULATE SEGMENTATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.uint8,
    ).reshape(-1)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    dice = (
        (2.0 * tp + 1e-6)
        / (2.0 * tp + fp + fn + 1e-6)
    )

    iou = (
        (tp + 1e-6)
        / (tp + fp + fn + 1e-6)
    )

    precision = (
        (tp + 1e-6)
        / (tp + fp + 1e-6)
    )

    recall = (
        (tp + 1e-6)
        / (tp + fn + 1e-6)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 6. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    dataset,
    image_names,
):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print("\n" + "=" * 65)
    print("Evaluating:", model_name)
    print("=" * 65)

    print("Loading model:", model_path)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    per_image_results = []

    global_tp = 0
    global_fp = 0
    global_fn = 0

    image_index = 0

    for batch_images, batch_masks in dataset:

        predictions = model.predict(
            batch_images,
            verbose=0,
        )

        predictions = (
            predictions >= PREDICTION_THRESHOLD
        ).astype(np.uint8)

        true_masks = (
            batch_masks.numpy() >= 0.5
        ).astype(np.uint8)

        for index in range(len(predictions)):

            y_true = true_masks[index]
            y_pred = predictions[index]

            metrics = calculate_metrics(
                y_true,
                y_pred,
            )

            global_tp += metrics["tp"]
            global_fp += metrics["fp"]
            global_fn += metrics["fn"]

            reference_area = float(
                np.mean(y_true)
            )

            predicted_area = float(
                np.mean(y_pred)
            )

            per_image_results.append({
                "image": image_names[image_index],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "reference_area": reference_area,
                "predicted_area": predicted_area,
            })

            image_index += 1

        print(
            f"Processed {image_index}/{len(image_names)} images",
            end="\r",
        )

    print()

    if image_index != len(image_names):
        raise RuntimeError(
            "Prediction count does not match test image count."
        )

    results_df = pd.DataFrame(
        per_image_results
    )

    # Mean of individual image metrics.
    mean_metrics = {
        f"mean_per_image_{column}": float(
            results_df[column].mean()
        )
        for column in [
            "dice",
            "iou",
            "precision",
            "recall",
        ]
    }

    # Global pixel-level metrics.
    global_metrics = calculate_metrics(
        np.array([global_tp + global_fn]),
        np.array([global_tp]),
    )

    # Calculate global metrics directly from pixel counts.
    global_dice = (
        2.0 * global_tp + 1e-6
    ) / (
        2.0 * global_tp
        + global_fp
        + global_fn
        + 1e-6
    )

    global_iou = (
        global_tp + 1e-6
    ) / (
        global_tp
        + global_fp
        + global_fn
        + 1e-6
    )

    global_precision = (
        global_tp + 1e-6
    ) / (
        global_tp
        + global_fp
        + 1e-6
    )

    global_recall = (
        global_tp + 1e-6
    ) / (
        global_tp
        + global_fn
        + 1e-6
    )

    summary = {
        "model": model_name,
        "test_images": len(results_df),
        "threshold": PREDICTION_THRESHOLD,
        **mean_metrics,
        "global_pixel_dice": float(global_dice),
        "global_pixel_iou": float(global_iou),
        "global_pixel_precision": float(global_precision),
        "global_pixel_recall": float(global_recall),
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_fn": global_fn,
    }

    print("\nTest results:")

    for key, value in summary.items():

        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    del model

    return results_df, summary


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL INFECTION SEGMENTATION TEST COMPARISON")
    print("=" * 70)

    pairs = collect_test_pairs()

    image_names = [
        image_path.name
        for image_path, _ in pairs
    ]

    test_dataset = create_test_dataset(pairs)

    all_summaries = []

    for model_name, model_path in MODEL_PATHS.items():

        results_df, summary = evaluate_model(
            model_name,
            model_path,
            test_dataset,
            image_names,
        )

        results_path = (
            OUTPUT_DIR
            / f"{model_name}_per_image_results.csv"
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        all_summaries.append(summary)

        print("\nSaved:", results_path)

    # Final model comparison.
    comparison_df = pd.DataFrame(
        all_summaries
    )

    comparison_path = (
        OUTPUT_DIR
        / "final_test_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    json_path = (
        OUTPUT_DIR
        / "final_test_model_comparison.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_summaries,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "test_images",
        "mean_per_image_dice",
        "mean_per_image_iou",
        "mean_per_image_precision",
        "mean_per_image_recall",
        "global_pixel_dice",
        "global_pixel_iou",
    ]

    print(
        comparison_df[display_columns].to_string(
            index=False
        )
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nSaved comparison:")
    print(comparison_path)

    print(
        "\nIMPORTANT: The test set was used only for final "
        "evaluation. No model training or threshold tuning "
        "was performed."
    )


if __name__ == "__main__":
    main()

# # to run this script, use the following command:
# python scripts/105_evaluate_selective_lung_constraint.py