
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

ORIGINAL_MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_balanced.keras"
)

FINETUNED_MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_covid_finetuned_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_final_test"
)

THRESHOLD = 0.30
BATCH_SIZE = 16


# ============================================================
# METRICS
# ============================================================

def calculate_dice(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)

    denominator = (
        np.sum(y_true) +
        np.sum(y_pred)
    )

    return (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )


def calculate_iou(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)

    union = (
        np.sum(y_true) +
        np.sum(y_pred) -
        intersection
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


# ============================================================
# LOAD IMAGE
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
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_path, test_df, model_name):

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print("Model loaded:", model_path)

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

    for index, row in test_df.iterrows():

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
                f"Processed {index + 1}/{len(test_df)} images"
            )

    evaluate_batch()

    del model

    return pd.DataFrame(records)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results_df, model_name, model_path):

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "dataset": "COVID-19 Radiography",
        "split": "test",
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COVID-DOMAIN U-NET FINAL TEST COMPARISON")
    print("=" * 65)

    for path in [
        MANIFEST_PATH,
        ORIGINAL_MODEL_PATH,
        FINETUNED_MODEL_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(MANIFEST_PATH)

    # IMPORTANT: Use the held-out test split only.
    test_df = manifest[
        manifest["split"] == "test"
    ].copy().reset_index(drop=True)

    if len(test_df) != 2250:
        raise ValueError(
            f"Expected 2250 test images, found {len(test_df)}"
        )

    print("\nTest images:", len(test_df))

    print("\nTest class distribution:")
    print(test_df["class_name"].value_counts().to_string())

    # --------------------------------------------------------
    # EVALUATE ORIGINAL MODEL
    # --------------------------------------------------------

    original_results = evaluate_model(
        ORIGINAL_MODEL_PATH,
        test_df,
        "Original Balanced U-Net",
    )

    # --------------------------------------------------------
    # EVALUATE FINE-TUNED MODEL
    # --------------------------------------------------------

    finetuned_results = evaluate_model(
        FINETUNED_MODEL_PATH,
        test_df,
        "COVID-Domain Fine-Tuned U-Net",
    )

    # --------------------------------------------------------
    # VERIFY IMAGE ORDER
    # --------------------------------------------------------

    if not (
        original_results["original_file_name"].tolist()
        ==
        finetuned_results["original_file_name"].tolist()
    ):
        raise ValueError(
            "Image ordering differs between model evaluations."
        )

    # --------------------------------------------------------
    # SAVE PER-IMAGE COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "class_name": original_results["class_name"],
        "original_file_name": original_results["original_file_name"],
        "original_dice": original_results["dice"],
        "finetuned_dice": finetuned_results["dice"],
        "dice_improvement": (
            finetuned_results["dice"] -
            original_results["dice"]
        ),
        "original_iou": original_results["iou"],
        "finetuned_iou": finetuned_results["iou"],
        "iou_improvement": (
            finetuned_results["iou"] -
            original_results["iou"]
        ),
    })

    comparison.to_csv(
        OUTPUT_DIR / "per_image_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------------

    original_summary = create_summary(
        original_results,
        "Original Balanced U-Net",
        ORIGINAL_MODEL_PATH,
    )

    finetuned_summary = create_summary(
        finetuned_results,
        "COVID-Domain Fine-Tuned U-Net",
        FINETUNED_MODEL_PATH,
    )

    summary = {
        "original_model": original_summary,
        "finetuned_model": finetuned_summary,
        "improvement": {
            "mean_dice_absolute_change": (
                finetuned_summary["mean_dice"] -
                original_summary["mean_dice"]
            ),
            "mean_iou_absolute_change": (
                finetuned_summary["mean_iou"] -
                original_summary["mean_iou"]
            ),
        },
    }

    with open(
        OUTPUT_DIR / "overall_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # CLASS-WISE COMPARISON
    # --------------------------------------------------------

    class_comparison = (
        comparison
        .groupby("class_name")
        .agg(
            image_count=("original_dice", "count"),
            original_mean_dice=("original_dice", "mean"),
            finetuned_mean_dice=("finetuned_dice", "mean"),
            original_mean_iou=("original_iou", "mean"),
            finetuned_mean_iou=("finetuned_iou", "mean"),
            mean_dice_improvement=("dice_improvement", "mean"),
            mean_iou_improvement=("iou_improvement", "mean"),
        )
        .reset_index()
    )

    class_comparison.to_csv(
        OUTPUT_DIR / "class_wise_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST COMPARISON")
    print("=" * 65)

    print(
        "Original mean Dice:",
        round(original_summary["mean_dice"], 4),
    )

    print(
        "Fine-tuned mean Dice:",
        round(finetuned_summary["mean_dice"], 4),
    )

    print(
        "Original mean IoU:",
        round(original_summary["mean_iou"], 4),
    )

    print(
        "Fine-tuned mean IoU:",
        round(finetuned_summary["mean_iou"], 4),
    )

    print("\nClass-wise comparison:")
    print(class_comparison.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print("\nFINAL TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/70_compare_unet_models_on_covid_test.py