
from pathlib import Path
import csv
import numpy as np
import tensorflow as tf

from aiml.config import IMAGE_HEIGHT, IMAGE_WIDTH

# Frozen evaluation configuration
TEST_DIR = Path("data/processed/segmentation/test")
MODEL_PATH = Path(
    "saved_models/unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = Path("results/segmentation")
OUTPUT_CSV = OUTPUT_DIR / "final_test_metrics.csv"

THRESHOLD = 0.30


def load_grayscale(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False,
    )
    return image.numpy().squeeze()


def calculate_metrics(true_mask, predicted_mask):
    true_mask = true_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    intersection = np.logical_and(
        true_mask, predicted_mask
    ).sum()

    union = np.logical_or(
        true_mask, predicted_mask
    ).sum()

    true_area = true_mask.sum()
    predicted_area = predicted_mask.sum()

    denominator = true_area + predicted_area

    dice = (
        2.0 * intersection / denominator
        if denominator > 0 else 1.0
    )

    iou = intersection / union if union > 0 else 1.0

    return float(dice), float(iou)


def main():

    image_dir = TEST_DIR / "images"
    mask_dir = TEST_DIR / "masks"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    image_paths = sorted(image_dir.glob("*.png"))

    mask_by_name = {
        path.name.removesuffix("_mask.png") + ".png": path
        for path in mask_dir.glob("*_mask.png")
    }

    if len(image_paths) != 107:
        raise ValueError(
            f"Expected 107 test images, found {len(image_paths)}"
        )

    if len(mask_by_name) != 107:
        raise ValueError(
            f"Expected 107 test masks, found {len(mask_by_name)}"
        )

    for image_path in image_paths:
        if image_path.name not in mask_by_name:
            raise ValueError(
                f"Missing mask: {image_path.name}"
            )

    print("Loading frozen balanced U-Net...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    rows = []

    for index, image_path in enumerate(image_paths, start=1):

        mask_path = mask_by_name[image_path.name]

        image = load_grayscale(image_path).astype(
            np.float32
        ) / 255.0

        true_mask = load_grayscale(mask_path) >= 128

        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

        if image.shape != expected_shape:
            raise ValueError(
                f"Invalid image shape: {image_path.name}"
            )

        if true_mask.shape != expected_shape:
            raise ValueError(
                f"Invalid mask shape: {mask_path.name}"
            )

        probabilities = model.predict(
            image[None, ..., None],
            verbose=0
        )[0, ..., 0]

        predicted_mask = probabilities >= THRESHOLD

        dice, iou = calculate_metrics(
            true_mask,
            predicted_mask
        )

        source = image_path.name.split("_")[0]

        rows.append({
            "image": image_path.name,
            "source": source,
            "threshold": THRESHOLD,
            "dice": dice,
            "iou": iou,
            "reference_area": float(true_mask.mean()),
            "predicted_area": float(predicted_mask.mean()),
        })

        print(
            f"[{index:03d}/107] {image_path.name}: "
            f"Dice={dice:.4f}, IoU={iou:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    dice_scores = np.array([r["dice"] for r in rows])
    iou_scores = np.array([r["iou"] for r in rows])

    print("\n========== FINAL TEST RESULTS ==========")
    print("Images:", len(rows))
    print("Threshold:", THRESHOLD)
    print("Mean Dice:", round(dice_scores.mean(), 4))
    print("Median Dice:", round(np.median(dice_scores), 4))
    print("Dice Std:", round(dice_scores.std(), 4))
    print("Mean IoU:", round(iou_scores.mean(), 4))
    print("Minimum Dice:", round(dice_scores.min(), 4))
    print("Maximum Dice:", round(dice_scores.max(), 4))

    print("\n========== SOURCE-WISE RESULTS ==========")

    import pandas as pd

    df = pd.DataFrame(rows)

    print(
        df.groupby("source")[["dice", "iou"]]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

    print("\nSaved:", OUTPUT_CSV.resolve())
    print("Test evaluation completed.")


if __name__ == "__main__":
    main()
