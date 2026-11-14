from pathlib import Path
import csv
import numpy as np
import tensorflow as tf

from aiml.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEGMENTATION_DATA_DIR,
    UNET_MODEL_PATH,
)

THRESHOLD = 0.30

OUTPUT_DIR = Path("results/segmentation")
OUTPUT_CSV = OUTPUT_DIR / "balanced_validation_metrics.csv"


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
        if denominator > 0
        else 1.0
    )

    iou = intersection / union if union > 0 else 1.0

    return float(dice), float(iou)


def main():
    data_root = Path(SEGMENTATION_DATA_DIR)

    image_dir = data_root / "val" / "images"
    mask_dir = data_root / "val" / "masks"
    model_path = Path(
    "saved_models/unet_lung_segmentation_balanced.keras"
)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    image_paths = sorted(image_dir.glob("*.png"))

    mask_by_name = {
        path.stem.removesuffix("_mask") + path.suffix: path
        for path in mask_dir.glob("*.png")
    }

    if len(image_paths) != 105:
        raise ValueError(
            f"Expected 105 validation images, "
            f"found {len(image_paths)}"
        )

    if len(mask_by_name) != len(image_paths):
        raise ValueError(
            "Validation image and mask counts do not match."
        )

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = mask_by_name.get(image_path.name)

        if mask_path is None:
            raise ValueError(
                f"Missing mask for {image_path.name}"
            )

        image = load_grayscale(image_path).astype(
            np.float32
        ) / 255.0

        true_mask = load_grayscale(mask_path) >= 128

        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

        if image.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape: {image_path.name} "
                f"{image.shape}"
            )

        if true_mask.shape != expected_shape:
            raise ValueError(
                f"Unexpected mask shape: {mask_path.name} "
                f"{true_mask.shape}"
            )

        probabilities = model.predict(
            image[None, ..., None],
            verbose=0,
        )[0, ..., 0]

        predicted_mask = probabilities >= THRESHOLD

        dice, iou = calculate_metrics(
            true_mask,
            predicted_mask,
        )

        rows.append({
            "image": image_path.name,
            "threshold": THRESHOLD,
            "dice": dice,
            "iou": iou,
            "reference_area": float(true_mask.mean()),
            "predicted_area": float(predicted_mask.mean()),
        })

        print(
            f"[{index:03d}/105] {image_path.name}: "
            f"Dice={dice:.4f}, IoU={iou:.4f}"
        )

    dice_scores = np.array([row["dice"] for row in rows])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "threshold",
                "dice",
                "iou",
                "reference_area",
                "predicted_area",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nVALIDATION SUMMARY")
    print("-" * 40)
    print(f"Images evaluated: {len(rows)}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Median Dice: {np.median(dice_scores):.4f}")
    print(f"Dice standard deviation: {np.std(dice_scores):.4f}")
    print(f"Minimum Dice: {np.min(dice_scores):.4f}")
    print(f"Maximum Dice: {np.max(dice_scores):.4f}")
    print(
        "Images with Dice below 0.50: "
        f"{np.sum(dice_scores < 0.50)}"
    )
    print(f"\nCSV saved to: {OUTPUT_CSV.resolve()}")
    print("The trained model and dataset were not modified.")
    print("The test set was not accessed.")


if __name__ == "__main__":
    main()



# to run python scripts/08_evaluate_validation_segmentation.py