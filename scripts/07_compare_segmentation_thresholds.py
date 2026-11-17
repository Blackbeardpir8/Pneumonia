from pathlib import Path

import numpy as np
import tensorflow as tf

from aiml.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEGMENTATION_DATA_DIR,
    UNET_MODEL_PATH,
)


THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


def load_grayscale(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False,
    )
    return image.numpy().squeeze()


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)


def main():
    root = Path(SEGMENTATION_DATA_DIR)
    image_dir = root / "val" / "images"
    mask_dir = root / "val" / "masks"

    model_path = Path(UNET_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    mask_by_name = {
        p.stem.removesuffix("_mask") + p.suffix: p
        for p in mask_dir.glob("*.png")
    }

    image_paths = sorted(image_dir.glob("*.png"))

    if len(image_paths) != 105:
        raise ValueError(
            f"Expected 105 validation images; "
            f"found {len(image_paths)}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = []
    ground_truths = []
    names = []

    for image_path in image_paths:
        mask_path = mask_by_name.get(image_path.name)

        if mask_path is None:
            raise ValueError(
                f"Missing mask for {image_path.name}"
            )

        image = load_grayscale(image_path).astype(
            np.float32
        ) / 255.0

        mask = load_grayscale(mask_path) >= 128

        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"Unexpected image shape: {image_path.name}"
            )

        if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"Unexpected mask shape: {mask_path.name}"
            )

        prediction = model.predict(
            image[None, ..., None],
            verbose=0,
        )[0, ..., 0]

        probabilities.append(prediction)
        ground_truths.append(mask)
        names.append(image_path.name)

    probabilities = np.asarray(probabilities)
    ground_truths = np.asarray(ground_truths, dtype=bool)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("-" * 76)
    print(
        f"{'Threshold':<12}"
        f"{'Mean Dice':<14}"
        f"{'Mean IoU':<14}"
        f"{'Predicted area':<18}"
        f"{'Reference area':<18}"
    )

    for threshold in THRESHOLDS:
        predicted = probabilities >= threshold

        dice_values = []
        iou_values = []

        for true_mask, pred_mask in zip(
            ground_truths,
            predicted,
        ):
            dice_values.append(
                dice_score(true_mask, pred_mask)
            )

            intersection = np.logical_and(
                true_mask,
                pred_mask,
            ).sum()

            union = np.logical_or(
                true_mask,
                pred_mask,
            ).sum()

            iou_values.append(
                intersection / union if union else 1.0
            )

        mean_dice = np.mean(dice_values)
        mean_iou = np.mean(iou_values)
        predicted_area = predicted.mean()
        reference_area = ground_truths.mean()

        print(
            f"{threshold:<12.2f}"
            f"{mean_dice:<14.4f}"
            f"{mean_iou:<14.4f}"
            f"{predicted_area:<18.4f}"
            f"{reference_area:<18.4f}"
        )

    print("\nValidation evaluation completed.")
    print("No model or dataset files were modified.")
    print("The test set was not accessed.")


if __name__ == "__main__":
    main()


# to run this script: 
# python scripts/07_compare_segmentation_thresholds.py