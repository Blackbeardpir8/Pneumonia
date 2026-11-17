
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_SIZE, BATCH_SIZE


MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "dataset_source_classifier.keras"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
    / "source_predictions.csv"
)


def load_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return tf.cast(image, tf.float32)


def main():

    print("Loading saved model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    df = pd.read_csv(PREDICTIONS_FILE)

    paths = df["processed_path"].astype(str).tolist()

    labels = (
        df["actual_source"]
        .map({
            "bacterial_viral_dataset": 0,
            "covid19_radiography": 1
        })
        .to_numpy()
    )

    images = np.stack([
        load_image(path).numpy()
        for path in paths
    ])

    print("\nRunning saved model predictions...")

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    ).flatten()

    predicted = (
        probabilities >= 0.5
    ).astype(int)

    saved_predictions = (
        df["predicted_source"]
        .map({
            "bacterial_viral_dataset": 0,
            "covid19_radiography": 1
        })
        .to_numpy()
    )

    accuracy = np.mean(
        predicted == labels
    )

    saved_accuracy = np.mean(
        saved_predictions == labels
    )

    prediction_match = np.mean(
        predicted == saved_predictions
    )

    probability_difference = np.mean(
        np.abs(
            probabilities -
            df["source_probability"].to_numpy()
        )
    )

    print("\nCHECKPOINT CONSISTENCY")
    print("=" * 50)

    print(
        f"Current checkpoint accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Saved prediction accuracy: "
        f"{saved_accuracy * 100:.2f}%"
    )

    print(
        f"Prediction agreement: "
        f"{prediction_match * 100:.2f}%"
    )

    print(
        f"Mean probability difference: "
        f"{probability_difference:.8f}"
    )


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/14_verify_source_classifier_checkpoint.py