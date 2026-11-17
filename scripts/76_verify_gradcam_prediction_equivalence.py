
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "final_test_evaluation"
    / "original_threshold_0_250_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "gradcam_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
NUM_IMAGES = 6
RANDOM_STATE = 42
VIRAL_THRESHOLD = 0.25

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

PROBABILITY_COLUMNS = [
    "prob_normal",
    "prob_viral",
    "prob_bacterial",
    "prob_covid",
]


# ============================================================
# LOAD MODEL AND EXISTING PREDICTIONS
# ============================================================

print("Loading original classifier...")

classifier = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Loading existing test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)


# ============================================================
# RECONSTRUCT CLASSIFIER INFERENCE PATH
# ============================================================

augmentation = classifier.get_layer("data_augmentation")
backbone = classifier.get_layer("efficientnetb0")

gap_layer = classifier.get_layer("global_average_pooling")
dropout_layer = classifier.get_layer("classification_dropout")
output_layer = classifier.get_layer("classification_output")


def reconstructed_predict(image_batch):

    augmented_images = augmentation(
        image_batch,
        training=False
    )

    features = backbone(
        augmented_images,
        training=False
    )

    features = gap_layer(features)

    features = dropout_layer(
        features,
        training=False
    )

    predictions = output_layer(features)

    return predictions


# ============================================================
# SELECT SAME 18 IMAGES AS SCRIPT 75
# ============================================================

groups = {
    "normal_to_covid": {
        "source": "covid19_radiography",
        "actual": "Normal",
        "predicted": "COVID",
    },

    "viral_to_bacterial": {
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Bacterial",
    },

    "correct_viral": {
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Viral",
    },
}

selected_groups = []

for group_name, config in groups.items():

    subset = df[
        (df["source_dataset"] == config["source"])
        & (df["actual_class"] == config["actual"])
        & (
            df["threshold_predicted_class"]
            == config["predicted"]
        )
    ].copy()

    subset = subset.sample(
        n=min(NUM_IMAGES, len(subset)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    subset["analysis_group"] = group_name

    selected_groups.append(subset)

selected_df = pd.concat(
    selected_groups,
    ignore_index=True
)

print("\nSelected images:", len(selected_df))


# ============================================================
# IMAGE PATH AND PREPROCESSING
# ============================================================

def resolve_image_path(path_value):

    path = Path(str(path_value))

    if path.exists():
        return path

    project_path = PROJECT_ROOT / path

    if project_path.exists():
        return project_path

    raise FileNotFoundError(
        f"Image not found: {path_value}"
    )


def load_image(path_value):

    image_path = resolve_image_path(path_value)

    image = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    image = tf.cast(image, tf.float32)

    return image


# ============================================================
# COMPARE PROBABILITIES
# ============================================================

results = []

for index, record in selected_df.iterrows():

    print(
        f"Checking image {index + 1}/{len(selected_df)}"
    )

    image = load_image(record["processed_path"])

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    # Original complete classifier
    original_probabilities = classifier(
        image_batch,
        training=False
    ).numpy()[0]

    # Reconstructed inference path
    reconstructed_probabilities = reconstructed_predict(
        image_batch
    ).numpy()[0]

    saved_probabilities = record[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=np.float64)

    original_difference = np.abs(
        original_probabilities - saved_probabilities
    )

    reconstruction_difference = np.abs(
        reconstructed_probabilities - original_probabilities
    )

    original_argmax = int(
        np.argmax(original_probabilities)
    )

    reconstructed_argmax = int(
        np.argmax(reconstructed_probabilities)
    )

    original_viral_decision = (
        1
        if original_probabilities[1] >= VIRAL_THRESHOLD
        else int(
            np.argmax(
                np.delete(original_probabilities, 1)
            )
        )
    )

    # Map non-Viral argmax back to original class index
    if original_probabilities[1] < VIRAL_THRESHOLD:

        non_viral_indices = [0, 2, 3]

        original_viral_decision = non_viral_indices[
            int(np.argmax(
                original_probabilities[non_viral_indices]
            ))
        ]

    if reconstructed_probabilities[1] >= VIRAL_THRESHOLD:

        reconstructed_viral_decision = 1

    else:

        non_viral_indices = [0, 2, 3]

        reconstructed_viral_decision = non_viral_indices[
            int(np.argmax(
                reconstructed_probabilities[non_viral_indices]
            ))
        ]

    result = {
        "analysis_group": record["analysis_group"],
        "image": Path(
            str(record["processed_path"])
        ).stem,

        "original_csv_max_difference": float(
            np.max(original_difference)
        ),

        "reconstruction_max_difference": float(
            np.max(reconstruction_difference)
        ),

        "reconstruction_mean_difference": float(
            np.mean(reconstruction_difference)
        ),

        "argmax_prediction_agreement": (
            original_argmax == reconstructed_argmax
        ),

        "threshold_prediction_agreement": (
            original_viral_decision
            == reconstructed_viral_decision
        ),

        "original_predicted_class": CLASS_NAMES[
            original_viral_decision
        ],

        "reconstructed_predicted_class": CLASS_NAMES[
            reconstructed_viral_decision
        ],
    }

    for class_index, class_name in enumerate(CLASS_NAMES):

        result[f"original_{class_name.lower()}"] = float(
            original_probabilities[class_index]
        )

        result[f"reconstructed_{class_name.lower()}"] = float(
            reconstructed_probabilities[class_index]
        )

    results.append(result)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_csv = (
    OUTPUT_DIR
    / "prediction_equivalence_results.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

max_csv_difference = results_df[
    "original_csv_max_difference"
].max()

max_reconstruction_difference = results_df[
    "reconstruction_max_difference"
].max()

mean_reconstruction_difference = results_df[
    "reconstruction_mean_difference"
].mean()

argmax_agreement = results_df[
    "argmax_prediction_agreement"
].mean() * 100

threshold_agreement = results_df[
    "threshold_prediction_agreement"
].mean() * 100

print("\n" + "=" * 70)
print("GRAD-CAM PREDICTION EQUIVALENCE VERIFICATION")
print("=" * 70)

print("Images checked:", len(results_df))

print(
    "Maximum original CSV difference:",
    f"{max_csv_difference:.10f}"
)

print(
    "Maximum reconstruction difference:",
    f"{max_reconstruction_difference:.10f}"
)

print(
    "Mean reconstruction difference:",
    f"{mean_reconstruction_difference:.10f}"
)

print(
    "Argmax prediction agreement:",
    f"{argmax_agreement:.2f}%"
)

print(
    "Threshold prediction agreement:",
    f"{threshold_agreement:.2f}%"
)

print("\nGroup-wise reconstruction difference:")

print(
    results_df.groupby("analysis_group")[
        "reconstruction_max_difference"
    ].agg(["count", "mean", "max"]).to_string()
)

print("\nSaved results:")
print(output_csv)

print("\nVerification completed.")


# to run this script, use the following command:
# python scripts/76_verify_gradcam_prediction_equivalence.py