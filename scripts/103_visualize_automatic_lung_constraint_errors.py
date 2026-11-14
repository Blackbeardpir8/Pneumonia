
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val"
    / "COVID-19"
)

IMAGE_DIR = DATA_DIR / "images"
INFECTION_MASK_DIR = DATA_DIR / "infection masks"
REFERENCE_LUNG_DIR = DATA_DIR / "lung masks"

MODEL_LUNG_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

MODEL_INFECTION_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

PREVIOUS_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "automatic_lung_constraint_per_image.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "automatic_lung_constraint_visual_investigation"
)

LUNG_SIZE = 224
INFECTION_SIZE = 256

LUNG_THRESHOLD = 0.30
INFECTION_THRESHOLD = 0.30

MASK_THRESHOLD = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE HELPERS
# ============================================================

def find_matching_file(folder, stem):

    matches = list(folder.glob(stem + ".*"))

    if not matches:
        raise FileNotFoundError(
            f"File not found for {stem} in {folder}"
        )

    return matches[0]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, size):

    image = Image.open(path).convert("L")

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image / 255.0


def load_mask(path, size):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (size, size),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    return (
        mask >= MASK_THRESHOLD
    ).astype(np.uint8)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def predict_lung_mask(model, image_path):

    image = load_image(
        image_path,
        LUNG_SIZE
    )

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    probability = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    binary_mask = (
        probability >= LUNG_THRESHOLD
    ).astype(np.uint8)

    # Convert 224x224 lung prediction to 256x256.
    mask_image = Image.fromarray(
        binary_mask * 255
    )

    mask_image = mask_image.resize(
        (INFECTION_SIZE, INFECTION_SIZE),
        Image.Resampling.NEAREST
    )

    return (
        np.asarray(mask_image) >= 128
    ).astype(np.uint8)


def predict_infection_probability(model, image_path):

    image = load_image(
        image_path,
        INFECTION_SIZE
    )

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    probability = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    return probability


# ============================================================
# METRICS
# ============================================================

def calculate_dice(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    tp = np.logical_and(
        reference,
        prediction
    ).sum()

    fp = np.logical_and(
        ~reference,
        prediction
    ).sum()

    fn = np.logical_and(
        reference,
        ~prediction
    ).sum()

    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        return 1.0

    return float((2 * tp) / denominator)


def calculate_lung_error_counts(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    true_positive = np.logical_and(
        reference,
        prediction
    ).sum()

    false_positive = np.logical_and(
        ~reference,
        prediction
    ).sum()

    false_negative = np.logical_and(
        reference,
        ~prediction
    ).sum()

    return {
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative)
    }


# ============================================================
# ERROR MAPS
# ============================================================

def create_binary_error_map(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    error_map = np.zeros(
        (*reference.shape, 3),
        dtype=np.float32
    )

    # Correct overlap = green
    true_positive = np.logical_and(
        reference,
        prediction
    )

    # Extra predicted area = red
    false_positive = np.logical_and(
        ~reference,
        prediction
    )

    # Missed reference area = blue
    false_negative = np.logical_and(
        reference,
        ~prediction
    )

    error_map[true_positive] = [0.0, 1.0, 0.0]
    error_map[false_positive] = [1.0, 0.0, 0.0]
    error_map[false_negative] = [0.0, 0.0, 1.0]

    return error_map


def create_lost_infection_map(reference, predicted_lung):

    reference = reference.astype(bool)
    predicted_lung = predicted_lung.astype(bool)

    lost = np.logical_and(
        reference,
        ~predicted_lung
    )

    result = np.zeros(
        (*reference.shape, 3),
        dtype=np.float32
    )

    result[reference] = [0.0, 0.6, 0.0]

    # Infection reference pixels excluded by predicted lung mask.
    result[lost] = [1.0, 0.0, 0.0]

    return result


# ============================================================
# SELECT FOUR DIFFERENT CASES
# ============================================================

def select_representative_cases(results_df):

    selected = []

    used_names = set()

    def add_case(row, case_type):

        image_name = row["image_name"]

        if image_name in used_names:
            return False

        used_names.add(image_name)

        selected.append({
            "image_name": image_name,
            "case_type": case_type
        })

        return True

    # --------------------------------------------------------
    # CASE 1: MOST MISSED REFERENCE LUNG PIXELS
    # --------------------------------------------------------

    lung_failure = results_df.sort_values(
        "lung_fn",
        ascending=False
    )

    for _, row in lung_failure.iterrows():

        if add_case(
            row,
            "MOST_LUNG_PIXELS_MISSED"
        ):
            break

    # --------------------------------------------------------
    # CASE 2: MOST VALID INFECTION PIXELS LOST
    # --------------------------------------------------------

    infection_loss = results_df.sort_values(
        "lost_reference_infection_pixels",
        ascending=False
    )

    for _, row in infection_loss.iterrows():

        if add_case(
            row,
            "MOST_REFERENCE_INFECTION_LOST"
        ):
            break

    # --------------------------------------------------------
    # CASE 3: MOST FALSE POSITIVES REMOVED
    # --------------------------------------------------------

    fp_removal = results_df.sort_values(
        "false_positives_removed",
        ascending=False
    )

    for _, row in fp_removal.iterrows():

        if add_case(
            row,
            "MOST_FALSE_POSITIVES_REMOVED"
        ):
            break

    # --------------------------------------------------------
    # CASE 4: MOST SIMILAR CONSTRAINED/UNCONSTRAINED DICE
    # --------------------------------------------------------

    similarity = results_df.sort_values(
        "dice_difference_absolute",
        ascending=True
    )

    for _, row in similarity.iterrows():

        if add_case(
            row,
            "MOST_SIMILAR_INFECTION_RESULTS"
        ):
            break

    return selected


# ============================================================
# VISUALIZE ONE CASE
# ============================================================

def visualize_case(
    image_path,
    case_type,
    lung_reference,
    lung_prediction,
    infection_reference,
    unconstrained_prediction,
    reference_constrained_prediction,
    predicted_constrained_prediction,
    output_path
):

    image = load_image(
        image_path,
        INFECTION_SIZE
    )

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(22, 10)
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    axes[0].imshow(
        image,
        cmap="gray"
    )
    axes[0].set_title("Original X-ray")

    axes[1].imshow(
        lung_reference,
        cmap="gray"
    )
    axes[1].set_title("Reference Lung Mask")

    axes[2].imshow(
        lung_prediction,
        cmap="gray"
    )
    axes[2].set_title("Predicted Lung Mask")

    axes[3].imshow(
        create_binary_error_map(
            lung_reference,
            lung_prediction
        )
    )
    axes[3].set_title(
        "Lung Error Map\nGreen: TP | Red: FP | Blue: FN"
    )

    axes[4].imshow(
        create_lost_infection_map(
            infection_reference,
            lung_prediction
        )
    )
    axes[4].set_title(
        "Reference Infection Excluded\nRed: Lost infection"
    )

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    axes[5].imshow(
        infection_reference,
        cmap="gray"
    )
    axes[5].set_title("Reference Infection")

    axes[6].imshow(
        unconstrained_prediction,
        cmap="gray"
    )
    axes[6].set_title("Unconstrained Prediction")

    axes[7].imshow(
        reference_constrained_prediction,
        cmap="gray"
    )
    axes[7].set_title("Reference Lung Constrained")

    axes[8].imshow(
        predicted_constrained_prediction,
        cmap="gray"
    )
    axes[8].set_title("Predicted Lung Constrained")

    axes[9].imshow(
        create_binary_error_map(
            infection_reference,
            predicted_constrained_prediction
        )
    )
    axes[9].set_title(
        "Automatic Constraint Error\nGreen: TP | Red: FP | Blue: FN"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{Path(image_path).name}\n{case_type}",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AUTOMATIC LUNG CONSTRAINT VISUAL ERROR INVESTIGATION")
    print("=" * 80)

    if not PREVIOUS_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Previous results not found: {PREVIOUS_RESULTS_PATH}"
        )

    if not MODEL_LUNG_PATH.exists():
        raise FileNotFoundError(
            f"Lung model not found: {MODEL_LUNG_PATH}"
        )

    if not MODEL_INFECTION_PATH.exists():
        raise FileNotFoundError(
            f"Infection model not found: {MODEL_INFECTION_PATH}"
        )

    # --------------------------------------------------------
    # LOAD PREVIOUS RESULTS
    # --------------------------------------------------------

    previous_df = pd.read_csv(
        PREVIOUS_RESULTS_PATH
    )

    required_columns = [
        "image_name",
        "lung_dice",
        "lung_fp",
        "lung_fn",
        "unconstrained_fp",
        "predicted_lung_constrained_fp",
        "unconstrained_dice",
        "predicted_lung_constrained_dice"
    ]

    # The Script 102 output does not contain lung FP/FN
    # columns in the original per-image CSV.
    # Recalculate them below from the actual masks.
    for column in required_columns:

        if column not in previous_df.columns:
            if column in ["lung_fp", "lung_fn"]:
                previous_df[column] = 0
            else:
                raise ValueError(
                    f"Required column missing: {column}"
                )

    print(
        f"\nPrevious validation records: {len(previous_df)}"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("\nLoading lung U-Net...")

    lung_model = tf.keras.models.load_model(
        MODEL_LUNG_PATH,
        compile=False
    )

    print("Lung U-Net loaded.")

    print("\nLoading infection U-Net...")

    infection_model = tf.keras.models.load_model(
        MODEL_INFECTION_PATH,
        compile=False
    )

    print("Infection U-Net loaded.")

    # --------------------------------------------------------
    # RECOMPUTE CASE-SPECIFIC ERROR INFORMATION
    # --------------------------------------------------------

    analysis_rows = []

    for index, row in previous_df.iterrows():

        image_name = row["image_name"]

        image_path = IMAGE_DIR / image_name

        stem = Path(image_name).stem

        infection_path = find_matching_file(
            INFECTION_MASK_DIR,
            stem
        )

        reference_lung_path = find_matching_file(
            REFERENCE_LUNG_DIR,
            stem
        )

        reference_lung = load_mask(
            reference_lung_path,
            INFECTION_SIZE
        )

        reference_infection = load_mask(
            infection_path,
            INFECTION_SIZE
        )

        predicted_lung = predict_lung_mask(
            lung_model,
            image_path
        )

        lung_errors = calculate_lung_error_counts(
            reference_lung,
            predicted_lung
        )

        lost_infection_pixels = int(
            np.logical_and(
                reference_infection == 1,
                predicted_lung == 0
            ).sum()
        )

        false_positives_removed = (
            int(row["unconstrained_fp"])
            - int(row["predicted_lung_constrained_fp"])
        )

        dice_difference = (
            float(row["predicted_lung_constrained_dice"])
            - float(row["unconstrained_dice"])
        )

        analysis_rows.append({

            "image_name": image_name,

            "lung_dice":
                float(row["lung_dice"]),

            "lung_fp":
                lung_errors["fp"],

            "lung_fn":
                lung_errors["fn"],

            "lost_reference_infection_pixels":
                lost_infection_pixels,

            "false_positives_removed":
                false_positives_removed,

            "unconstrained_dice":
                float(row["unconstrained_dice"]),

            "predicted_lung_constrained_dice":
                float(row["predicted_lung_constrained_dice"]),

            "dice_difference":
                dice_difference,

            "dice_difference_absolute":
                abs(dice_difference)

        })

        if (index + 1) % 100 == 0 or index + 1 == len(previous_df):

            print(
                f"Analyzed {index + 1}/{len(previous_df)} images"
            )

    analysis_df = pd.DataFrame(
        analysis_rows
    )

    # --------------------------------------------------------
    # SELECT CASES
    # --------------------------------------------------------

    selected_cases = select_representative_cases(
        analysis_df
    )

    if len(selected_cases) < 4:
        print(
            "\nWARNING: Fewer than four unique cases were selected."
        )

    print("\nSelected cases:")

    for case in selected_cases:

        print(
            f"{case['case_type']}: {case['image_name']}"
        )

    # --------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # --------------------------------------------------------

    visualization_rows = []

    for case in selected_cases:

        image_name = case["image_name"]

        case_type = case["case_type"]

        stem = Path(image_name).stem

        image_path = IMAGE_DIR / image_name

        infection_path = find_matching_file(
            INFECTION_MASK_DIR,
            stem
        )

        reference_lung_path = find_matching_file(
            REFERENCE_LUNG_DIR,
            stem
        )

        reference_lung = load_mask(
            reference_lung_path,
            INFECTION_SIZE
        )

        infection_reference = load_mask(
            infection_path,
            INFECTION_SIZE
        )

        predicted_lung = predict_lung_mask(
            lung_model,
            image_path
        )

        probability = predict_infection_probability(
            infection_model,
            image_path
        )

        unconstrained = (
            probability >= INFECTION_THRESHOLD
        ).astype(np.uint8)

        reference_constrained = (
            unconstrained * reference_lung
        ).astype(np.uint8)

        predicted_constrained = (
            unconstrained * predicted_lung
        ).astype(np.uint8)

        unconstrained_dice = calculate_dice(
            infection_reference,
            unconstrained
        )

        reference_constrained_dice = calculate_dice(
            infection_reference,
            reference_constrained
        )

        predicted_constrained_dice = calculate_dice(
            infection_reference,
            predicted_constrained
        )

        output_path = (
            OUTPUT_DIR
            / f"{stem}_{case_type.lower()}.png"
        )

        visualize_case(
            image_path,
            case_type,
            reference_lung,
            predicted_lung,
            infection_reference,
            unconstrained,
            reference_constrained,
            predicted_constrained,
            output_path
        )

        visualization_rows.append({

            "image_name": image_name,

            "case_type": case_type,

            "unconstrained_dice":
                unconstrained_dice,

            "reference_constrained_dice":
                reference_constrained_dice,

            "predicted_constrained_dice":
                predicted_constrained_dice,

            "lung_dice":
                calculate_dice(
                    reference_lung,
                    predicted_lung
                ),

            "visualization_path":
                str(output_path)

        })

        print(
            f"\nSaved: {output_path}"
        )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    analysis_path = (
        OUTPUT_DIR
        / "automatic_lung_error_investigation.csv"
    )

    analysis_df.to_csv(
        analysis_path,
        index=False
    )

    selected_path = (
        OUTPUT_DIR
        / "selected_visual_cases.csv"
    )

    pd.DataFrame(
        visualization_rows
    ).to_csv(
        selected_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("VISUAL ERROR INVESTIGATION COMPLETED")
    print("=" * 80)

    print("\nSelected case summary:")

    print(
        pd.DataFrame(
            visualization_rows
        ).to_string(index=False)
    )

    print("\nSaved directory:")
    print(OUTPUT_DIR)

    print("\nAnalysis CSV:")
    print(analysis_path)

    print("\nSelected cases CSV:")
    print(selected_path)


if __name__ == "__main__":
    main()
