import sys
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "validation_evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "error_analysis"
    / "visualizations"
)

MAX_IMAGES_PER_GROUP = 20
RANDOM_SEED = 42
COLUMNS = 4


# --------------------------------------------------
# CONFUSION GROUPS
# --------------------------------------------------

CONFUSION_GROUPS = [
    ("Viral", "Bacterial"),
    ("Viral", "Viral"),
    ("Normal", "COVID"),
    ("COVID", "Normal"),
    ("Normal", "Normal"),
]


# --------------------------------------------------
# LOAD PREDICTIONS
# --------------------------------------------------

def load_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "processed_path",
        "source_dataset",
        "actual_class",
        "predicted_class",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# --------------------------------------------------
# CREATE CONTACT SHEET
# --------------------------------------------------

def create_contact_sheet(group_df, actual, predicted, source):

    if group_df.empty:
        print(
            f"No images found: {actual} -> {predicted} | {source}"
        )
        return None

    group_df = group_df.sample(
        n=min(len(group_df), MAX_IMAGES_PER_GROUP),
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    image_count = len(group_df)
    rows = (image_count + COLUMNS - 1) // COLUMNS

    fig, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(16, rows * 4)
    )

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, ax in enumerate(axes):

        if index >= image_count:
            ax.axis("off")
            continue

        record = group_df.iloc[index]

        image_path = Path(record["processed_path"])

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            ax.set_title("Image missing")
            ax.axis("off")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)

            confidence_column = f"probability_{predicted}"

            confidence = (
                record[confidence_column]
                if confidence_column in group_df.columns
                else None
            )

            confidence_text = (
                f"\nPredicted confidence: {float(confidence):.3f}"
                if pd.notna(confidence)
                else ""
            )

            ax.set_title(
                f"Actual: {actual}\n"
                f"Predicted: {predicted}"
                f"{confidence_text}",
                fontsize=9
            )

            ax.axis("off")

        except Exception as error:
            ax.set_title(f"Image error\n{error}")
            ax.axis("off")

    fig.suptitle(
        f"{actual} -> {predicted}\nSource: {source}",
        fontsize=14
    )

    plt.tight_layout()

    safe_source = (
        str(source)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    filename = (
        f"{actual}_to_{predicted}_{safe_source}.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

    return group_df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_predictions()

    selected_examples = []

    for actual, predicted in CONFUSION_GROUPS:

        group = df[
            (df["actual_class"] == actual)
            &
            (df["predicted_class"] == predicted)
        ]

        if group.empty:
            print(f"\nNo records: {actual} -> {predicted}")
            continue

        print(
            f"\n{actual} -> {predicted}: "
            f"{len(group)} images"
        )

        for source in sorted(
            group["source_dataset"].dropna().unique()
        ):

            source_group = group[
                group["source_dataset"] == source
            ]

            print(
                f"  Source: {source} | "
                f"Images: {len(source_group)}"
            )

            selected = create_contact_sheet(
                source_group,
                actual,
                predicted,
                source
            )

            if selected is not None:
                selected_examples.append(selected)

    if selected_examples:

        audit_df = pd.concat(
            selected_examples,
            ignore_index=True
        )

        audit_path = OUTPUT_DIR / "selected_examples.csv"

        audit_df.to_csv(
            audit_path,
            index=False
        )

        print(f"\nSelected examples saved: {audit_path}")

    print("\nVisual inspection sheets generated successfully.")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/09_visualize_classification_errors.py

