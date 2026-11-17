
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    / "error_visual_analysis"
)

NUM_IMAGES = 20
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD EXISTING TEST PREDICTIONS
# =========================================================

print("Loading existing classifier test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

required_columns = [
    "processed_path",
    "actual_class",
    "threshold_predicted_class",
    "source_dataset",
    "prob_normal",
    "prob_viral",
    "prob_bacterial",
    "prob_covid",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Total predictions loaded: {len(df)}")


# =========================================================
# IMAGE PATH RESOLUTION
# =========================================================

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


# =========================================================
# SELECT REPRESENTATIVE EXAMPLES
# =========================================================

# 1. Normal images incorrectly classified as COVID
normal_to_covid = df[
    (df["source_dataset"] == "covid19_radiography")
    & (df["actual_class"] == "Normal")
    & (df["threshold_predicted_class"] == "COVID")
].copy()


# 2. Viral images incorrectly classified as Bacterial
viral_to_bacterial = df[
    (df["source_dataset"] == "bacterial_viral_dataset")
    & (df["actual_class"] == "Viral")
    & (df["threshold_predicted_class"] == "Bacterial")
].copy()


# 3. Correctly classified Viral images from the same source
correct_viral = df[
    (df["source_dataset"] == "bacterial_viral_dataset")
    & (df["actual_class"] == "Viral")
    & (df["threshold_predicted_class"] == "Viral")
].copy()


# =========================================================
# RANDOM SAMPLING
# =========================================================

def select_samples(data, description):

    print(f"\n{description}: {len(data)} available images")

    if len(data) == 0:
        print("WARNING: No matching images found.")
        return data

    count = min(NUM_IMAGES, len(data))

    selected = data.sample(
        n=count,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"Selected: {len(selected)} images")

    return selected


normal_to_covid = select_samples(
    normal_to_covid,
    "Normal -> COVID errors"
)

viral_to_bacterial = select_samples(
    viral_to_bacterial,
    "Viral -> Bacterial errors"
)

correct_viral = select_samples(
    correct_viral,
    "Correctly classified Viral images"
)


# =========================================================
# GENERATE CONTACT SHEET
# =========================================================

def generate_contact_sheet(data, title, filename):

    if data.empty:
        print(f"Skipping {filename}: no images available.")
        return

    columns = 4
    rows = (len(data) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4)
    )

    axes = axes.flatten()

    for index, (_, record) in enumerate(data.iterrows()):

        image_path = resolve_image_path(
            record["processed_path"]
        )

        image = Image.open(image_path).convert("L")

        axes[index].imshow(image, cmap="gray")
        axes[index].axis("off")

        axes[index].set_title(
            f"Actual: {record['actual_class']}\n"
            f"Predicted: {record['threshold_predicted_class']}\n"
            f"Normal: {record['prob_normal']:.2%} | "
            f"Viral: {record['prob_viral']:.2%}\n"
            f"Bacterial: {record['prob_bacterial']:.2%} | "
            f"COVID: {record['prob_covid']:.2%}",
            fontsize=8
        )

    # Hide unused cells
    for index in range(len(data), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# =========================================================
# SAVE SELECTED IMAGE DETAILS
# =========================================================

normal_to_covid.to_csv(
    OUTPUT_DIR / "normal_to_covid_selected.csv",
    index=False
)

viral_to_bacterial.to_csv(
    OUTPUT_DIR / "viral_to_bacterial_selected.csv",
    index=False
)

correct_viral.to_csv(
    OUTPUT_DIR / "correct_viral_selected.csv",
    index=False
)


# =========================================================
# CREATE THREE CONTACT SHEETS
# =========================================================

generate_contact_sheet(
    normal_to_covid,
    "Normal Images Misclassified as COVID",
    "normal_to_covid_contact_sheet.png"
)

generate_contact_sheet(
    viral_to_bacterial,
    "Viral Images Misclassified as Bacterial",
    "viral_to_bacterial_contact_sheet.png"
)

generate_contact_sheet(
    correct_viral,
    "Correctly Classified Viral Images",
    "correct_viral_contact_sheet.png"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("CLASSIFIER ERROR VISUALIZATION COMPLETED")
print("=" * 65)

print(f"Normal -> COVID: {len(normal_to_covid)}")
print(f"Viral -> Bacterial: {len(viral_to_bacterial)}")
print(f"Correct Viral: {len(correct_viral)}")

print("\nOutput directory:")
print(OUTPUT_DIR)

## to run this script, use the following command:
# python scripts/74_visualize_classifier_errors.py