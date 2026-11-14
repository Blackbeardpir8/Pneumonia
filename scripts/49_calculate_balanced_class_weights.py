from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import CLASS_NAMES

DATASET_DIR = ROOT / "data" / "processed" / "classification_source_balanced"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
OUTPUT_PATH = DATASET_DIR / "class_weights.json"

REQUIRED_COLUMNS = ["split", "class_name", "source_dataset", "processed_path"]


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Balanced manifest not found: {MANIFEST_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Class weights already exist: {OUTPUT_PATH}. "
            "Review/archive the existing file before regenerating."
        )

    df = pd.read_csv(MANIFEST_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    train = df[df["split"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("No training rows found in balanced manifest.")

    unexpected = sorted(set(train["class_name"].dropna()) - set(CLASS_NAMES))
    if unexpected:
        raise ValueError(f"Unexpected class names in manifest: {unexpected}")

    counts = (
        train["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    if (counts == 0).any():
        missing_classes = counts[counts == 0].index.tolist()
        raise ValueError(f"Classes with zero training images: {missing_classes}")

    total = int(counts.sum())
    number_of_classes = len(CLASS_NAMES)

    # Balanced class weight: N / (K * class_count)
    weights = {
        class_name: float(total / (number_of_classes * int(counts[class_name])))
        for class_name in CLASS_NAMES
    }

    OUTPUT_PATH.write_text(
        json.dumps(weights, indent=4) + "\n",
        encoding="utf-8",
    )

    report = pd.DataFrame({
        "class_name": CLASS_NAMES,
        "image_count": [int(counts[name]) for name in CLASS_NAMES],
        "class_weight": [weights[name] for name in CLASS_NAMES],
    })

    print("\nBALANCED TRAINING CLASS COUNTS AND WEIGHTS")
    print(report.to_string(index=False))
    print(f"\nTotal training images: {total:,}")
    print(f"Saved class weights to: {OUTPUT_PATH}")
    print("\nOnly the balanced training manifest was read.")
    print("Original dataset, validation data, test data, and model checkpoints were not modified.")


if __name__ == "__main__":
    main()
