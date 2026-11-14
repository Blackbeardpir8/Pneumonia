from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import CLASSIFICATION_DATA_DIR

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "classification_source_balanced"
OUTPUT_TRAIN_DIR = OUTPUT_DIR / "train"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.csv"

RANDOM_SEED = 42

# Keep Normal and Viral counts equal within each source:
# - Bacterial/Viral source: 832 Normal + 832 Viral
# - COVID Radiography source: 946 Normal + 946 Viral
#
# Keep every Bacterial and COVID training image.
# Validation and test rows are deliberately excluded.
TARGET_COUNTS = {
    ("bacterial_viral_dataset", "Normal"): 832,
    ("bacterial_viral_dataset", "Viral"): 832,
    ("covid19_radiography", "Normal"): 946,
    ("covid19_radiography", "Viral"): 946,
}

REQUIRED_COLUMNS = ["split", "class_name", "source_dataset", "processed_path"]


def resolve_source_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    candidates = [ROOT / path, CLASSIFICATION_DATA_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Classification manifest not found: {MANIFEST_PATH}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Do not overwrite an existing experiment. Rename/archive it first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest is missing columns: {missing}\n"
            f"Available columns: {list(manifest.columns)}"
        )

    train = manifest[
        manifest["split"].astype(str).str.lower().eq("train")
    ].copy()

    if train.empty:
        raise ValueError("No training rows found in the classification manifest.")

    selected_parts = []

    # Deterministic sampling for the four source/class combinations.
    for (source, class_name), target_count in TARGET_COUNTS.items():
        group = train[
            train["source_dataset"].astype(str).eq(source)
            & train["class_name"].astype(str).eq(class_name)
        ]

        if len(group) < target_count:
            raise ValueError(
                f"Not enough rows for {source}/{class_name}: "
                f"found {len(group)}, need {target_count}."
            )

        selected = group.sample(
            n=target_count,
            random_state=RANDOM_SEED,
            replace=False,
        )
        selected_parts.append(selected)

    # Preserve all Bacterial and COVID training rows.
    retained_special = train[
        train["class_name"].astype(str).isin(["Bacterial", "COVID"])
    ].copy()
    selected_parts.append(retained_special)

    selected = pd.concat(selected_parts, ignore_index=True)

    # Check that there are no duplicated manifest rows or processed paths.
    if selected["processed_path"].astype(str).duplicated().any():
        duplicates = selected[
            selected["processed_path"].astype(str).duplicated(keep=False)
        ]
        raise ValueError(
            "Duplicate processed_path values found in selected rows:\n"
            + duplicates[["source_dataset", "class_name", "processed_path"]]
            .head(10)
            .to_string(index=False)
        )

    OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=False)

    new_paths = []
    copied = 0

    for row_number, row in enumerate(selected.itertuples(index=False), start=1):
        source_path = resolve_source_path(getattr(row, "processed_path"))

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source image does not exist: {source_path}\n"
                "No source files have been modified."
            )

        class_name = str(getattr(row, "class_name"))
        class_dir = OUTPUT_TRAIN_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Prefix with a stable row number to prevent filename collisions.
        destination = class_dir / f"{row_number:05d}_{source_path.name}"
        shutil.copy2(source_path, destination)
        new_paths.append(str(destination.relative_to(ROOT)))
        copied += 1

        if copied % 500 == 0 or copied == len(selected):
            print(f"Copied {copied:,}/{len(selected):,} training images")

    selected = selected.copy()
    selected["original_processed_path"] = selected["processed_path"]
    selected["processed_path"] = new_paths
    selected["split"] = "train"
    selected.to_csv(OUTPUT_MANIFEST, index=False)

    # Save a compact count report beside the new manifest.
    count_report = (
        selected.groupby(["source_dataset", "class_name"], dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values(["source_dataset", "class_name"])
    )
    count_report.to_csv(OUTPUT_DIR / "class_source_counts.csv", index=False)

    print("\nNEW SOURCE-BALANCED TRAINING COUNTS")
    print(count_report.to_string(index=False))

    class_totals = (
        selected.groupby("class_name").size().reindex(
            ["Normal", "Viral", "Bacterial", "COVID"], fill_value=0
        )
    )
    print("\nTOTALS BY CLASS")
    print(class_totals.to_string())

    print(f"\nTotal selected training images: {len(selected):,}")
    print(f"Copied training images: {copied:,}")
    print(f"New dataset folder: {OUTPUT_DIR}")
    print(f"New manifest: {OUTPUT_MANIFEST}")
    print("\nOriginal classification data and manifest were not modified.")
    print("Validation and test data were not copied or accessed.")
    print(
        "\nIMPORTANT: This creates a training-data subset only. "
        "Do not train with the old class_weights.json without recalculating "
        "class weights for this new subset."
    )


if __name__ == "__main__":
    main()
