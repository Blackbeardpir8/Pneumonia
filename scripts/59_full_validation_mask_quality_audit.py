"""
Script 59: Full validation-set U-Net mask quality screening audit.

This script:
- Loads the balanced U-Net checkpoint (does not train or modify it).
- Recomputes masks for every validation image in the classification manifest.
- Measures mask area, connected components, largest-component share,
  left/right area balance, and mask bounding-box coverage.
- Produces per-image, per-class, and per-source summaries.

IMPORTANT:
The categories are automated screening flags, NOT ground-truth segmentation
quality labels. This script does not use the held-out test set.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "classification" / "manifest.csv"
)
MODEL_PATH = (
    PROJECT_ROOT / "saved_models" / "unet_lung_segmentation_balanced.keras"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "lung_only_classifier_ablation"
    / "full_validation_mask_quality_audit"
)

IMAGE_SIZE = 224
MASK_THRESHOLD = 0.30
EXPECTED_SPLIT = "val"


def resolve_image_path(value):
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path


def load_grayscale_for_unet(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[..., np.newaxis]


def connected_component_metrics(binary_mask):
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(binary_mask, structure=structure)

    if component_count == 0:
        return 0, 0.0, []

    counts = np.bincount(labels.ravel())[1:]
    counts = np.sort(counts)[::-1]
    total = int(counts.sum())
    largest_share = float(counts[0] / total) if total else 0.0
    return int(component_count), largest_share, counts.tolist()


def categorize_mask(area_pct, component_count, largest_share, lr_ratio):
    # These thresholds are deliberately conservative screening heuristics.
    if area_pct < 1.0:
        return "failed_or_near_empty"
    if component_count > 8 or largest_share < 0.50:
        return "fragmented_review"
    if area_pct < 10.0 or lr_ratio > 4.0:
        return "partial_or_imbalanced_review"
    if area_pct > 45.0:
        return "unusually_large_review"
    return "plausible_shape_review"


def summarize_group(frame, group_columns):
    rows = []
    grouped = frame.groupby(group_columns, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        row.update({
            "image_count": int(len(group)),
            "mean_mask_area_pct": round(float(group["mask_area_pct"].mean()), 3),
            "median_mask_area_pct": round(float(group["mask_area_pct"].median()), 3),
            "empty_or_near_empty_count": int(
                (group["mask_area_pct"] < 1.0).sum()
            ),
            "low_area_below_10pct_count": int(
                (group["mask_area_pct"] < 10.0).sum()
            ),
            "fragmented_review_count": int(
                (group["screening_category"] == "fragmented_review").sum()
            ),
            "partial_or_imbalanced_review_count": int(
                (group["screening_category"] == "partial_or_imbalanced_review").sum()
            ),
            "plausible_shape_review_count": int(
                (group["screening_category"] == "plausible_shape_review").sum()
            ),
            "unusually_large_review_count": int(
                (group["screening_category"] == "unusually_large_review").sum()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Balanced U-Net checkpoint not found: {MODEL_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    required = {"split", "class_name", "processed_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    validation = manifest[
        manifest["split"].astype(str).str.lower().isin(
            ["val", "valid", "validation"]
        )
    ].copy()

    if validation.empty:
        raise ValueError(
            "No validation rows found. Check the manifest split column; "
            f"observed values: {manifest['split'].dropna().unique().tolist()}"
        )

    if "source_dataset" not in validation.columns:
        validation["source_dataset"] = "unknown"

    print(f"Validation images: {len(validation)}")
    print(f"Loading balanced U-Net: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    records = []
    failures = []

    for index, row in validation.reset_index(drop=True).iterrows():
        image_path = resolve_image_path(row["processed_path"])

        try:
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))

            image = load_grayscale_for_unet(image_path)
            prediction = model.predict(image[np.newaxis, ...], verbose=0)
            probability_map = np.asarray(prediction)[0]

            if probability_map.ndim == 3:
                probability_map = probability_map[..., 0]
            elif probability_map.ndim != 2:
                raise ValueError(
                    f"Unexpected U-Net output shape: {np.asarray(prediction).shape}"
                )

            binary = probability_map >= MASK_THRESHOLD
            height, width = binary.shape
            total_pixels = int(binary.size)
            mask_pixels = int(binary.sum())
            area_pct = 100.0 * mask_pixels / total_pixels

            component_count, largest_share, component_sizes = (
                connected_component_metrics(binary)
            )

            left_area = int(binary[:, : width // 2].sum())
            right_area = int(binary[:, width // 2 :].sum())
            smaller_side = min(left_area, right_area)
            larger_side = max(left_area, right_area)
            lr_ratio = (
                float(larger_side / smaller_side)
                if smaller_side > 0
                else (float("inf") if larger_side > 0 else 1.0)
            )

            ys, xs = np.where(binary)
            if len(xs):
                bbox_width_pct = 100.0 * (xs.max() - xs.min() + 1) / width
                bbox_height_pct = 100.0 * (ys.max() - ys.min() + 1) / height
                bbox_area_pct = bbox_width_pct * bbox_height_pct / 100.0
            else:
                bbox_width_pct = 0.0
                bbox_height_pct = 0.0
                bbox_area_pct = 0.0

            category = categorize_mask(
                area_pct, component_count, largest_share, lr_ratio
            )

            records.append({
                "processed_path": str(image_path),
                "class_name": row["class_name"],
                "source_dataset": row["source_dataset"],
                "mask_threshold": MASK_THRESHOLD,
                "mask_area_pct": area_pct,
                "mask_pixel_count": mask_pixels,
                "connected_component_count": component_count,
                "largest_component_share": largest_share,
                "left_half_mask_pixels": left_area,
                "right_half_mask_pixels": right_area,
                "left_right_area_ratio": lr_ratio,
                "bbox_width_pct": bbox_width_pct,
                "bbox_height_pct": bbox_height_pct,
                "bbox_area_pct": bbox_area_pct,
                "component_sizes_pixels_desc": json.dumps(component_sizes),
                "screening_category": category,
            })

        except Exception as exc:
            failures.append({
                "processed_path": str(image_path),
                "class_name": row.get("class_name", "unknown"),
                "source_dataset": row.get("source_dataset", "unknown"),
                "error": repr(exc),
            })

        if (index + 1) % 100 == 0 or index + 1 == len(validation):
            print(f"Processed {index + 1}/{len(validation)}")

    results = pd.DataFrame(records)
    failures_df = pd.DataFrame(failures)

    results_path = OUTPUT_DIR / "validation_mask_quality_per_image.csv"
    class_path = OUTPUT_DIR / "validation_mask_quality_by_class.csv"
    source_path = OUTPUT_DIR / "validation_mask_quality_by_source.csv"
    class_source_path = OUTPUT_DIR / "validation_mask_quality_by_class_and_source.csv"
    failures_path = OUTPUT_DIR / "processing_failures.csv"
    summary_path = OUTPUT_DIR / "audit_summary.json"

    results.to_csv(results_path, index=False)

    by_class = summarize_group(results, ["class_name"])
    by_source = summarize_group(results, ["source_dataset"])
    by_class_source = summarize_group(
        results, ["class_name", "source_dataset"]
    )

    by_class.to_csv(class_path, index=False)
    by_source.to_csv(source_path, index=False)
    by_class_source.to_csv(class_source_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    category_counts = (
        results["screening_category"].value_counts(dropna=False).to_dict()
        if not results.empty else {}
    )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "manifest": str(MANIFEST_PATH),
        "model": str(MODEL_PATH),
        "split_used": "validation only",
        "validation_rows_in_manifest": int(len(validation)),
        "successfully_processed": int(len(results)),
        "processing_failures": int(len(failures_df)),
        "mask_threshold": MASK_THRESHOLD,
        "mean_mask_area_pct": (
            float(results["mask_area_pct"].mean()) if not results.empty else None
        ),
        "median_mask_area_pct": (
            float(results["mask_area_pct"].median()) if not results.empty else None
        ),
        "empty_or_near_empty_count": (
            int((results["mask_area_pct"] < 1.0).sum())
            if not results.empty else 0
        ),
        "low_area_below_10pct_count": (
            int((results["mask_area_pct"] < 10.0).sum())
            if not results.empty else 0
        ),
        "screening_category_counts": category_counts,
        "interpretation_note": (
            "Categories are heuristic screening flags, not ground-truth "
            "segmentation-quality labels. No training or test-set evaluation "
            "was performed."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nAUDIT COMPLETE")
    print(f"Successfully processed: {len(results)}")
    print(f"Processing failures: {len(failures_df)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nScreening category counts:")
    for name, count in category_counts.items():
        print(f"  {name}: {count}")

    print("\nSaved files:")
    for path in [
        results_path, class_path, source_path, class_source_path,
        failures_path, summary_path
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()


# to run this script, use the following command:    
# python scripts/59_full_validation_mask_quality_audit.py