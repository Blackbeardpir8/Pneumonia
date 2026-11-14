"""
Script 61: Read-only image geometry and intensity comparison.

Compares:
1. Original segmentation training images.
2. Processed segmentation training images.
3. Classification validation images.

Measures original dimensions, aspect ratio, grayscale intensity, and border
brightness. Does not modify images, manifests, models, or checkpoints.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_MANIFEST = PROJECT_ROOT / "data" / "processed" / "segmentation" / "manifest.csv"
CLS_MANIFEST = PROJECT_ROOT / "data" / "processed" / "classification" / "manifest.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "classification"
    / "lung_only_classifier_ablation"
    / "preprocessing_domain_audit"
)

BORDER_FRACTION = 0.08
MAX_IMAGES_PER_GROUP = None  # None = use every image in the selected group


def resolve_path(value):
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path


def image_measurements(path):
    with Image.open(path) as image:
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    h, w = gray.shape
    bx = max(1, int(round(w * BORDER_FRACTION)))
    by = max(1, int(round(h * BORDER_FRACTION)))

    border = np.concatenate([
        gray[:by, :].ravel(),
        gray[-by:, :].ravel(),
        gray[by:-by, :bx].ravel() if h > 2 * by else np.array([], dtype=np.float32),
        gray[by:-by, -bx:].ravel() if h > 2 * by else np.array([], dtype=np.float32),
    ])

    center = gray[
        int(h * 0.25):max(int(h * 0.25) + 1, int(h * 0.75)),
        int(w * 0.25):max(int(w * 0.25) + 1, int(w * 0.75)),
    ]

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio_w_over_h": float(width / height),
        "pixel_mean": float(gray.mean()),
        "pixel_std": float(gray.std()),
        "pixel_p05": float(np.percentile(gray, 5)),
        "pixel_p50": float(np.percentile(gray, 50)),
        "pixel_p95": float(np.percentile(gray, 95)),
        "border_mean": float(border.mean()) if border.size else None,
        "border_std": float(border.std()) if border.size else None,
        "center_mean": float(center.mean()),
        "border_below_15_pct": float((border < 15).mean() * 100) if border.size else None,
        "border_below_30_pct": float((border < 30).mean() * 100) if border.size else None,
    }


def inspect_rows(frame, path_column, group_name, extra_columns=None):
    records = []
    failures = []
    extra_columns = extra_columns or []

    if MAX_IMAGES_PER_GROUP is not None:
        frame = frame.head(MAX_IMAGES_PER_GROUP)

    for idx, row in frame.reset_index(drop=True).iterrows():
        path = resolve_path(row[path_column])
        try:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            record = {
                "group": group_name,
                "image_path": str(path),
                **image_measurements(path),
            }
            for column in extra_columns:
                if column in row.index:
                    record[column] = row[column]
            records.append(record)
        except Exception as exc:
            failures.append({
                "group": group_name,
                "image_path": str(path),
                "error": repr(exc),
            })

        if (idx + 1) % 500 == 0 or idx + 1 == len(frame):
            print(f"{group_name}: {idx + 1}/{len(frame)}")

    return records, failures


def summarize(data):
    metric_columns = [
        "width", "height", "aspect_ratio_w_over_h",
        "pixel_mean", "pixel_std", "pixel_p05", "pixel_p50", "pixel_p95",
        "border_mean", "border_std", "center_mean",
        "border_below_15_pct", "border_below_30_pct",
    ]
    rows = []
    for group, frame in data.groupby("group"):
        row = {"group": group, "image_count": len(frame)}
        for col in metric_columns:
            row[f"{col}_mean"] = frame[col].mean()
            row[f"{col}_median"] = frame[col].median()
            row[f"{col}_std"] = frame[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEG_MANIFEST.is_file():
        raise FileNotFoundError(SEG_MANIFEST)
    if not CLS_MANIFEST.is_file():
        raise FileNotFoundError(CLS_MANIFEST)

    seg = pd.read_csv(SEG_MANIFEST)
    cls = pd.read_csv(CLS_MANIFEST)

    seg_required = {"split", "original_image", "processed_image"}
    cls_required = {"split", "processed_path", "class_name"}
    if not seg_required.issubset(seg.columns):
        raise ValueError(f"Segmentation manifest missing: {seg_required - set(seg.columns)}")
    if not cls_required.issubset(cls.columns):
        raise ValueError(f"Classification manifest missing: {cls_required - set(cls.columns)}")

    seg_train = seg[seg["split"].astype(str).str.lower() == "train"].copy()
    cls_val = cls[cls["split"].astype(str).str.lower().isin(["val", "valid", "validation"])].copy()

    if "source_dataset" not in cls_val.columns:
        cls_val["source_dataset"] = "unknown"

    all_records = []
    all_failures = []

    jobs = [
        (seg_train, "original_image", "segmentation_original_train", ["pair_id"]),
        (seg_train, "processed_image", "segmentation_processed_train", ["pair_id"]),
        (cls_val, "processed_path", "classification_processed_validation",
         ["class_name", "source_dataset"]),
    ]

    print(f"Segmentation training pairs: {len(seg_train)}")
    print(f"Classification validation images: {len(cls_val)}")

    for frame, path_col, group, extras in jobs:
        records, failures = inspect_rows(frame, path_col, group, extras)
        all_records.extend(records)
        all_failures.extend(failures)

    measurements = pd.DataFrame(all_records)
    failures = pd.DataFrame(all_failures)

    measurements.to_csv(OUTPUT_DIR / "image_geometry_intensity_per_image.csv", index=False)
    failures.to_csv(OUTPUT_DIR / "image_read_failures.csv", index=False)

    overall = summarize(measurements)
    overall.to_csv(OUTPUT_DIR / "summary_by_image_group.csv", index=False)

    if "class_name" in measurements.columns:
        class_summary = (
            measurements[
                measurements["group"] == "classification_processed_validation"
            ]
            .groupby(["class_name", "source_dataset"], dropna=False)
            .agg(
                image_count=("image_path", "count"),
                aspect_ratio_median=("aspect_ratio_w_over_h", "median"),
                pixel_mean_median=("pixel_mean", "median"),
                pixel_std_median=("pixel_std", "median"),
                border_mean_median=("border_mean", "median"),
                border_below_15_pct_median=("border_below_15_pct", "median"),
                border_below_30_pct_median=("border_below_30_pct", "median"),
            )
            .reset_index()
        )
        class_summary.to_csv(
            OUTPUT_DIR / "classification_summary_by_class_and_source.csv",
            index=False,
        )

    summary_json = {
        "segmentation_training_images": int(len(seg_train)),
        "classification_validation_images": int(len(cls_val)),
        "successfully_measured": int(len(measurements)),
        "read_failures": int(len(failures)),
        "border_fraction_per_side": BORDER_FRACTION,
        "note": (
            "Read-only descriptive audit. Border darkness is a proxy, not "
            "a validated padding detector. Processed segmentation images "
            "and classification images are both measured at their stored "
            "dimensions; original segmentation images retain native dimensions."
        ),
        "output_directory": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\nAUDIT COMPLETE")
    print(f"Successfully measured: {len(measurements)}")
    print(f"Read failures: {len(failures)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGroup summary:")
    if not overall.empty:
        print(overall[[
            "group", "image_count",
            "width_median", "height_median",
            "aspect_ratio_w_over_h_median",
            "pixel_mean_median", "pixel_std_median",
            "border_mean_median", "border_below_15_pct_median",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
