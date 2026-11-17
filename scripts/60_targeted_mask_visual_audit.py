"""
Script 60: Targeted visual audit of severe validation-set U-Net masks.

Selects:
- Every near-empty mask (<1% area).
- 20 most fragmented masks (component count, then smallest largest-component share).
- 20 most left/right-imbalanced masks (largest image-half area ratio).

For each selected image, recomputes the balanced U-Net mask and saves:
original X-ray, binary mask, and green overlay.
Also creates one contact sheet per selection group.

This is a validation-only diagnostic. It does not train or modify models.
The automated selection criteria are screening heuristics, not clinical labels.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    PROJECT_ROOT / "results" / "classification"
    / "lung_only_classifier_ablation"
    / "full_validation_mask_quality_audit"
    / "validation_mask_quality_per_image.csv"
)
MODEL_PATH = PROJECT_ROOT / "saved_models" / "unet_lung_segmentation_balanced.keras"
OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "classification"
    / "lung_only_classifier_ablation" / "targeted_mask_visual_audit"
)

IMAGE_SIZE = 224
THRESHOLD = 0.30
TOP_N = 20
THUMB_W = 224
THUMB_H = 224
LABEL_H = 58
COLUMNS = 4


def resolve_path(value):
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path


def load_gray(path):
    image = Image.open(path).convert("L")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def make_panels(image_path, model):
    gray = load_gray(image_path)
    pred = model.predict(gray[np.newaxis, ..., np.newaxis], verbose=0)[0]
    if pred.ndim == 3:
        pred = pred[..., 0]
    mask = pred >= THRESHOLD

    original = Image.fromarray(
        np.clip(gray * 255, 0, 255).astype(np.uint8)
    ).convert("RGB")

    mask_img = Image.fromarray(
        (mask.astype(np.uint8) * 255), mode="L"
    ).convert("RGB")

    overlay = np.asarray(original).copy()
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    return original, mask_img, overlay_img, float(mask.mean() * 100)


def safe_name(value):
    return "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(value)
    )[:100]


def make_contact_sheet(items, output_path, title):
    if not items:
        return

    tile_w = THUMB_W * 3
    tile_h = THUMB_H + LABEL_H
    rows = math.ceil(len(items) / COLUMNS)
    sheet = Image.new(
        "RGB",
        (COLUMNS * tile_w, rows * tile_h + 70),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 15), title, fill="black")

    for idx, item in enumerate(items):
        row = idx // COLUMNS
        col = idx % COLUMNS
        x = col * tile_w
        y = 70 + row * tile_h

        panels = item["panels"]
        for panel_idx, panel in enumerate(panels[:3]):
            sheet.paste(panel.resize((THUMB_W, THUMB_H)), (
                x + panel_idx * THUMB_W, y
            ))

        label = (
            f'{item["class_name"]} | {item["screening_category"]}\n'
            f'Area {item["recomputed_area"]:.2f}% | '
            f'Components {item["component_count"]} | '
            f'L/R {item["lr_ratio_text"]}'
        )
        draw.text((x + 4, y + THUMB_H + 4), label, fill="black")

    sheet.save(output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {CSV_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {MODEL_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {
        "processed_path", "class_name", "screening_category",
        "mask_area_pct", "connected_component_count",
        "largest_component_share", "left_right_area_ratio",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    near_empty = df[df["mask_area_pct"] < 1.0].copy()

    fragmented = df[
        df["screening_category"] == "fragmented_review"
    ].copy()
    fragmented = fragmented.sort_values(
        ["connected_component_count", "largest_component_share"],
        ascending=[False, True],
    ).head(TOP_N)

    imbalance = df[
        pd.to_numeric(df["left_right_area_ratio"], errors="coerce") > 4.0
    ].copy()
    imbalance = imbalance.sort_values(
        "left_right_area_ratio", ascending=False
    ).head(TOP_N)

    selections = {
        "near_empty": near_empty,
        "most_fragmented": fragmented,
        "most_imbalanced": imbalance,
    }

    print("Loading balanced U-Net...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    summary_rows = []

    for group_name, group_df in selections.items():
        group_dir = OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        contact_items = []
        print(f"\nProcessing {group_name}: {len(group_df)} images")

        for _, row in group_df.iterrows():
            image_path = resolve_path(row["processed_path"])
            if not image_path.is_file():
                print(f"SKIPPED missing image: {image_path}")
                continue

            original, mask_img, overlay, area = make_panels(image_path, model)
            stem = safe_name(image_path.stem)

            combined = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE), "white")
            combined.paste(original, (0, 0))
            combined.paste(mask_img, (IMAGE_SIZE, 0))
            combined.paste(overlay, (IMAGE_SIZE * 2, 0))
            combined_path = group_dir / f"{stem}_three_panel.png"
            combined.save(combined_path)

            lr_value = pd.to_numeric(
                pd.Series([row["left_right_area_ratio"]]),
                errors="coerce"
            ).iloc[0]
            lr_text = (
                "inf" if not np.isfinite(lr_value)
                else f"{lr_value:.2f}"
            )

            item = {
                "panels": [original, mask_img, overlay],
                "class_name": row["class_name"],
                "screening_category": row["screening_category"],
                "recomputed_area": area,
                "component_count": int(row["connected_component_count"]),
                "lr_ratio_text": lr_text,
            }
            contact_items.append(item)

            summary_rows.append({
                "selection_group": group_name,
                "processed_path": str(image_path),
                "class_name": row["class_name"],
                "source_dataset": row.get("source_dataset", "unknown"),
                "original_screening_category": row["screening_category"],
                "original_mask_area_pct": row["mask_area_pct"],
                "recomputed_mask_area_pct": area,
                "connected_component_count": row["connected_component_count"],
                "largest_component_share": row["largest_component_share"],
                "left_right_area_ratio": row["left_right_area_ratio"],
                "saved_three_panel": str(combined_path),
            })

            print(
                f'{row["class_name"]}: {image_path.name} | '
                f'recomputed area={area:.2f}%'
            )

        make_contact_sheet(
            contact_items,
            OUTPUT_DIR / f"{group_name}_contact_sheet.png",
            f"{group_name.replace('_', ' ').title()} | "
            f"{len(contact_items)} selected images "
            "(original | mask | overlay)",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "selected_image_audit.csv", index=False)

    print("\nTARGETED VISUAL AUDIT COMPLETE")
    print(f"Near-empty selected: {len(near_empty)}")
    print(f"Most fragmented selected: {len(fragmented)}")
    print(f"Most imbalanced selected: {len(imbalance)}")
    print(f"Successfully visualized: {len(summary_df)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nContact sheets:")
    for name in selections:
        print(f"  {OUTPUT_DIR / (name + '_contact_sheet.png')}")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/60_targeted_mask_visual_audit.py