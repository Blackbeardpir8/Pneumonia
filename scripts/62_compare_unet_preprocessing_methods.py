"""
Script 62: Compare three U-Net inference preprocessing methods on a
representative classification validation sample.

A: Direct resize to 224x224 (current baseline).
B: Aspect-preserving resize + padding with source-image mean intensity.
C: Center crop to square + resize.

Selects up to 5 images per class/source group from validation only.
Saves per-image metrics and one contact sheet per method.
No training, model changes, dataset changes, or test-set access.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/classification/manifest.csv"
MODEL = ROOT / "saved_models/unet_lung_segmentation_balanced.keras"
OUT = ROOT / "results/classification/lung_only_classifier_ablation/preprocessing_method_comparison"
SIZE = 224
THRESHOLD = 0.30
PER_GROUP = 5
SEED = 42
COLS = 3
TILE_W = 672
TILE_H = 282


def resolve_path(value):
    p = Path(str(value))
    if p.is_file():
        return p
    if not p.is_absolute() and (ROOT / p).is_file():
        return ROOT / p
    return p


def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def resize(gray, w, h):
    return np.asarray(
        Image.fromarray(gray, mode="L").resize(
            (w, h), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def direct(gray):
    return resize(gray, SIZE, SIZE)


def aspect_pad(gray):
    h, w = gray.shape
    scale = min(SIZE / w, SIZE / h)
    nw = max(1, min(SIZE, int(round(w * scale))))
    nh = max(1, min(SIZE, int(round(h * scale))))
    small = resize(gray, nw, nh)
    canvas = np.full((SIZE, SIZE), int(round(float(gray.mean()))), dtype=np.uint8)
    x0, y0 = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def center_crop(gray):
    h, w = gray.shape
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return resize(gray[y0:y0 + side, x0:x0 + side], SIZE, SIZE)


def predict(model, gray224):
    x = gray224.astype(np.float32) / 255.0
    p = model.predict(x[None, ..., None], verbose=0)[0]
    return p[..., 0] if p.ndim == 3 else p


def metrics(prob):
    mask = prob >= THRESHOLD
    count = int(ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1])
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:]
    largest_share = float(sizes.max() / sizes.sum()) if sizes.size and sizes.sum() else 0.0
    h, w = mask.shape
    left, right = int(mask[:, :w // 2].sum()), int(mask[:, w // 2:].sum())
    small, large = min(left, right), max(left, right)
    ratio = large / small if small else (float("inf") if large else 1.0)
    return {
        "mask_area_pct": float(mask.mean() * 100),
        "connected_component_count": count,
        "largest_component_share": largest_share,
        "left_right_area_ratio": float(ratio),
    }, mask


def overlay(gray, mask):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    return Image.fromarray(rgb)


def contact_sheet(items, title, output):
    if not items:
        return
    rows = math.ceil(len(items) / COLS)
    sheet = Image.new("RGB", (COLS * TILE_W, rows * TILE_H + 45), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title + " | original | mask | overlay", fill="black")
    for i, item in enumerate(items):
        x, y = (i % COLS) * TILE_W, 45 + (i // COLS) * TILE_H
        sheet.paste(item["original"], (x, y))
        sheet.paste(item["mask"], (x + 224, y))
        sheet.paste(item["overlay"], (x + 448, y))
        m = item["metrics"]
        ratio = "inf" if not np.isfinite(m["left_right_area_ratio"]) else f'{m["left_right_area_ratio"]:.2f}'
        draw.text((x + 3, y + 228),
                  f'{item["class"]} | {item["source"]}\n'
                  f'Area {m["mask_area_pct"]:.1f}% | CC {m["connected_component_count"]} | L/R {ratio}',
                  fill="black")
    sheet.save(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    df = pd.read_csv(MANIFEST)
    required = {"split", "processed_path", "class_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing columns: {required - set(df.columns)}")
    val = df[df["split"].astype(str).str.lower().isin(["val", "valid", "validation"])].copy()
    if "source_dataset" not in val.columns:
        val["source_dataset"] = "unknown"

    selected_parts = []
    for _, group in val.groupby(["class_name", "source_dataset"], dropna=False, sort=True):
        selected_parts.append(group.sample(n=min(PER_GROUP, len(group)), random_state=SEED))
    selected = pd.concat(selected_parts).sort_values(
        ["class_name", "source_dataset", "processed_path"]
    ).reset_index(drop=True)

    print(f"Validation images available: {len(val)}")
    print(f"Representative images selected: {len(selected)}")
    print("Loading balanced U-Net...")
    model = tf.keras.models.load_model(MODEL, compile=False)

    methods = {
        "direct_resize": direct,
        "aspect_preserving_pad": aspect_pad,
        "center_crop": center_crop,
    }
    rows = []
    visuals = {name: [] for name in methods}

    for i, row in selected.iterrows():
        path = resolve_path(row["processed_path"])
        if not path.is_file():
            print(f"Missing image, skipped: {path}")
            continue
        raw = read_gray(path)
        original_panel = Image.fromarray(direct(raw)).convert("RGB")

        for name, transform in methods.items():
            prepared = transform(raw)
            prob = predict(model, prepared)
            met, mask = metrics(prob)
            rows.append({
                "processed_path": str(path),
                "class_name": row["class_name"],
                "source_dataset": row["source_dataset"],
                "method": name,
                "mask_threshold": THRESHOLD,
                "original_width": raw.shape[1],
                "original_height": raw.shape[0],
                **met,
            })
            visuals[name].append({
                "original": original_panel,
                "mask": Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"),
                "overlay": overlay(prepared, mask),
                "class": row["class_name"],
                "source": row["source_dataset"],
                "metrics": met,
            })
        if (i + 1) % 10 == 0 or i + 1 == len(selected):
            print(f"Processed sample {i + 1}/{len(selected)}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "metrics_by_image_and_method.csv", index=False)
    summary = result.groupby("method").agg(
        image_count=("processed_path", "count"),
        mean_mask_area_pct=("mask_area_pct", "mean"),
        median_mask_area_pct=("mask_area_pct", "median"),
        mean_component_count=("connected_component_count", "mean"),
        median_component_count=("connected_component_count", "median"),
        mean_largest_component_share=("largest_component_share", "mean"),
        median_left_right_ratio=("left_right_area_ratio", "median"),
        near_empty_count=("mask_area_pct", lambda x: int((x < 1).sum())),
        low_area_below_10_count=("mask_area_pct", lambda x: int((x < 10).sum())),
    ).reset_index()
    summary.to_csv(OUT / "method_summary.csv", index=False)

    for name, items in visuals.items():
        contact_sheet(items, name, OUT / f"contact_sheet_{name}.png")

    print("\nPREPROCESSING COMPARISON COMPLETE")
    print(f"Selected images: {selected['processed_path'].nunique()}")
    print(f"Predictions generated: {len(result)}")
    print(f"Output directory: {OUT}")
    print(summary.to_string(index=False))
    print("Contact sheets:")
    for name in methods:
        print(OUT / f"contact_sheet_{name}.png")


if __name__ == "__main__":
    main()
