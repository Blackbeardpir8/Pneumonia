from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
)

PREDICTIONS_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "source_balanced_model_comparison"
    / "validation_predictions.csv"
)
VALIDATION_MANIFEST = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "source_balanced_threshold_analysis"

VIRAL_INDEX = CLASS_TO_INDEX["Viral"]
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.025), 3)
MIN_VIRAL_PRECISION = 0.80


def load_inputs():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}. "
            "Run script 51 first."
        )
    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(VALIDATION_MANIFEST)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    manifest = pd.read_csv(VALIDATION_MANIFEST)

    required_predictions = [
        "model",
        "processed_path",
        "actual_class_index",
        *[f"probability_{name.lower()}" for name in CLASS_NAMES],
    ]
    missing = [
        column for column in required_predictions
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {missing}"
        )

    required_manifest = ["split", "processed_path", "source_dataset"]
    missing = [
        column for column in required_manifest
        if column not in manifest.columns
    ]
    if missing:
        raise ValueError(
            f"Validation manifest is missing columns: {missing}"
        )

    val_sources = manifest[
        manifest["split"].astype(str).str.lower().eq("val")
    ][["processed_path", "source_dataset"]].copy()

    predictions = predictions.merge(
        val_sources,
        on="processed_path",
        how="left",
        validate="many_to_one",
    )

    if predictions["source_dataset"].isna().any():
        missing_count = int(predictions["source_dataset"].isna().sum())
        raise ValueError(
            f"Could not map source_dataset for {missing_count} prediction rows."
        )

    return predictions


def predict_with_viral_threshold(probabilities, threshold):
    nonviral_indices = [
        index for index in range(len(CLASS_NAMES))
        if index != VIRAL_INDEX
    ]
    best_nonviral = np.asarray(nonviral_indices)[
        np.argmax(probabilities[:, nonviral_indices], axis=1)
    ]
    predicted = best_nonviral.astype(np.int32)
    predicted[probabilities[:, VIRAL_INDEX] >= threshold] = VIRAL_INDEX
    return predicted


def binary_metrics(y_true, y_pred):
    actual_viral = y_true == VIRAL_INDEX
    predicted_viral = y_pred == VIRAL_INDEX

    tp = int(np.sum(actual_viral & predicted_viral))
    fp = int(np.sum(~actual_viral & predicted_viral))
    fn = int(np.sum(actual_viral & ~predicted_viral))
    tn = int(np.sum(~actual_viral & ~predicted_viral))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "viral_tp": tp,
        "viral_fp": fp,
        "viral_fn": fn,
        "viral_tn": tn,
        "viral_precision": precision,
        "viral_recall": recall,
        "viral_f1": f1,
    }


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive the previous results before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    probability_columns = [
        f"probability_{name.lower()}" for name in CLASS_NAMES
    ]

    sweep_rows = []
    selected_rows = []
    source_rows = []

    for model_name, model_df in data.groupby("model", sort=False):
        y_true = model_df["actual_class_index"].astype(int).to_numpy()
        probabilities = model_df[probability_columns].astype(float).to_numpy()

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite probabilities for {model_name}")

        # Record standard argmax baseline.
        argmax_pred = np.argmax(probabilities, axis=1)
        baseline = binary_metrics(y_true, argmax_pred)
        selected_rows.append({
            "model": model_name,
            "operating_point": "standard_argmax",
            "threshold": np.nan,
            "accuracy": float(np.mean(argmax_pred == y_true)),
            **baseline,
        })

        model_sweep = []
        for threshold in THRESHOLDS:
            y_pred = predict_with_viral_threshold(probabilities, threshold)
            metrics = binary_metrics(y_true, y_pred)
            row = {
                "model": model_name,
                "threshold": float(threshold),
                "accuracy": float(np.mean(y_pred == y_true)),
                **metrics,
            }
            model_sweep.append(row)
            sweep_rows.append(row)

        sweep_df = pd.DataFrame(model_sweep)

        # Highest F1; ties prefer higher precision, then higher threshold.
        best_f1 = sweep_df.sort_values(
            ["viral_f1", "viral_precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

        selected_rows.append({
            "model": model_name,
            "operating_point": "best_viral_f1",
            "threshold": float(best_f1["threshold"]),
            **best_f1.drop(labels=["model", "threshold"]).to_dict(),
        })

        precision_qualified = sweep_df[
            sweep_df["viral_precision"] >= MIN_VIRAL_PRECISION
        ]
        if not precision_qualified.empty:
            best_precision_recall = precision_qualified.sort_values(
                ["viral_recall", "viral_f1", "accuracy"],
                ascending=[False, False, False],
            ).iloc[0]
            selected_rows.append({
                "model": model_name,
                "operating_point": "best_recall_at_precision_080",
                "threshold": float(best_precision_recall["threshold"]),
                **best_precision_recall.drop(
                    labels=["model", "threshold"]
                ).to_dict(),
            })

        # Source-wise performance at standard argmax and selected thresholds.
        operating_points = [
            ("standard_argmax", None),
            ("best_viral_f1", float(best_f1["threshold"])),
        ]
        if not precision_qualified.empty:
            operating_points.append((
                "best_recall_at_precision_080",
                float(best_precision_recall["threshold"]),
            ))

        for point_name, threshold in operating_points:
            if threshold is None:
                y_pred = argmax_pred
            else:
                y_pred = predict_with_viral_threshold(probabilities, threshold)

            for source in model_df["source_dataset"].dropna().unique():
                source_mask = (
                    model_df["source_dataset"].to_numpy() == source
                )
                metrics = binary_metrics(
                    y_true[source_mask],
                    y_pred[source_mask],
                )
                source_rows.append({
                    "model": model_name,
                    "operating_point": point_name,
                    "threshold": threshold,
                    "source_dataset": source,
                    "total_images": int(source_mask.sum()),
                    **metrics,
                    "actual_normal_predicted_viral": int(np.sum(
                        (y_true[source_mask] == CLASS_TO_INDEX["Normal"])
                        & (y_pred[source_mask] == VIRAL_INDEX)
                    )),
                    "actual_bacterial_predicted_viral": int(np.sum(
                        (y_true[source_mask] == CLASS_TO_INDEX["Bacterial"])
                        & (y_pred[source_mask] == VIRAL_INDEX)
                    )),
                    "actual_covid_predicted_viral": int(np.sum(
                        (y_true[source_mask] == CLASS_TO_INDEX["COVID"])
                        & (y_pred[source_mask] == VIRAL_INDEX)
                    )),
                })

    sweep_result = pd.DataFrame(sweep_rows)
    selected_result = pd.DataFrame(selected_rows)
    source_result = pd.DataFrame(source_rows)

    sweep_result.to_csv(
        OUTPUT_DIR / "viral_threshold_sweep.csv",
        index=False,
    )
    selected_result.to_csv(
        OUTPUT_DIR / "selected_threshold_comparison.csv",
        index=False,
    )
    source_result.to_csv(
        OUTPUT_DIR / "source_wise_selected_threshold_metrics.csv",
        index=False,
    )

    print("\nSTANDARD ARGMAX AND SELECTED THRESHOLD RESULTS")
    print(selected_result.to_string(index=False))

    print("\nSOURCE-WISE RESULTS AT SELECTED OPERATING POINTS")
    print(source_result.to_string(index=False))

    print(f"\nSaved results to:\n{OUTPUT_DIR}")
    print("\nValidation predictions only. Test split was not accessed.")
    print(
        "Thresholds are selected on this validation set and are exploratory; "
        "do not treat them as final until tested on an untouched test set."
    )


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/52_source_balanced_viral_threshold_analysis.py