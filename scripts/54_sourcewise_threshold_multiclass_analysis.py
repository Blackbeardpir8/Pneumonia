from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASS_NAMES,
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
)

PREDICTIONS_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "source_balanced_threshold_multiclass_comparison"
    / "selected_rule_predictions.csv"
)
VALIDATION_MANIFEST = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUTPUT_DIR = (
    CLASSIFICATION_RESULTS_DIR
    / "source_balanced_threshold_sourcewise_multiclass"
)
LABELS = list(range(len(CLASS_NAMES)))


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Run script 53 first. Missing: {PREDICTIONS_PATH}"
        )
    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(VALIDATION_MANIFEST)
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive previous results before rerunning."
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    manifest = pd.read_csv(VALIDATION_MANIFEST)

    required_predictions = [
        "scenario",
        "processed_path",
        "actual_class_index",
        "predicted_class_index",
    ]
    missing = [
        column for column in required_predictions
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")

    required_manifest = ["split", "processed_path", "source_dataset"]
    missing = [
        column for column in required_manifest
        if column not in manifest.columns
    ]
    if missing:
        raise ValueError(f"Validation manifest missing columns: {missing}")

    val_sources = manifest[
        manifest["split"].astype(str).str.lower().eq("val")
    ][["processed_path", "source_dataset"]].copy()

    data = predictions.merge(
        val_sources,
        on="processed_path",
        how="left",
        validate="many_to_one",
    )

    if data["source_dataset"].isna().any():
        missing_count = int(data["source_dataset"].isna().sum())
        raise ValueError(
            f"Could not map source_dataset for {missing_count} prediction rows."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    class_rows = []
    confusion_rows = []

    for (scenario, source), group in data.groupby(
        ["scenario", "source_dataset"], sort=False
    ):
        y_true = group["actual_class_index"].astype(int).to_numpy()
        y_pred = group["predicted_class_index"].astype(int).to_numpy()

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        overall_rows.append({
            "scenario": scenario,
            "source_dataset": source,
            "total_images": int(len(group)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        })

        for class_name in CLASS_NAMES:
            metrics = report[class_name]
            class_rows.append({
                "scenario": scenario,
                "source_dataset": source,
                "class_name": class_name,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            })

        matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
        for actual_index, actual_name in enumerate(CLASS_NAMES):
            for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                confusion_rows.append({
                    "scenario": scenario,
                    "source_dataset": source,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(matrix[actual_index, predicted_index]),
                })

    outputs = {
        "source_wise_overall_metrics.csv": pd.DataFrame(overall_rows),
        "source_wise_class_metrics.csv": pd.DataFrame(class_rows),
        "source_wise_confusion_matrices.csv": pd.DataFrame(confusion_rows),
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    print("\nSOURCE-WISE OVERALL METRICS")
    print(outputs["source_wise_overall_metrics.csv"].to_string(index=False))

    print("\nSOURCE-WISE CLASS METRICS")
    print(outputs["source_wise_class_metrics.csv"].to_string(index=False))

    print("\nSOURCE-WISE CONFUSION MATRICES (rows=actual, columns=predicted)")
    confusions = outputs["source_wise_confusion_matrices.csv"]
    for (scenario, source), group in confusions.groupby(
        ["scenario", "source_dataset"], sort=False
    ):
        matrix = (
            group.pivot(
                index="actual_class",
                columns="predicted_class",
                values="count",
            )
            .reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
            .fillna(0)
            .astype(int)
        )
        print(f"\nScenario: {scenario} | Source: {source}")
        print(matrix.to_string())

    print(f"\nSaved results to:\n{OUTPUT_DIR}")
    print("\nValidation data only. Test split was not accessed.")


if __name__ == "__main__":
    main()



# to run this script, use the following command:
# python scripts/54_sourcewise_threshold_multiclass_analysis.py