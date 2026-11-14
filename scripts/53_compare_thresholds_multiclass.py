from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CLASSIFICATION_RESULTS_DIR,
)

PREDICTIONS_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "source_balanced_model_comparison"
    / "validation_predictions.csv"
)
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "source_balanced_threshold_multiclass_comparison"

VIRAL_INDEX = CLASS_TO_INDEX["Viral"]
LABELS = list(range(len(CLASS_NAMES)))

# Candidate decision rules selected in the previous validation analysis.
# These are exploratory only; final threshold selection requires untouched test data.
SCENARIOS = [
    ("original_argmax", "original_baseline", None),
    ("source_balanced_argmax", "source_balanced", None),
    ("original_threshold_0_250", "original_baseline", 0.250),
    ("source_balanced_threshold_0_275", "source_balanced", 0.275),
]


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Run script 51 first. Missing: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)
    required = [
        "model",
        "processed_path",
        "actual_class_index",
        *[f"probability_{name.lower()}" for name in CLASS_NAMES],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    return df


def make_predictions(probabilities, threshold):
    if threshold is None:
        return np.argmax(probabilities, axis=1)

    nonviral_indices = [
        index for index in LABELS if index != VIRAL_INDEX
    ]
    best_nonviral = np.asarray(nonviral_indices)[
        np.argmax(probabilities[:, nonviral_indices], axis=1)
    ]
    predicted = best_nonviral.astype(np.int32)
    predicted[probabilities[:, VIRAL_INDEX] >= threshold] = VIRAL_INDEX
    return predicted


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive previous results before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions()
    probability_columns = [
        f"probability_{name.lower()}" for name in CLASS_NAMES
    ]

    overall_rows = []
    class_rows = []
    confusion_rows = []
    source_rows = []
    prediction_rows = []

    for scenario_name, model_name, threshold in SCENARIOS:
        model_df = df[df["model"].eq(model_name)].copy()
        if model_df.empty:
            raise ValueError(f"No prediction rows for model: {model_name}")

        y_true = model_df["actual_class_index"].astype(int).to_numpy()
        probabilities = model_df[probability_columns].astype(float).to_numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite probabilities for {model_name}")

        y_pred = make_predictions(probabilities, threshold)

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )

        overall_rows.append({
            "scenario": scenario_name,
            "source_model": model_name,
            "viral_threshold": threshold,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "validation_images": int(len(y_true)),
        })

        for index, class_name in enumerate(CLASS_NAMES):
            class_rows.append({
                "scenario": scenario_name,
                "class_name": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            })

        matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
        for actual_index, actual_name in enumerate(CLASS_NAMES):
            for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                confusion_rows.append({
                    "scenario": scenario_name,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(matrix[actual_index, predicted_index]),
                })

        if "source_dataset" in model_df.columns:
            for source in model_df["source_dataset"].dropna().unique():
                mask = model_df["source_dataset"].to_numpy() == source
                source_true = y_true[mask]
                source_pred = y_pred[mask]
                source_report = classification_report(
                    source_true,
                    source_pred,
                    labels=LABELS,
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )
                source_rows.append({
                    "scenario": scenario_name,
                    "source_dataset": source,
                    "total_images": int(mask.sum()),
                    "accuracy": float(accuracy_score(source_true, source_pred)),
                    "viral_precision": float(source_report["Viral"]["precision"]),
                    "viral_recall": float(source_report["Viral"]["recall"]),
                    "viral_f1": float(source_report["Viral"]["f1-score"]),
                    "normal_recall": float(source_report["Normal"]["recall"]),
                    "bacterial_recall": float(source_report["Bacterial"]["recall"]),
                    "covid_recall": float(source_report["COVID"]["recall"]),
                })

        pred_df = model_df[["processed_path", "actual_class_index"]].copy()
        pred_df["scenario"] = scenario_name
        pred_df["predicted_class_index"] = y_pred
        pred_df["actual_class"] = [
            CLASS_NAMES[index] for index in y_true
        ]
        pred_df["predicted_class"] = [
            CLASS_NAMES[index] for index in y_pred
        ]
        prediction_rows.append(pred_df)

    outputs = {
        "overall_multiclass_comparison.csv": pd.DataFrame(overall_rows),
        "class_wise_multiclass_comparison.csv": pd.DataFrame(class_rows),
        "multiclass_confusion_matrices.csv": pd.DataFrame(confusion_rows),
        "source_wise_multiclass_metrics.csv": pd.DataFrame(source_rows),
        "selected_rule_predictions.csv": pd.concat(prediction_rows, ignore_index=True),
    }

    for filename, output_df in outputs.items():
        output_df.to_csv(OUTPUT_DIR / filename, index=False)

    print("\nOVERALL MULTICLASS COMPARISON")
    print(outputs["overall_multiclass_comparison.csv"].to_string(index=False))

    print("\nCLASS-WISE METRICS")
    print(outputs["class_wise_multiclass_comparison.csv"].to_string(index=False))

    print("\nSOURCE-WISE METRICS")
    print(outputs["source_wise_multiclass_metrics.csv"].to_string(index=False))

    print("\nCONFUSION MATRICES (rows=actual, columns=predicted)")
    confusion_df = outputs["multiclass_confusion_matrices.csv"]
    for scenario in confusion_df["scenario"].drop_duplicates():
        subset = confusion_df[confusion_df["scenario"].eq(scenario)]
        matrix = (
            subset.pivot(
                index="actual_class",
                columns="predicted_class",
                values="count",
            )
            .reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
            .fillna(0)
            .astype(int)
        )
        print(f"\n{scenario}")
        print(matrix.to_string())

    print(f"\nSaved comparison files to:\n{OUTPUT_DIR}")
    print("\nValidation predictions only. Test split was not accessed.")
    print(
        "Thresholds were selected using validation data. "
        "Treat this as exploratory analysis, not an independent final evaluation."
    )


if __name__ == "__main__":
    main()


