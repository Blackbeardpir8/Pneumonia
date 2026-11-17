
import json
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT /
    "results/classification/final_test_evaluation"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results/classification/source_wise_analysis"
)

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

SOURCES = [
    "covid19_radiography",
    "bacterial_viral_dataset",
]

CANDIDATES = [
    "original_threshold_0_250",
    "source_balanced_threshold_0_275",
]

DECISION_RULES = [
    "argmax",
    "threshold",
]


def calculate_source_metrics(df, source, candidate, rule):

    subset = df[df["source_dataset"] == source].copy()

    if subset.empty:
        return None

    prediction_column = f"{rule}_predicted_class"
    y_true = subset["actual_class"]
    y_pred = subset[prediction_column]

    # Evaluate only classes actually represented in this source.
    supported_classes = [
        name for name in CLASS_NAMES
        if name in set(y_true)
    ]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=supported_classes,
            average=None,
            zero_division=0,
        )
    )

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=supported_classes,
            average="macro",
            zero_division=0,
        )
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=supported_classes,
            average="weighted",
            zero_division=0,
        )
    )

    class_report = classification_report(
        y_true,
        y_pred,
        labels=supported_classes,
        target_names=supported_classes,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=supported_classes,
    )

    metrics = {
        "candidate": candidate,
        "decision_rule": rule,
        "source_dataset": source,
        "image_count": len(subset),
        "supported_classes": supported_classes,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "class_report": class_report,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_details = []

    for candidate in CANDIDATES:

        prediction_path = (
            INPUT_DIR / f"{candidate}_predictions.csv"
        )

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        df = pd.read_csv(prediction_path)

        required_columns = {
            "actual_class",
            "source_dataset",
            "argmax_predicted_class",
            "threshold_predicted_class",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {prediction_path}: {missing}"
            )

        print("\nEvaluating:", candidate)

        for source in SOURCES:

            for rule in DECISION_RULES:

                result = calculate_source_metrics(
                    df,
                    source,
                    candidate,
                    rule,
                )

                if result is None:
                    continue

                all_details.append(result)

                summary_rows.append({
                    "candidate": candidate,
                    "decision_rule": rule,
                    "source_dataset": source,
                    "image_count": result["image_count"],
                    "supported_classes": ", ".join(
                        result["supported_classes"]
                    ),
                    "accuracy": result["accuracy"],
                    "macro_precision": result["macro_precision"],
                    "macro_recall": result["macro_recall"],
                    "macro_f1": result["macro_f1"],
                    "weighted_f1": result["weighted_f1"],
                })

                # Save source-specific confusion matrix.
                labels = result["supported_classes"]

                cm_df = pd.DataFrame(
                    result["confusion_matrix"],
                    index=labels,
                    columns=labels,
                )

                cm_df.index.name = "actual_class"

                cm_df.to_csv(
                    OUTPUT_DIR /
                    f"{candidate}_{rule}_{source}_confusion_matrix.csv"
                )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "source_wise_metrics.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "source_wise_detailed_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_details,
            file,
            indent=4,
        )

    print("\n" + "=" * 75)
    print("SOURCE-WISE CLASSIFIER PERFORMANCE")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved results to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/73_classifier_source_wise_analysis.py