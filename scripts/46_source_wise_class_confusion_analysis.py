from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import CLASS_NAMES, CLASS_TO_INDEX, CLASSIFICATION_RESULTS_DIR

PREDICTION_ROOT = CLASSIFICATION_RESULTS_DIR / "viral_weight_model_comparison"
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "source_wise_class_confusion"

MODELS = ["original", "viral_1_2x", "viral_1_4x"]
PROBABILITY_COLUMNS = [f"probability_{name.lower()}" for name in CLASS_NAMES]


def load_model_predictions(model_name):
    path = PREDICTION_ROOT / model_name / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = ["actual_class_index", "source_dataset", *PROBABILITY_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    y_true = df["actual_class_index"].astype(int).to_numpy()
    probabilities = df[PROBABILITY_COLUMNS].astype(float).to_numpy()

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite prediction probabilities in {path}")

    y_pred = np.argmax(probabilities, axis=1)
    return df, y_true, y_pred


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Move or archive the existing results before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    class_rows = []
    confusion_rows = []

    labels = list(range(len(CLASS_NAMES)))

    for model_name in MODELS:
        df, y_true, y_pred = load_model_predictions(model_name)

        for source, source_df in df.groupby("source_dataset", dropna=False):
            mask = df["source_dataset"].eq(source).to_numpy()
            source_true = y_true[mask]
            source_pred = y_pred[mask]
            source_name = str(source)

            overall_rows.append({
                "model": model_name,
                "source_dataset": source_name,
                "total_images": int(len(source_true)),
                "accuracy": float(accuracy_score(source_true, source_pred)),
            })

            report = classification_report(
                source_true,
                source_pred,
                labels=labels,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )

            for class_name in CLASS_NAMES:
                metrics = report[class_name]
                class_rows.append({
                    "model": model_name,
                    "source_dataset": source_name,
                    "class_name": class_name,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1-score"]),
                    "support": int(metrics["support"]),
                })

            matrix = confusion_matrix(
                source_true, source_pred, labels=labels
            )

            # Rows are actual classes; columns are predicted classes.
            for actual_index, actual_name in enumerate(CLASS_NAMES):
                for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                    confusion_rows.append({
                        "model": model_name,
                        "source_dataset": source_name,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[actual_index, predicted_index]),
                    })

    overall_df = pd.DataFrame(overall_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    overall_df.to_csv(OUTPUT_DIR / "source_overall_accuracy.csv", index=False)
    class_df.to_csv(OUTPUT_DIR / "source_classification_metrics.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "source_confusion_matrices.csv", index=False)

    print("\nSOURCE-WISE OVERALL ACCURACY")
    print(overall_df.to_string(index=False))

    print("\nSOURCE-WISE CLASS METRICS")
    print(class_df.to_string(index=False))

    print("\nSOURCE-WISE CONFUSION MATRICES")
    for (model, source), group in confusion_df.groupby(
        ["model", "source_dataset"], sort=False
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
        print(f"\nModel: {model} | Source: {source}")
        print("Rows = actual classes; columns = predicted classes")
        print(matrix.to_string())

    print("\nSaved results to:")
    print(OUTPUT_DIR)
    print("\nValidation predictions only. Test split was not accessed.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/46_source_wise_class_confusion_analysis.py