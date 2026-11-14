from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    CLASS_TO_INDEX,
)

PREDICTION_ROOT = CLASSIFICATION_RESULTS_DIR / "viral_weight_model_comparison"
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "viral_source_threshold_analysis"

# Thresholds are taken from the prior validation-only threshold sweep.
# Each model has two operating points:
# 1) Best Viral F1
# 2) Best Viral recall among thresholds with Viral precision >= 0.80
OPERATING_POINTS = {
    "original": {
        "best_viral_f1": 0.200,
        "precision_at_least_080": 0.250,
    },
    "viral_1_2x": {
        "best_viral_f1": 0.250,
        "precision_at_least_080": 0.325,
    },
    "viral_1_4x": {
        "best_viral_f1": 0.275,
        "precision_at_least_080": 0.350,
    },
}

VIRAL = CLASS_TO_INDEX["Viral"]
NORMAL = CLASS_TO_INDEX["Normal"]
BACTERIAL = CLASS_TO_INDEX["Bacterial"]
COVID = CLASS_TO_INDEX["COVID"]


def load_predictions(model_name):
    path = PREDICTION_ROOT / model_name / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = ["actual_class_index", "source_dataset"] + [
        f"probability_{name.lower()}" for name in CLASS_NAMES
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    y_true = df["actual_class_index"].astype(int).to_numpy()
    probability_columns = [
        f"probability_{name.lower()}" for name in CLASS_NAMES
    ]
    probabilities = df[probability_columns].astype(float).to_numpy()

    if len(df) == 0:
        raise ValueError(f"No validation rows in {path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {path}")

    return df, y_true, probabilities


def predict_with_viral_threshold(probabilities, threshold):
    # If P(Viral) reaches threshold, choose Viral.
    # Otherwise, select the highest-probability non-Viral class.
    nonviral_indices = [i for i in range(len(CLASS_NAMES)) if i != VIRAL]
    nonviral_best = np.asarray(nonviral_indices)[
        np.argmax(probabilities[:, nonviral_indices], axis=1)
    ]

    predictions = nonviral_best.astype(np.int32)
    predictions[probabilities[:, VIRAL] >= threshold] = VIRAL
    return predictions


def analyze_group(y_true, y_pred, group_mask, source, model, point, threshold):
    actual = y_true[group_mask]
    predicted = y_pred[group_mask]

    actual_viral = actual == VIRAL
    predicted_viral = predicted == VIRAL

    viral_support = int(actual_viral.sum())
    viral_tp = int(np.sum(actual_viral & predicted_viral))
    viral_fn = viral_support - viral_tp
    viral_fp = int(np.sum(~actual_viral & predicted_viral))

    precision = (
        viral_tp / (viral_tp + viral_fp)
        if (viral_tp + viral_fp) else 0.0
    )
    recall = viral_tp / viral_support if viral_support else 0.0

    return {
        "model": model,
        "operating_point": point,
        "viral_threshold": threshold,
        "source_dataset": source,
        "total_images": int(len(actual)),
        "actual_viral_support": viral_support,
        "viral_true_positives": viral_tp,
        "viral_false_negatives": viral_fn,
        "viral_false_positives": viral_fp,
        "viral_precision": precision,
        "viral_recall": recall,
        "actual_normal_predicted_viral": int(
            np.sum((actual == NORMAL) & predicted_viral)
        ),
        "actual_bacterial_predicted_viral": int(
            np.sum((actual == BACTERIAL) & predicted_viral)
        ),
        "actual_covid_predicted_viral": int(
            np.sum((actual == COVID) & predicted_viral)
        ),
        "viral_predicted_as_bacterial": int(
            np.sum(actual_viral & (predicted == BACTERIAL))
        ),
        "viral_predicted_as_normal": int(
            np.sum(actual_viral & (predicted == NORMAL))
        ),
        "viral_predicted_as_covid": int(
            np.sum(actual_viral & (predicted == COVID))
        ),
    }


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive its contents before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_rows = []
    overall_rows = []

    for model_name, points in OPERATING_POINTS.items():
        df, y_true, probabilities = load_predictions(model_name)

        for point_name, threshold in points.items():
            y_pred = predict_with_viral_threshold(probabilities, threshold)

            # Overall operating-point metrics, plus source breakdown.
            actual_viral = y_true == VIRAL
            predicted_viral = y_pred == VIRAL
            tp = int(np.sum(actual_viral & predicted_viral))
            fp = int(np.sum(~actual_viral & predicted_viral))
            fn = int(np.sum(actual_viral & ~predicted_viral))

            overall_rows.append({
                "model": model_name,
                "operating_point": point_name,
                "viral_threshold": threshold,
                "total_images": len(y_true),
                "viral_support": int(actual_viral.sum()),
                "viral_true_positives": tp,
                "viral_false_positives": fp,
                "viral_false_negatives": fn,
                "viral_precision": tp / (tp + fp) if tp + fp else 0.0,
                "viral_recall": tp / (tp + fn) if tp + fn else 0.0,
            })

            for source, source_df in df.groupby("source_dataset", dropna=False):
                mask = df["source_dataset"].eq(source).to_numpy()
                source_rows.append(
                    analyze_group(
                        y_true, y_pred, mask, str(source),
                        model_name, point_name, threshold,
                    )
                )

    overall_df = pd.DataFrame(overall_rows)
    source_df = pd.DataFrame(source_rows)

    overall_df.to_csv(OUTPUT_DIR / "overall_threshold_metrics.csv", index=False)
    source_df.to_csv(OUTPUT_DIR / "source_wise_threshold_metrics.csv", index=False)

    print("\nOVERALL THRESHOLD METRICS")
    print(overall_df.to_string(index=False))

    print("\nSOURCE-WISE THRESHOLD METRICS")
    print(source_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "overall_threshold_metrics.csv")
    print(OUTPUT_DIR / "source_wise_threshold_metrics.csv")
    print("\nValidation predictions only. Test split was not accessed.")
    print(
        "These operating points were selected using the validation set, "
        "so results remain exploratory until evaluated on untouched test data."
    )


if __name__ == "__main__":
    main()
