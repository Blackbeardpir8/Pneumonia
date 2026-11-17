from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Run from the project structure created by scripts/43_compare_viral_weight_models.py.
ROOT = Path(__file__).resolve().parent.parent

from sys import path as sys_path
if str(ROOT) not in sys_path:
    sys_path.insert(0, str(ROOT))

from aiml.config import (
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    CLASS_TO_INDEX,
)

PREDICTION_ROOT = (
    CLASSIFICATION_RESULTS_DIR / "viral_weight_model_comparison"
)
OUTPUT_DIR = (
    CLASSIFICATION_RESULTS_DIR / "viral_threshold_analysis"
)

MODEL_NAMES = ["original", "viral_1_2x", "viral_1_4x"]

# Fixed grid for an exploratory validation-only threshold sweep.
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.025), 3)

VIRAL_INDEX = CLASS_TO_INDEX["Viral"]
NON_VIRAL_INDICES = [
    CLASS_TO_INDEX[name] for name in CLASS_NAMES if name != "Viral"
]


def load_predictions(model_name):
    csv_path = PREDICTION_ROOT / model_name / "validation_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [
        "actual_class_index",
        "probability_viral",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    probability_columns = [
        f"probability_{name.lower()}" for name in CLASS_NAMES
    ]
    missing_probs = [c for c in probability_columns if c not in df.columns]
    if missing_probs:
        raise ValueError(f"{csv_path} is missing probability columns: {missing_probs}")

    y_true = df["actual_class_index"].astype(int).to_numpy()
    probabilities = df[probability_columns].astype(float).to_numpy()

    if len(y_true) == 0:
        raise ValueError(f"No validation predictions in {csv_path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities found in {csv_path}")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"Probabilities outside [0, 1] in {csv_path}")

    return df, y_true, probabilities


def apply_viral_threshold(probabilities, threshold):
    """
    If P(Viral) >= threshold, predict Viral.
    Otherwise, choose the highest-probability non-Viral class.

    This is a one-vs-rest decision rule; it does not retrain or recalibrate
    the model probabilities.
    """
    predictions = np.empty(len(probabilities), dtype=np.int32)
    viral_selected = probabilities[:, VIRAL_INDEX] >= threshold

    nonviral_winner_positions = np.argmax(
        probabilities[:, NON_VIRAL_INDICES],
        axis=1,
    )
    nonviral_winners = np.asarray(NON_VIRAL_INDICES)[
        nonviral_winner_positions
    ]

    predictions[viral_selected] = VIRAL_INDEX
    predictions[~viral_selected] = nonviral_winners[~viral_selected]
    return predictions


def calculate_row(model_name, y_true, probabilities, threshold):
    y_pred = apply_viral_threshold(probabilities, threshold)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    viral = report["Viral"]
    actual_viral = y_true == VIRAL_INDEX
    predicted_viral = y_pred == VIRAL_INDEX

    return {
        "model": model_name,
        "viral_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "viral_precision": float(viral["precision"]),
        "viral_recall": float(viral["recall"]),
        "viral_f1": float(viral["f1-score"]),
        "viral_support": int(viral["support"]),
        "predicted_viral_count": int(predicted_viral.sum()),
        "viral_false_positives": int(
            np.sum(predicted_viral & ~actual_viral)
        ),
        "actual_normal_predicted_viral": int(
            np.sum((y_true == CLASS_TO_INDEX["Normal"]) & predicted_viral)
        ),
        "actual_bacterial_predicted_viral": int(
            np.sum((y_true == CLASS_TO_INDEX["Bacterial"]) & predicted_viral)
        ),
        "actual_covid_predicted_viral": int(
            np.sum((y_true == CLASS_TO_INDEX["COVID"]) & predicted_viral)
        ),
    }


def add_source_metrics(rows, model_name, df, y_true, probabilities, threshold):
    if "source_dataset" not in df.columns:
        return

    y_pred = apply_viral_threshold(probabilities, threshold)
    viral_mask = y_true == VIRAL_INDEX

    source_df = df.loc[viral_mask, ["source_dataset"]].copy()
    source_df["predicted_class_index"] = y_pred[viral_mask]

    for source, group in source_df.groupby("source_dataset", dropna=False):
        predictions = group["predicted_class_index"].to_numpy()
        rows.append({
            "model": model_name,
            "viral_threshold": float(threshold),
            "source_dataset": str(source),
            "actual_viral_support": int(len(predictions)),
            "viral_recall": float(np.mean(predictions == VIRAL_INDEX)),
            "viral_to_bacterial": int(
                np.sum(predictions == CLASS_TO_INDEX["Bacterial"])
            ),
            "viral_to_normal": int(
                np.sum(predictions == CLASS_TO_INDEX["Normal"])
            ),
            "viral_to_covid": int(
                np.sum(predictions == CLASS_TO_INDEX["COVID"])
            ),
        })


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive its contents before rerunning."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    source_rows = []
    baseline_rows = []

    for model_name in MODEL_NAMES:
        df, y_true, probabilities = load_predictions(model_name)
        print(f"\nLoaded {model_name}: {len(y_true)} validation predictions")

        # Original argmax performance, retained as a reference point.
        argmax_predictions = np.argmax(probabilities, axis=1)
        argmax_report = classification_report(
            y_true,
            argmax_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        baseline_rows.append({
            "model": model_name,
            "decision_rule": "standard_argmax",
            "accuracy": float(accuracy_score(y_true, argmax_predictions)),
            "macro_f1": float(f1_score(
                y_true, argmax_predictions, average="macro", zero_division=0
            )),
            "viral_precision": float(argmax_report["Viral"]["precision"]),
            "viral_recall": float(argmax_report["Viral"]["recall"]),
            "viral_f1": float(argmax_report["Viral"]["f1-score"]),
        })

        for threshold in THRESHOLDS:
            row = calculate_row(
                model_name, y_true, probabilities, float(threshold)
            )
            all_rows.append(row)
            add_source_metrics(
                source_rows, model_name, df, y_true,
                probabilities, float(threshold)
            )

    results = pd.DataFrame(all_rows)
    baselines = pd.DataFrame(baseline_rows)

    results.to_csv(OUTPUT_DIR / "threshold_sweep_all_models.csv", index=False)
    baselines.to_csv(OUTPUT_DIR / "standard_argmax_baselines.csv", index=False)

    if source_rows:
        pd.DataFrame(source_rows).to_csv(
            OUTPUT_DIR / "source_wise_viral_recall.csv", index=False
        )

    # Best threshold by Viral F1 for each model.
    best_f1_indices = results.groupby("model")["viral_f1"].idxmax()
    best_by_viral_f1 = results.loc[best_f1_indices].sort_values("model")
    best_by_viral_f1.to_csv(
        OUTPUT_DIR / "best_threshold_by_viral_f1.csv", index=False
    )

    # Also report the best Viral recall subject to a minimum precision of 0.80.
    precision_constrained = results[results["viral_precision"] >= 0.80]
    if not precision_constrained.empty:
        best_recall_indices = precision_constrained.groupby("model")[
            "viral_recall"
        ].idxmax()
        best_by_recall = precision_constrained.loc[
            best_recall_indices
        ].sort_values("model")
        best_by_recall.to_csv(
            OUTPUT_DIR / "best_threshold_recall_precision_at_least_080.csv",
            index=False,
        )

    print("\nSTANDARD ARGMAX BASELINES")
    print(baselines.to_string(index=False))

    print("\nBEST THRESHOLD BY VIRAL F1")
    display_columns = [
        "model", "viral_threshold", "accuracy", "macro_f1",
        "viral_precision", "viral_recall", "viral_f1",
        "viral_false_positives", "actual_bacterial_predicted_viral",
    ]
    print(best_by_viral_f1[display_columns].to_string(index=False))

    if not precision_constrained.empty:
        print("\nBEST VIRAL RECALL WITH PRECISION >= 0.80")
        print(best_by_recall[display_columns].to_string(index=False))
    else:
        print("\nNo threshold reached Viral precision >= 0.80.")

    print("\nOutput directory:", OUTPUT_DIR)
    print("Validation data only. The test split was not accessed.")
    print(
        "Thresholds are exploratory and selected on validation data; "
        "do not treat their best scores as unbiased final test performance."
    )


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/44_viral_threshold_analysis.py
