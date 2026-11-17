
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "loss_function_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENTS = {
    "original_bce_dice": {
        "label": "Original BCE + Dice",
        "history": RESULTS_ROOT / "training_history.csv",
    },
    "weighted_bce_dice": {
        "label": "Weighted BCE + Dice",
        "history": (
            RESULTS_ROOT
            / "loss_function_experiments"
            / "weighted_bce_dice"
            / "training_history.csv"
        ),
    },
    "tversky": {
        "label": "Tversky",
        "history": (
            RESULTS_ROOT
            / "loss_function_experiments"
            / "tversky"
            / "training_history.csv"
        ),
    },
}


# ============================================================
# 2. REQUIRED HISTORY COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "epoch",
    "loss",
    "dice_coefficient",
    "iou_score",
    "precision",
    "recall",
    "val_loss",
    "val_dice_coefficient",
    "val_iou_score",
    "val_precision",
    "val_recall",
]


# ============================================================
# 3. LOAD AND VALIDATE HISTORY
# ============================================================

def load_experiment_history(name, config):

    history_path = config["history"]

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found for {name}:\n"
            f"{history_path}"
        )

    df = pd.read_csv(history_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            f"Training history is empty: {history_path}"
        )

    df = df.copy()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df["val_dice_coefficient"].isna().all():
        raise ValueError(
            f"No valid validation Dice values in {history_path}"
        )

    # Select the epoch with the highest validation Dice.
    best_index = df["val_dice_coefficient"].idxmax()
    best_row = df.loc[best_index]

    result = {
        "experiment": name,
        "label": config["label"],
        "epochs_completed": int(len(df)),
        "best_epoch": int(best_row["epoch"]) + 1,
        "train_dice": float(best_row["dice_coefficient"]),
        "validation_dice": float(
            best_row["val_dice_coefficient"]
        ),
        "train_iou": float(best_row["iou_score"]),
        "validation_iou": float(
            best_row["val_iou_score"]
        ),
        "train_precision": float(best_row["precision"]),
        "validation_precision": float(
            best_row["val_precision"]
        ),
        "train_recall": float(best_row["recall"]),
        "validation_recall": float(
            best_row["val_recall"]
        ),
        "train_loss": float(best_row["loss"]),
        "validation_loss": float(best_row["val_loss"]),
        "history_path": str(history_path),
    }

    print("\n" + "-" * 65)
    print(config["label"])
    print("-" * 65)

    print("Epochs completed:", result["epochs_completed"])
    print("Best epoch:", result["best_epoch"])

    print(
        "Validation Dice:",
        round(result["validation_dice"], 4),
    )

    print(
        "Validation IoU:",
        round(result["validation_iou"], 4),
    )

    print(
        "Validation Precision:",
        round(result["validation_precision"], 4),
    )

    print(
        "Validation Recall:",
        round(result["validation_recall"], 4),
    )

    print(
        "Validation Loss:",
        round(result["validation_loss"], 4),
    )

    return df, result


# ============================================================
# 4. CREATE TRAINING CURVES
# ============================================================

def create_training_curves(histories):

    # Validation and training Dice.
    plt.figure(figsize=(11, 6))

    for name, config in EXPERIMENTS.items():

        df = histories[name]

        plt.plot(
            df["epoch"] + 1,
            df["dice_coefficient"],
            linestyle="--",
            label=f'{config["label"]} - Train',
        )

        plt.plot(
            df["epoch"] + 1,
            df["val_dice_coefficient"],
            label=f'{config["label"]} - Validation',
        )

    plt.title("Infection U-Net: Training and Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    dice_path = OUTPUT_DIR / "dice_comparison.png"

    plt.savefig(
        dice_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Validation loss.
    plt.figure(figsize=(11, 6))

    for name, config in EXPERIMENTS.items():

        df = histories[name]

        plt.plot(
            df["epoch"] + 1,
            df["val_loss"],
            label=config["label"],
        )

    plt.title("Infection U-Net: Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_path = OUTPUT_DIR / "validation_loss_comparison.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nTraining charts saved:")
    print(dice_path)
    print(loss_path)


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INFECTION SEGMENTATION - LOSS FUNCTION COMPARISON")
    print("=" * 70)

    histories = {}
    results = []

    for name, config in EXPERIMENTS.items():

        df, result = load_experiment_history(
            name,
            config,
        )

        histories[name] = df
        results.append(result)

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="validation_dice",
        ascending=False,
    ).reset_index(drop=True)

    comparison_df.insert(
        0,
        "validation_rank",
        range(1, len(comparison_df) + 1),
    )

    # Save detailed comparison CSV.
    csv_path = OUTPUT_DIR / "loss_function_comparison.csv"

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    # Save detailed comparison JSON.
    json_path = OUTPUT_DIR / "loss_function_comparison.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison_df.to_dict(orient="records"),
            file,
            indent=4,
        )

    # Generate training charts.
    create_training_curves(histories)

    # Display ranking.
    print("\n" + "=" * 70)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 70)

    display_columns = [
        "validation_rank",
        "label",
        "best_epoch",
        "validation_dice",
        "validation_iou",
        "validation_precision",
        "validation_recall",
        "validation_loss",
    ]

    print(
        comparison_df[display_columns].to_string(
            index=False
        )
    )

    best = comparison_df.iloc[0]

    print("\n" + "=" * 70)
    print("BEST VALIDATION CANDIDATE")
    print("=" * 70)

    print("Experiment:", best["label"])
    print("Best epoch:", int(best["best_epoch"]))
    print(
        "Validation Dice:",
        round(best["validation_dice"], 4),
    )

    print("\nComparison CSV:", csv_path)
    print("Comparison JSON:", json_path)
    print("Output directory:", OUTPUT_DIR)

    print(
        "\nNOTE: This is validation-based model comparison. "
        "The held-out test set was not used."
    )


if __name__ == "__main__":
    main()


# # to run this script, use the following command:
# python scripts/113_compare_infection_loss_results.py