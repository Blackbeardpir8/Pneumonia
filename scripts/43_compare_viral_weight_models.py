import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR, CLASSIFICATION_RESULTS_DIR, SAVED_MODELS_DIR,
    CLASS_NAMES, CLASS_TO_INDEX, IMAGE_SIZE, BATCH_SIZE,
)

MODELS = {
    "original": SAVED_MODELS_DIR / "efficientnetb0_pneumonia_classifier_best.keras",
    "viral_1_2x": SAVED_MODELS_DIR / "efficientnetb0_viral_weight_1.2x_best.keras",
    "viral_1_4x": SAVED_MODELS_DIR / "efficientnetb0_viral_weight_1.4x_best.keras",
}
MANIFEST = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUT = CLASSIFICATION_RESULTS_DIR / "viral_weight_model_comparison"
AUTOTUNE = tf.data.AUTOTUNE


def load_validation():
    df = pd.read_csv(MANIFEST)
    required = ["split", "class_name", "class_index", "processed_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    df = df[df["split"].astype(str).str.lower() == "val"].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Validation split is empty.")

    paths = []
    for value in df["processed_path"].astype(str):
        p = Path(value)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        paths.append(str(p))

    y = df["class_index"].astype(np.int32).to_numpy()
    for _, row in df.iterrows():
        if CLASS_TO_INDEX.get(row["class_name"]) != int(row["class_index"]):
            raise ValueError(f"Class mapping mismatch: {row['class_name']}")
    return df, paths, y


def preprocess(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    return tf.cast(image, tf.float32), label


def evaluate(name, path, ds, y_true):
    print("\n" + "=" * 68 + f"\nEvaluating {name}\n" + "=" * 68)
    model = tf.keras.models.load_model(path, compile=False)
    probs = model.predict(ds, verbose=1)
    if probs.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected prediction shape for {name}: {probs.shape}")
    pred = np.argmax(probs, axis=1)
    report = classification_report(
        y_true, pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true, pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0,
    ))
    cm = confusion_matrix(y_true, pred, labels=list(range(len(CLASS_NAMES))))
    print("Confusion matrix (actual rows, predicted columns):\n", cm)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
    }
    del model
    tf.keras.backend.clear_session()
    return {"probabilities": probs, "predictions": pred, "report": report,
            "confusion_matrix": cm, "metrics": metrics}


def save_predictions(name, result, df, y):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["report"]).transpose().to_csv(folder / "classification_report.csv")
    cm = pd.DataFrame(result["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    cm.index.name = "actual_class"
    cm.to_csv(folder / "confusion_matrix.csv")
    pred_df = df.copy()
    pred_df["actual_class_index"] = y
    pred_df["predicted_class_index"] = result["predictions"]
    pred_df["predicted_class"] = [CLASS_NAMES[i] for i in result["predictions"]]
    pred_df["correct"] = pred_df["actual_class_index"] == pred_df["predicted_class_index"]
    for i, class_name in enumerate(CLASS_NAMES):
        pred_df[f"probability_{class_name.lower()}"] = result["probabilities"][:, i]
    pred_df.to_csv(folder / "validation_predictions.csv", index=False)


def save_comparisons(results, df, y):
    overall, classes, viral_errors, sources = [], [], [], []
    vi, bi = CLASS_TO_INDEX["Viral"], CLASS_TO_INDEX["Bacterial"]
    actual_viral = y == vi

    for name, result in results.items():
        overall.append(result["metrics"])
        for class_name in CLASS_NAMES:
            m = result["report"][class_name]
            classes.append({
                "model": name, "class": class_name,
                "precision": m["precision"], "recall": m["recall"],
                "f1_score": m["f1-score"], "support": int(m["support"]),
            })

        viral_pred = result["predictions"][actual_viral]
        viral_errors.append({
            "model": name, "actual_viral_support": int(actual_viral.sum()),
            "viral_correct": int(np.sum(viral_pred == vi)),
            "viral_to_bacterial": int(np.sum(viral_pred == bi)),
            "viral_to_normal": int(np.sum(viral_pred == CLASS_TO_INDEX["Normal"])),
            "viral_to_covid": int(np.sum(viral_pred == CLASS_TO_INDEX["COVID"])),
        })

        if "source_dataset" in df.columns:
            temp = df.loc[actual_viral, ["source_dataset"]].copy()
            temp["prediction"] = viral_pred
            for source, group in temp.groupby("source_dataset", dropna=False):
                p = group["prediction"].to_numpy()
                sources.append({
                    "model": name, "source_dataset": str(source),
                    "actual_viral_support": len(p),
                    "viral_recall": float(np.mean(p == vi)),
                    "viral_to_bacterial": int(np.sum(p == bi)),
                    "viral_to_normal": int(np.sum(p == CLASS_TO_INDEX["Normal"])),
                    "viral_to_covid": int(np.sum(p == CLASS_TO_INDEX["COVID"])),
                })

    pd.DataFrame(overall).to_csv(OUT / "overall_model_comparison.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "class_wise_comparison.csv", index=False)
    pd.DataFrame(viral_errors).to_csv(OUT / "viral_misclassification_comparison.csv", index=False)
    if sources:
        pd.DataFrame(sources).to_csv(OUT / "viral_source_wise_comparison.csv", index=False)


def main():
    for p in [MANIFEST, *MODELS.values()]:
        if not p.exists():
            raise FileNotFoundError(p)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUT}. Archive it before rerunning.")
    df, paths, y = load_validation()
    print("Validation images:", len(y))
    print("Validation distribution:\n", df["class_name"].value_counts().reindex(CLASS_NAMES))
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in MODELS.items():
        results[name] = evaluate(name, path, ds, y)
        save_predictions(name, results[name], df, y)
    save_comparisons(results, df, y)

    print("\nSaved comparison results to:", OUT)
    print("Files: overall_model_comparison.csv, class_wise_comparison.csv,")
    print("       viral_misclassification_comparison.csv")
    if "source_dataset" in df.columns:
        print("       viral_source_wise_comparison.csv")
    print("Test dataset was not used.")


if __name__ == "__main__":
    main()


