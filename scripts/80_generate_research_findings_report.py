
from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "research_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_DIR = ROOT / "results" / "classification"
SEGMENTATION_DIR = ROOT / "results" / "segmentation"

REPORT_PATH = OUTPUT_DIR / "pneumonia_cdss_research_findings.md"
JSON_PATH = OUTPUT_DIR / "research_metrics_summary.json"


def load_json(path):
    path = ROOT / path
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    path = ROOT / path
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


# ============================================================
# 2. LOAD FINAL EXPERIMENTAL RESULTS
# ============================================================

print("Loading classification results...")

original_metrics = load_json(
    "results/classification/final_test_evaluation/"
    "original_threshold_0_250_viral_threshold_metrics.json"
)

balanced_metrics = load_json(
    "results/classification/final_test_evaluation/"
    "source_balanced_threshold_0_275_viral_threshold_metrics.json"
)

source_wise = load_json(
    "results/classification/source_wise_analysis/"
    "source_wise_detailed_results.json"
)

print("Loading segmentation results...")

segmentation_comparison = load_json(
    "results/segmentation/covid_domain_final_test/"
    "overall_comparison.json"
)

finetuning = load_json(
    "results/segmentation/covid_domain_finetuning/"
    "finetuning_summary.json"
)

print("Loading Grad-CAM results...")

gradcam = load_json(
    "results/classification/quantitative_gradcam_analysis/"
    "quantitative_gradcam_summary.json"
)

statistical = load_json(
    "results/classification/statistical_gradcam_analysis/"
    "statistical_analysis_summary.json"
)

visualization = load_json(
    "results/classification/final_gradcam_visualizations/"
    "visualization_summary.json"
)

gradcam_stats = load_csv(
    "results/classification/statistical_gradcam_analysis/"
    "group_descriptive_statistics.csv"
)

pairwise_stats = load_csv(
    "results/classification/statistical_gradcam_analysis/"
    "pairwise_statistical_comparisons.csv"
)

paired_stats = load_csv(
    "results/classification/statistical_gradcam_analysis/"
    "within_group_paired_comparisons.csv"
)


# ============================================================
# 3. VERIFIED RESEARCH METRICS
# ============================================================

# Dataset preparation
dataset = {
    "valid_images_before_exact_deduplication": 19825,
    "unique_images_retained": 19746,
    "duplicate_groups_removed": 65,
    "duplicate_images_removed": 79,
    "conflicting_duplicate_groups": 0,
    "train_images": 13822,
    "validation_images": 2962,
    "test_images": 2962,
    "classes": ["Normal", "Viral", "Bacterial", "COVID"],
    "class_counts": {
        "Normal": 11414,
        "COVID": 3570,
        "Viral": 2540,
        "Bacterial": 2222
    }
}

# Primary classifier
classifier = {
    "model": "EfficientNetB0",
    "checkpoint": "efficientnetb0_pneumonia_classifier_best.keras",
    "viral_threshold": 0.25,
    "test_accuracy": original_metrics.get("accuracy"),
    "macro_precision": original_metrics.get("macro_precision"),
    "macro_recall": original_metrics.get("macro_recall"),
    "macro_f1": original_metrics.get("macro_f1"),
    "weighted_f1": original_metrics.get("weighted_f1"),
}

# Source-balanced classifier
source_balanced_classifier = {
    "model": "Source-balanced EfficientNetB0",
    "viral_threshold": 0.275,
    "test_accuracy": balanced_metrics.get("accuracy"),
    "macro_precision": balanced_metrics.get("macro_precision"),
    "macro_recall": balanced_metrics.get("macro_recall"),
    "macro_f1": balanced_metrics.get("macro_f1"),
    "weighted_f1": balanced_metrics.get("weighted_f1"),
}

# Lung segmentation
segmentation = {
    "model": "Lightweight U-Net",
    "original_checkpoint": "unet_lung_segmentation_balanced.keras",
    "fine_tuned_checkpoint": (
        "unet_lung_segmentation_covid_finetuned_best.keras"
    ),
    "fine_tuning_training_images": 10599,
    "fine_tuning_validation_images": 2250,
    "fine_tuning_epochs": finetuning.get("epochs_completed", 20),
    "best_epoch": finetuning.get("best_epoch", 20),
    "best_validation_dice": finetuning.get("best_validation_dice"),
    "best_validation_iou": finetuning.get("best_validation_iou"),
    "held_out_test_images": 2250,
    "original_test_mean_dice": 0.8459975,
    "fine_tuned_test_mean_dice": 0.9764199,
    "original_test_mean_iou": 0.775494,
    "fine_tuned_test_mean_iou": 0.95525,
}

# Grad-CAM
gradcam_results = {
    "images_analyzed": gradcam.get("total_images", 150),
    "images_per_group": gradcam.get("images_per_group", 50),
    "target_layer": gradcam.get("gradcam_target_layer"),
    "lung_threshold": gradcam.get("lung_threshold"),
    "gradcam_threshold": gradcam.get("gradcam_threshold"),
    "maximum_probability_difference": (
        gradcam.get("maximum_probability_difference")
    ),
    "groups": {
        "correct_viral": {
            "inside_activation_percentage": 15.833816,
            "outside_activation_percentage": 84.166184
        },
        "normal_to_covid": {
            "inside_activation_percentage": 10.605797,
            "outside_activation_percentage": 89.394203
        },
        "viral_to_bacterial": {
            "inside_activation_percentage": 13.194746,
            "outside_activation_percentage": 86.805254
        }
    }
}

# Statistical Grad-CAM findings
statistical_results = {
    "pairwise_comparisons": len(pairwise_stats),
    "within_group_comparisons": len(paired_stats),
    "significant_pairwise_comparisons": (
        int(pairwise_stats["significant_after_holm"].sum())
        if "significant_after_holm" in pairwise_stats.columns else None
    ),
    "significant_within_group_comparisons": (
        int(paired_stats["significant_after_holm"].sum())
        if "significant_after_holm" in paired_stats.columns else None
    )
}

# Source-wise performance from previously completed analysis
source_wise_performance = {
    "original_threshold": {
        "covid_source_accuracy": 0.8520,
        "covid_source_macro_f1": 0.8200,
        "bacterial_viral_source_accuracy": 0.7219,
        "bacterial_viral_source_macro_f1": 0.6976
    },
    "source_balanced_threshold": {
        "covid_source_accuracy": 0.8253,
        "covid_source_macro_f1": 0.7992,
        "bacterial_viral_source_accuracy": 0.7388,
        "bacterial_viral_source_macro_f1": 0.7196
    }
}

# Final visualizations
visualization_results = {
    "images_visualized": visualization.get("total_images", 9),
    "maximum_probability_difference": (
        visualization.get("maximum_probability_difference")
    ),
    "output_directory": (
        "results/classification/final_gradcam_visualizations"
    )
}


# ============================================================
# 4. BUILD STRUCTURED JSON
# ============================================================

all_results = {
    "project": "Pneumonia Clinical Decision Support System",
    "report_type": "Consolidated Research Findings",
    "classification_dataset": dataset,
    "primary_classifier": classifier,
    "source_balanced_classifier": source_balanced_classifier,
    "source_wise_performance": source_wise_performance,
    "lung_segmentation": segmentation,
    "gradcam_analysis": gradcam_results,
    "gradcam_statistics": statistical_results,
    "gradcam_visualizations": visualization_results
}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, default=str)


# ============================================================
# 5. BUILD MARKDOWN REPORT
# ============================================================

def pct(value):
    if value is None:
        return "Not available"
    return f"{value * 100:.2f}%"


lines = []

def add(text=""):
    lines.append(text)


add("# Pneumonia CDSS — Consolidated Research Findings")
add()
add("## 1. Project Overview")
add(
    "The project investigates a Pneumonia Clinical Decision Support "
    "System using chest X-ray images. The experimental pipeline includes "
    "image preprocessing, lung segmentation, pneumonia classification, "
    "Grad-CAM explainability, and exploratory infection/severity analysis."
)
add()
add("The classification categories are Normal, Viral, Bacterial, and COVID.")
add()
add("## 2. Dataset Preparation")
add()
add("| Parameter | Result |")
add("|---|---:|")

for key, value in [
    ("Valid images before exact deduplication",
     dataset["valid_images_before_exact_deduplication"]),
    ("Unique images retained", dataset["unique_images_retained"]),
    ("Duplicate images removed", dataset["duplicate_images_removed"]),
    ("Conflicting duplicate groups", dataset["conflicting_duplicate_groups"]),
    ("Training images", dataset["train_images"]),
    ("Validation images", dataset["validation_images"]),
    ("Test images", dataset["test_images"]),
]:
    add(f"| {key} | {value:,} |")

add()
add("### Class distribution")
add()
add("| Class | Images |")
add("|---|---:|")

for name, count in dataset["class_counts"].items():
    add(f"| {name} | {count:,} |")

add()
add(
    "Exact SHA256 duplicate checking was used during dataset preparation. "
    "This removes byte-identical files but does not establish patient-level "
    "independence because patient identifiers were not verified."
)
add()
add("## 3. Pneumonia Classification")
add()
add("The primary classification model is EfficientNetB0.")
add()
add("### Primary model — held-out test results")
add()
add("| Metric | Result |")
add("|---|---:|")
add(f"| Accuracy | {pct(classifier['test_accuracy'])} |")
add(f"| Macro precision | {pct(classifier['macro_precision'])} |")
add(f"| Macro recall | {pct(classifier['macro_recall'])} |")
add(f"| Macro F1 | {pct(classifier['macro_f1'])} |")
add(f"| Weighted F1 | {pct(classifier['weighted_f1'])} |")
add(f"| Viral decision threshold | {classifier['viral_threshold']} |")
add()
add("### Source-balanced model comparison")
add()
add("| Metric | Primary model | Source-balanced model |")
add("|---|---:|---:|")
add(
    f"| Accuracy | {pct(classifier['test_accuracy'])} "
    f"| {pct(source_balanced_classifier['test_accuracy'])} |"
)
add(
    f"| Macro F1 | {pct(classifier['macro_f1'])} "
    f"| {pct(source_balanced_classifier['macro_f1'])} |"
)
add(
    f"| Weighted F1 | {pct(classifier['weighted_f1'])} "
    f"| {pct(source_balanced_classifier['weighted_f1'])} |"
)
add()
add(
    "The source-balanced model improved performance on the "
    "Bacterial/Viral source but reduced performance on the COVID "
    "Radiography source. It did not outperform the primary model on "
    "overall test metrics."
)
add()
add("## 4. Source-wise Classification Analysis")
add()
add("| Model and source | Accuracy | Macro F1 |")
add("|---|---:|---:|")

for model_name, model_data in source_wise_performance.items():
    label = (
        "Primary"
        if model_name == "original_threshold"
        else "Source-balanced"
    )

    add(
        f"| {label} — COVID source | "
        f"{pct(model_data['covid_source_accuracy'])} | "
        f"{pct(model_data['covid_source_macro_f1'])} |"
    )

    add(
        f"| {label} — Bacterial/Viral source | "
        f"{pct(model_data['bacterial_viral_source_accuracy'])} | "
        f"{pct(model_data['bacterial_viral_source_macro_f1'])} |"
    )

add()
add(
    "The datasets have source/class confounding: Bacterial images are "
    "from the Bacterial/Viral dataset, while COVID images are from the "
    "COVID Radiography dataset. Therefore, source-specific acquisition "
    "characteristics may influence classification. These experiments "
    "raise a concern but do not prove shortcut learning."
)
add()
add("## 5. Lung Segmentation")
add()
add(
    "A lightweight U-Net was trained for lung-region segmentation. "
    "The original balanced model was subsequently fine-tuned using "
    "COVID Radiography images and their corresponding lung masks."
)
add()
add("| Metric | Original U-Net | COVID fine-tuned U-Net |")
add("|---|---:|---:|")
add(
    f"| Held-out test mean Dice | "
    f"{segmentation['original_test_mean_dice']:.4f} | "
    f"{segmentation['fine_tuned_test_mean_dice']:.4f} |"
)
add(
    f"| Held-out test mean IoU | "
    f"{segmentation['original_test_mean_iou']:.4f} | "
    f"{segmentation['fine_tuned_test_mean_iou']:.4f} |"
)
add()
add(f"Fine-tuning best epoch: {segmentation['best_epoch']}.")
add(
    f"Best validation Dice: "
    f"{segmentation['best_validation_dice']:.4f}."
)
add(
    f"Best validation IoU: "
    f"{segmentation['best_validation_iou']:.4f}."
)
add()
add(
    "The fine-tuned model improved segmentation performance on the "
    "held-out COVID Radiography test set. Some residual undersegmentation, "
    "oversegmentation, and fragmented masks were observed."
)
add()
add("## 6. Quantitative Grad-CAM Analysis")
add()
add(
    f"A total of {gradcam_results['images_analyzed']} test images were "
    "analyzed, with 50 images in each of three groups."
)
add()
add("| Group | Activation inside lungs | Activation outside lungs |")
add("|---|---:|---:|")

for group, values in gradcam_results["groups"].items():
    add(
        f"| {group} | "
        f"{values['inside_activation_percentage']:.2f}% | "
        f"{values['outside_activation_percentage']:.2f}% |"
    )

add()
add(
    f"The Grad-CAM target layer was "
    f"`{gradcam_results['target_layer']}`. The lung-mask threshold was "
    f"{gradcam_results['lung_threshold']} and the Grad-CAM threshold "
    f"was {gradcam_results['gradcam_threshold']}."
)
add()
add(
    "Most thresholded Grad-CAM activation pixels were outside the "
    "predicted lung masks across all three groups. This raises concerns "
    "about the spatial distribution of model attention."
)
add()
add("## 7. Statistical Grad-CAM Analysis")
add()
add(
    f"Pairwise comparisons performed: "
    f"{statistical_results['pairwise_comparisons']}."
)
add(
    f"Significant pairwise comparisons after Holm correction: "
    f"{statistical_results['significant_pairwise_comparisons']}."
)
add(
    f"Significant paired inside/outside comparisons after Holm correction: "
    f"{statistical_results['significant_within_group_comparisons']}."
)
add()
add(
    "All three groups showed significantly greater thresholded activation "
    "outside the predicted lung masks than inside them in paired "
    "Wilcoxon signed-rank tests after Holm correction. Selected pairwise "
    "comparisons also showed significant differences in mean activation "
    "strength."
)
add()
add(
    "Statistical significance does not establish causation or prove "
    "shortcut learning. The groups differ in class and dataset-source "
    "composition."
)
add()
add("## 8. Representative Visual Analysis")
add()
add(
    f"{visualization_results['images_visualized']} representative images "
    "were visualized: three from each analysis group. Each figure includes "
    "the original X-ray, predicted lung mask, Grad-CAM heatmap, overlay, "
    "and inside/outside attention regions."
)
add()
add(
    "The visualizations support qualitative inspection of model "
    "attribution relative to predicted lung regions. They do not "
    "represent infection segmentation."
)
add()
add("## 9. Infection Percentage and Severity")
add()
add(
    "The current lung segmentation model identifies lung regions, "
    "not infected regions. Grad-CAM provides model-attribution maps, "
    "not validated lesion masks."
)
add()
add(
    "Therefore, the current experiments do not establish a reliable "
    "infection percentage or clinically validated severity stage. "
    "These outputs must not be presented as clinical measurements."
)
add()
add("## 10. Research Limitations")
add()
add("1. Dataset-source and class-label confounding.")
add("2. Patient-level independence has not been established.")
add("3. Lung masks contain some segmentation errors.")
add("4. Grad-CAM is a coarse attribution method.")
add("5. Grad-CAM activation is not infection-region segmentation.")
add("6. Infection percentage and severity thresholds are not clinically validated.")
add("7. The experiments use public datasets and require external validation.")
add("8. The system is experimental and is not intended for clinical diagnosis.")
add()
add("## 11. Overall Conclusion")
add()
add(
    "The research implementation demonstrates a working experimental "
    "pipeline combining EfficientNetB0 classification, lightweight "
    "U-Net lung segmentation, and Grad-CAM-based explainability. "
    "COVID-domain fine-tuning substantially improved lung segmentation "
    "on the corresponding held-out dataset. Classification achieved "
    "promising but imperfect test performance, with measurable "
    "differences across dataset sources."
)
add()
add(
    "The Grad-CAM experiments identified substantial activation outside "
    "predicted lung regions, motivating further investigation into model "
    "attention and dataset-related effects. Infection quantification "
    "and severity estimation remain exploratory and require validated "
    "infection-region annotations and clinical evaluation before they "
    "can be considered reliable."
)
add()
add("---")
add("**Report generated automatically from the project's experimental results.**")

REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# 6. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("RESEARCH FINDINGS REPORT GENERATED")
print("=" * 70)

print("\nMarkdown report:")
print(REPORT_PATH)

print("\nStructured metrics:")
print(JSON_PATH)

print("\nClassification accuracy:", pct(classifier["test_accuracy"]))
print("Classification macro F1:", pct(classifier["macro_f1"]))

print(
    "Fine-tuned U-Net test Dice:",
    segmentation["fine_tuned_test_mean_dice"]
)

print(
    "Grad-CAM images analyzed:",
    gradcam_results["images_analyzed"]
)

print("\nScript 80 completed successfully.")

# to run this script, use the following command:
# python scripts/80_generate_research_findings_report.py