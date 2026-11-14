"""
Central configuration for the Pneumonia Clinical Decision Support System.

This file contains shared settings and project paths used by the
preprocessing, segmentation, classification, evaluation, and prediction
modules.

Keeping settings in one place makes the project easier to maintain.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

# Project root is the parent directory of the aiml folder.
# This avoids hardcoding an absolute Windows path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CLASSIFICATION_DATA_DIR = PROCESSED_DATA_DIR / "classification"
SEGMENTATION_DATA_DIR = PROCESSED_DATA_DIR / "segmentation"

AIML_DIR = PROJECT_ROOT / "aiml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 2. IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


# ============================================================
# 3. CLASSIFICATION SETTINGS
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)

BACTERIAL_VIRAL_CLASS_MAPPING = {
    "0": "Normal",
    "1": "Bacterial",
    "2": "Viral",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# 4. MODEL TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001


# ============================================================
# 5. SEGMENTATION AND EXPLAINABILITY SETTINGS
# ============================================================

SEGMENTATION_THRESHOLD = 0.5
GRADCAM_THRESHOLD = 0.5


# ============================================================
# 6. INFECTION SEVERITY THRESHOLDS
# ============================================================

MILD_MAX = 25.0
MODERATE_MAX = 50.0
SEVERE_MAX = 75.0


# ============================================================
# 7. SAVED MODEL PATHS
# ============================================================

UNET_MODEL_PATH = (
    SAVED_MODELS_DIR / "unet_lung_segmentation.keras"
)

EFFICIENTNET_MODEL_PATH = (
    SAVED_MODELS_DIR / "efficientnetb0_classifier.keras"
)

CLASS_INDICES_PATH = SAVED_MODELS_DIR / "class_indices.json"


# ============================================================
# 8. RESULTS DIRECTORIES
# ============================================================

CLASSIFICATION_RESULTS_DIR = RESULTS_DIR / "classification"
SEGMENTATION_RESULTS_DIR = RESULTS_DIR / "segmentation"
GRADCAM_RESULTS_DIR = RESULTS_DIR / "gradcam"
SEVERITY_RESULTS_DIR = RESULTS_DIR / "severity"
PROGRESSION_RESULTS_DIR = RESULTS_DIR / "progression"
MODEL_COMPARISON_RESULTS_DIR = RESULTS_DIR / "model_comparison"
DEMO_SCREENSHOTS_DIR = RESULTS_DIR / "demo_screenshots"

METRICS_COMPARISON_PATH = RESULTS_DIR / "metrics_comparison.csv"
