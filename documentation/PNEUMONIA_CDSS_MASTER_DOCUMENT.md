# Pneumonia Clinical Decision Support System (CDSS)
## Master Project Documentation

**Project:** Pneumonia Clinical Decision Support System using Deep Learning  
**Implementation:** Python, TensorFlow/Keras and Django (planned integration)  
**Primary Models:** Lightweight U-Net and EfficientNetB0  
**Image Input Size:** 224 × 224 pixels  
**Classification Classes:** Normal, Viral, Bacterial, COVID  

---

# 1. Project Overview

The Pneumonia Clinical Decision Support System (CDSS) is a research-oriented application designed to analyze chest X-ray images using deep learning.

The system aims to identify the possible category of a chest X-ray, generate visual explanations of model predictions, and provide supporting information for clinical decision-making.

The project uses two deep learning models:

- **Lightweight U-Net:** Used to identify and segment the lung regions in chest X-ray images.
- **EfficientNetB0:** Used to classify chest X-ray images into Normal, Viral, Bacterial, or COVID categories.

Grad-CAM is used to visualize image regions that influence the classification model's predictions.

The current implementation focuses on dataset preparation, model development, evaluation, segmentation, classification, and single-image inference. Further clinical features and backend integration remain part of the planned development.

# 2. Complete System Workflow

The proposed system follows this pipeline:

1. Chest X-ray image input.
2. Image preprocessing and resizing.
3. Lung segmentation using Lightweight U-Net.
4. Pneumonia classification using EfficientNetB0.
5. Prediction explanation using Grad-CAM.
6. Infection-region quantification using a suitable lesion segmentation method (future work).
7. Severity assessment (future work).
8. Progression analysis using multiple patient images (future work).
9. CDSS result generation.
10. Django backend integration.
11. Optional frontend application.

**Important:** Lung segmentation and infection segmentation are different tasks. The current U-Net identifies lung regions; it does not directly identify pneumonia-infected regions. Therefore, infection percentage and severity calculations are not currently implemented as validated clinical measurements.

# 3. Project Structure

The project is maintained in the following root directory:

`F:\Hemlata Thesis\Code`

Important directories:

| Directory | Purpose |
|---|---|
| `aiml/` | Configuration and common project settings |
| `data/raw/` | Original downloaded datasets |
| `data/processed/classification/` | Prepared classification dataset |
| `data/processed/classification_source_balanced/` | Experimental source-balanced training data |
| `data/processed/segmentation/` | Prepared lung segmentation dataset |
| `scripts/` | Data preparation, training, evaluation and inference scripts |
| `saved_models/` | Trained model checkpoints |
| `results/` | Evaluation reports, visualizations and experiment outputs |
| `documentation/` | Project documentation |

The existing scripts and model files are retained. Their organization can be improved after checking dependencies and file paths.

# 4. Dataset Description

Three main dataset sources were inspected during the research.

## 4.1 Bacterial and Viral Pneumonia Dataset

This dataset contains labelled chest X-ray images with the following categories:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

The training labels were obtained from `labels_train.csv`.

A total of 4,672 labelled training images were available. The separate unlabeled test images were not used for supervised classification.

## 4.2 COVID-19 Radiography Dataset

The COVID-19 Radiography dataset contains the following categories:

- COVID
- Normal
- Viral Pneumonia
- Lung Opacity

The selected categories for classification were COVID, Normal and Viral Pneumonia.

The Lung Opacity category was excluded because it does not directly correspond to one of the four target classification classes.

## 4.3 Lung Segmentation Dataset

The lung segmentation dataset contains chest X-ray images and corresponding lung masks.

Dataset audit findings:

- Original X-ray images: 800
- Images with matching masks: 704
- Images without matching masks: 96

The 704 matched image-mask pairs were used for segmentation preparation.

## 4.4 Dataset Quality and Leakage Checks

Dataset auditing included:

- Image file inventory.
- Image readability checks.
- Label inspection.
- Exact duplicate detection using SHA256 hashes.
- Duplicate and potential split-leakage analysis.
- Image-mask matching.
- Dataset source and class distribution analysis.

The SHA256 process identifies exact byte-level duplicates. It does not guarantee detection of visually similar images or images belonging to the same patient.

Patient-level separation could not be verified because reliable patient identifiers were not established for all sources.

# 5. Classification Dataset Preparation

The classification dataset was prepared using:

`scripts/02_prepare_classification_data.py`

The preparation process included:

1. Reading labelled images from the selected datasets.
2. Mapping source labels into the four target classes.
3. Checking image validity.
4. Removing exact duplicate images.
5. Handling conflicting duplicate groups.
6. Creating training, validation and testing splits.
7. Generating a dataset manifest.

The final prepared dataset contains:

| Split | Image Count |
|---|---:|
| Training | 13,822 |
| Validation | 2,962 |
| Testing | 2,962 |
| **Total** | **19,746** |

Class distribution:

| Class | Images |
|---|---:|
| Normal | 11,414 |
| Viral | 2,540 |
| Bacterial | 2,222 |
| COVID | 3,570 |
| **Total** | **19,746** |

The split was created using a stratified 70:15:15 approach.

The prepared dataset manifest is stored at:

`data/processed/classification/manifest.csv`

# 6. Image Preprocessing

Preprocessing is performed separately for classification and segmentation because the models have different input requirements.

## 6.1 Classification Preprocessing

The classification pipeline uses TensorFlow image operations:

- Read image files.
- Decode images as three-channel RGB.
- Resize images to 224 × 224 pixels.
- Convert image values to float32.
- Retain the 0–255 pixel value range.

EfficientNetB0 includes its own input rescaling operations, so the classification pipeline does not divide image values by 255 before passing them to the model.

## 6.2 Segmentation Preprocessing

The segmentation pipeline uses:

- Grayscale image conversion.
- Resizing to 224 × 224 pixels.
- Image normalization to the 0–1 range.
- Binary lung masks.
- Nearest-neighbour resizing for masks.

The segmentation model is trained using grayscale images and binary masks.

# 7. Lung Segmentation Model

## 7.1 Model Architecture

A Lightweight U-Net architecture is used for lung segmentation.

Its purpose is to identify the lung regions in chest X-ray images and generate a binary lung mask.

The segmentation dataset was divided as follows:

| Split | Image Count |
|---|---:|
| Training | 492 |
| Validation | 105 |
| Testing | 107 |
| **Total** | **704** |

## 7.2 Training and Checkpoint

The balanced U-Net model was trained using:

`scripts/19_train_balanced_segmentation.py`

Primary checkpoint:

`saved_models/unet_lung_segmentation_balanced.keras`

The final test evaluation was performed using:

`scripts/25_evaluate_final_test_segmentation.py`

## 7.3 Segmentation Results

| Metric | Result |
|---|---:|
| Validation Dice | 0.9583 |
| Final Test Dice | 0.9561 |
| Median Test Dice | 0.9659 |
| Test Dice Standard Deviation | 0.0283 |
| Test IoU | 0.9172 |

These results indicate strong overlap between the predicted lung masks and reference masks on the prepared segmentation test dataset.

The checkpoint used by the integrated inference pipeline is the balanced U-Net model.

The inference threshold is 0.30. This is distinct from the 0.50 threshold used when preparing binary training masks.

# 8. Pneumonia Classification Model

## 8.1 EfficientNetB0 Architecture

EfficientNetB0 is used as the image classification model.

The architecture uses ImageNet-pretrained weights.

The original training approach includes:

- A pretrained EfficientNetB0 backbone.
- Frozen backbone layers during initial training.
- Image augmentation.
- Global Average Pooling.
- Dropout with a rate of 0.30.
- A four-unit softmax output layer.

The model predicts probabilities for:

1. Normal
2. Viral
3. Bacterial
4. COVID

## 8.2 Primary Classification Checkpoint

The primary model selected for the integrated inference pipeline is:

`saved_models/efficientnetb0_pneumonia_classifier_best.keras`

The original model was retained as the primary checkpoint after comparing experimental alternatives.

## 8.3 Classification Decision Threshold

The final inference pipeline uses a Viral probability threshold of 0.25.

If the Viral probability is at least 0.25, the system selects Viral. Otherwise, it selects the highest-probability class among the remaining classes.

This is a research decision rule selected using validation data. It is not a clinically validated diagnostic threshold.

# 9. Final Classification Evaluation

The final held-out test evaluation was performed using:

`scripts/55_final_test_classifier_evaluation.py`

The test dataset contains 2,962 images.

The primary classifier was evaluated using the previously selected Viral threshold of 0.25.

## 9.1 Final Test Metrics

| Metric | Result |
|---|---:|
| Accuracy | 82.07% |
| Macro Precision | 76.04% |
| Macro Recall | 82.04% |
| Macro F1-score | 77.41% |
| Weighted F1-score | 82.61% |

## 9.2 Class-wise Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 0.97 | 0.81 | 0.88 |
| Viral | 0.78 | 0.63 | 0.70 |
| Bacterial | 0.58 | 0.92 | 0.71 |
| COVID | 0.71 | 0.92 | 0.80 |

## 9.3 Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

Class order: Normal, Viral, Bacterial, COVID.

| Actual / Predicted | Normal | Viral | Bacterial | COVID |
|---|---:|---:|---:|---:|
| Normal | 1392 | 39 | 82 | 199 |
| Viral | 10 | 240 | 131 | 0 |
| Bacterial | 1 | 26 | 307 | 0 |
| COVID | 33 | 4 | 6 | 492 |

The results show that Viral and Bacterial pneumonia remain difficult to distinguish. This is an important limitation to discuss in the research findings.

The held-out test set must remain untouched for future model tuning and threshold selection.

# 10. Grad-CAM Explainability

Grad-CAM is used to visualize image regions that influence the classification model's prediction.

The implementation uses the EfficientNetB0 layer:

`block5c_project_conv`

The generated heatmap is overlaid on the original chest X-ray image.

Grad-CAM helps visualize the model's attention, but it does not provide a medically verified infection boundary.

In particular:

- Grad-CAM is not a lesion segmentation mask.
- Heatmap intensity is not an infection percentage.
- Attention inside the lung region does not establish that the model has identified pneumonia.

# 11. Integrated Single-Image Inference

The integrated inference implementation is:

`scripts/56_single_image_unet_efficientnet_gradcam.py`

It combines the balanced U-Net, primary EfficientNetB0 classifier and Grad-CAM.

The process is:

1. Load a chest X-ray image.
2. Preprocess the image for the classifier.
3. Convert the image to grayscale for U-Net.
4. Generate a lung probability mask.
5. Apply the lung threshold of 0.30.
6. Run EfficientNetB0 classification.
7. Apply the Viral decision threshold of 0.25.
8. Generate Grad-CAM.
9. Save the prediction and visual outputs.

Each image produces an output folder containing:

| File | Purpose |
|---|---|
| `lung_mask.png` | Predicted lung mask |
| `gradcam_overlay.png` | Grad-CAM visualization |
| `combined_inference_visualization.png` | Combined visual output |
| `inference_result.json` | Prediction probabilities and inference information |

Single-image examples were checked for all four target classes using validation images. These examples demonstrate that the integrated pipeline executes; they are not a substitute for a separate performance evaluation.

# 12. Research Experiments

Several experiments were conducted to investigate model performance and possible dataset-related issues.

These experiments are retained as research findings and should not be confused with the final selected model.

## 12.1 Fine-Tuning Experiment

The fine-tuned EfficientNetB0 checkpoint achieved approximately 82.14% validation accuracy.

However, Viral recall decreased compared with the original model. Therefore, the fine-tuned model was not selected as the primary classifier.

## 12.2 Viral Class-Weight Experiments

Additional models were trained using increased Viral class weights:

- 1.2× Viral weighting.
- 1.4× Viral weighting.

These experiments investigated the effect of class weighting on Viral classification. They did not replace the primary model.

## 12.3 Source-Balanced Training Experiment

A source-balanced training subset was created to investigate possible dataset-source effects.

The experimental model was:

`saved_models/efficientnetb0_source_balanced_best.keras`

Its test performance did not exceed the primary model's performance under the selected decision rules. It was therefore retained as an experimental checkpoint.

## 12.4 Dataset Source Bias Investigation

The classification dataset combines images from different sources.

Some classes are associated with only one source dataset:

- Bacterial images come from the Bacterial/Viral source.
- COVID images come from the COVID-19 Radiography source.

Normal and Viral images are represented across sources.

Source-class associations can influence the model's learning. Source classification and image masking experiments indicated that source-related visual information is learnable.

These findings raise a domain-generalization concern. They do not, by themselves, prove that the primary classifier relies exclusively on dataset-specific shortcuts.

## 12.5 Lung-Only Classification Ablation

The lung-only experiment compared original image predictions with predictions using images restricted by predicted lung masks.

The original predictions were reproduced with very small numerical differences.

The experiment showed a reduction in classification performance when using lung-only images.

However, the result is affected by mask quality, removed image context and distribution changes. It cannot be interpreted as definitive proof of shortcut learning.

## 12.6 Lung Mask Quality Audit

The full validation mask audit covered 2,962 images.

| Heuristic Category | Count | Percentage |
|---|---:|---:|
| Plausible shape | 2,434 | 82.11% |
| Partial or imbalanced | 275 | 9.28% |
| Fragmented | 232 | 7.83% |
| Failed or near-empty | 21 | 0.71% |

These categories are heuristic screening results, not manually verified segmentation ground truth.

The audit identified variability in predicted lung masks, including incomplete, fragmented and near-empty masks.

## 12.7 Preprocessing Domain Audit

The preprocessing audit examined 3,946 images and completed without processing failures.

The audit compared image dimensions, aspect ratios and pixel-intensity summaries across classification and segmentation data.

The segmentation preparation pipeline resizes images directly to 224 × 224 pixels. This can alter the aspect ratio of images whose original dimensions are not square.

The audit identified preprocessing and dataset-domain differences that should be considered when interpreting segmentation performance on classification images.

# 13. Current Primary Model Files

The following are the main checkpoints used by the integrated pipeline:

| Model | Checkpoint |
|---|---|
| Lung segmentation | `unet_lung_segmentation_balanced.keras` |
| Pneumonia classification | `efficientnetb0_pneumonia_classifier_best.keras` |

Other checkpoints are retained for comparison and reproducibility. They should not be deleted merely because they are not the primary models.

# 14. Current Implementation Status

## Completed

- Dataset inventory and inspection.
- Classification dataset preparation.
- Duplicate and label auditing.
- Segmentation dataset preparation.
- Balanced U-Net training.
- U-Net validation and final test evaluation.
- EfficientNetB0 classification training.
- Classification experiments and comparisons.
- Final held-out classification test evaluation.
- Grad-CAM implementation.
- Integrated single-image inference.
- Validation examples for all four classes.
- Mask quality and preprocessing audits.
- Initial project documentation structure.

## Not Yet Implemented or Not Yet Validated

- Reliable infection/lesion segmentation.
- Validated infection percentage calculation.
- Clinical severity staging.
- Longitudinal progression analysis.
- Complete CDSS result generation.
- Django backend integration.
- Complete frontend integration.
- Clinical validation with an appropriate external dataset or expert-reviewed cases.

# 15. Important Research Limitations

The following limitations must be considered when presenting the results:

1. **Dataset source bias:** The classes are not uniformly represented across dataset sources.
2. **Patient-level leakage:** Patient-level independence could not be fully verified for all datasets.
3. **Segmentation generalization:** The U-Net was evaluated on its prepared segmentation dataset; performance on classification-source images may differ.
4. **Classification confusion:** Viral and Bacterial pneumonia are frequently confused.
5. **Grad-CAM limitations:** Grad-CAM visualizations are explanations of model activation, not infection boundaries.
6. **No infection quantification:** Lung masks cannot be used as infection masks.
7. **No clinical validation:** The model is a research prototype, not a replacement for a qualified medical professional.

# 16. Reproducibility Guidelines

To maintain a reproducible project:

- Keep the primary model checkpoints unchanged.
- Preserve the original dataset manifests.
- Do not modify the held-out test set.
- Use the same preprocessing expected by each model.
- Load the balanced U-Net checkpoint explicitly.
- Use the primary EfficientNetB0 checkpoint for the main inference pipeline.
- Keep experimental checkpoints and outputs separate from final results.
- Record any future model changes and evaluation results.
- Do not describe experimental findings as final clinical conclusions.

# 17. Conclusion

The current research implementation establishes a working deep learning pipeline for chest X-ray lung segmentation, pneumonia classification and Grad-CAM visualization.

The balanced U-Net achieved a final test Dice score of 0.9561 on the prepared segmentation test dataset. The selected EfficientNetB0 classifier achieved 82.07% accuracy and a macro F1-score of 77.41% on the held-out classification test dataset using the selected Viral threshold.

The project also investigated class imbalance, dataset-source effects, lung-only classification and segmentation mask quality.

Further work is required to implement reliable infection segmentation, severity estimation, progression analysis and complete CDSS integration. These components should be developed and validated separately before being presented as completed clinical functionality.

---

**Document status:** Master research documentation — initial version  
**Project root:** `F:\Hemlata Thesis\Code`  
**Primary models:** Balanced U-Net and original best EfficientNetB0  
