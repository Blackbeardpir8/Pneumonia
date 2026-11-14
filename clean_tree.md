```bash
Code/
|-- aiml
|   |-- __init__.py
|   `-- config.py
|-- clean_tree.md
|-- data
|   |-- processed
|   |   |-- classification
|   |   |   |-- manifest.csv
|   |   |   |-- test
|   |   |   |   |-- Bacterial
|   |   |   |   |-- COVID
|   |   |   |   |-- Normal
|   |   |   |   `-- Viral
|   |   |   |-- train
|   |   |   |   |-- Bacterial
|   |   |   |   |-- COVID
|   |   |   |   |-- Normal
|   |   |   |   `-- Viral
|   |   |   `-- val
|   |   |       |-- Bacterial
|   |   |       |-- COVID
|   |   |       |-- Normal
|   |   |       `-- Viral
|   |   |-- classification_source_balanced
|   |   |   |-- class_source_counts.csv
|   |   |   |-- class_weights.json
|   |   |   |-- manifest.csv
|   |   |   `-- train
|   |   |       |-- Bacterial
|   |   |       |-- COVID
|   |   |       |-- Normal
|   |   |       `-- Viral
|   |   `-- segmentation
|   |       |-- manifest.csv
|   |       |-- test
|   |       |   |-- images
|   |       |   `-- masks
|   |       |-- train
|   |       |   |-- images
|   |       |   `-- masks
|   |       `-- val
|   |           |-- images
|   |           `-- masks
|   `-- raw
|       |-- archive
|       |   `-- COVID-19_Radiography_Dataset
|       |       |-- COVID
|       |       |   |-- images
|       |       |   `-- masks
|       |       |-- COVID.metadata.xlsx
|       |       |-- Lung_Opacity
|       |       |   |-- images
|       |       |   `-- masks
|       |       |-- Lung_Opacity.metadata.xlsx
|       |       |-- Normal
|       |       |   |-- images
|       |       |   `-- masks
|       |       |-- Normal.metadata.xlsx
|       |       |-- Viral Pneumonia
|       |       |   |-- images
|       |       |   `-- masks
|       |       `-- Viral Pneumonia.metadata.xlsx
|       |-- archive (1)
|       |   `-- chest_xray
|       |       |-- __MACOSX
|       |       |   `-- chest_xray
|       |       |       |-- test
|       |       |       |   |-- NORMAL
|       |       |       |   `-- PNEUMONIA
|       |       |       |-- train
|       |       |       |   |-- NORMAL
|       |       |       |   `-- PNEUMONIA
|       |       |       `-- val
|       |       |           |-- NORMAL
|       |       |           `-- PNEUMONIA
|       |       |-- chest_xray
|       |       |   |-- test
|       |       |   |   |-- NORMAL
|       |       |   |   `-- PNEUMONIA
|       |       |   |-- train
|       |       |   |   |-- NORMAL
|       |       |   |   `-- PNEUMONIA
|       |       |   `-- val
|       |       |       |-- NORMAL
|       |       |       `-- PNEUMONIA
|       |       |-- test
|       |       |   |-- NORMAL
|       |       |   `-- PNEUMONIA
|       |       |-- train
|       |       |   |-- NORMAL
|       |       |   `-- PNEUMONIA
|       |       `-- val
|       |           |-- NORMAL
|       |           `-- PNEUMONIA
|       |-- archive (2)
|       |   |-- Lung Segmentation
|       |   |   |-- CXR_png
|       |   |   |-- ClinicalReadings
|       |   |   |-- NLM-ChinaCXRSet-ReadMe.docx
|       |   |   |-- NLM-MontgomeryCXRSet-ReadMe.pdf
|       |   |   |-- masks
|       |   |   `-- test
|       |   `-- data
|       |       `-- Lung Segmentation
|       |           |-- CXR_png
|       |           |-- ClinicalReadings
|       |           |-- NLM-ChinaCXRSet-ReadMe.docx
|       |           |-- NLM-MontgomeryCXRSet-ReadMe.pdf
|       |           |-- masks
|       |           `-- test
|       `-- archive (3)
|           |-- labels_train.csv
|           |-- test_images
|           |   `-- test_images
|           `-- train_images
|               `-- train_images
|-- generate_clean_tree.py
|-- results
|   |-- classification
|   |   |-- clahe_segmentation_experiment
|   |   |   |-- clahe_group_summary.csv
|   |   |   |-- clahe_per_image_results.csv
|   |   |   `-- failed_image_original_vs_clahe.csv
|   |   |-- class_distribution.csv
|   |   |-- connected_component_comparison
|   |   |   `-- component_comparison_summary.csv
|   |   |-- dataset_audit
|   |   |   |-- conflicting_classification_labels.json
|   |   |   |-- duplicate_classification_images.csv
|   |   |   `-- invalid_classification_images.csv
|   |   |-- error_analysis
|   |   |   |-- all_misclassified_images.csv
|   |   |   |-- confusion_pairs.csv
|   |   |   |-- errors_by_source.csv
|   |   |   |-- viral_predicted_as_bacterial.csv
|   |   |   |-- viral_source_error_rates.csv
|   |   |   `-- visualizations
|   |   |       `-- selected_examples.csv
|   |   |-- failed_image_intensity_analysis
|   |   |   |-- group_intensity_summary.csv
|   |   |   |-- image_intensity_statistics.csv
|   |   |   `-- selected_images.csv
|   |   |-- failed_successful_unet_comparison
|   |   |   |-- comparison_group_summary.csv
|   |   |   |-- failed_image_details.csv
|   |   |   `-- failed_successful_per_image.csv
|   |   |-- final_test_evaluation
|   |   |   |-- candidate_comparison_summary.csv
|   |   |   |-- final_test_evaluation_summary.json
|   |   |   |-- original_threshold_0_250_argmax_confusion_matrix.csv
|   |   |   |-- original_threshold_0_250_argmax_metrics.json
|   |   |   |-- original_threshold_0_250_predictions.csv
|   |   |   |-- original_threshold_0_250_viral_threshold_confusion_matrix.csv
|   |   |   |-- original_threshold_0_250_viral_threshold_metrics.json
|   |   |   |-- source_balanced_threshold_0_275_argmax_confusion_matrix.csv
|   |   |   |-- source_balanced_threshold_0_275_argmax_metrics.json
|   |   |   |-- source_balanced_threshold_0_275_predictions.csv
|   |   |   |-- source_balanced_threshold_0_275_viral_threshold_confusion_matrix.csv
|   |   |   `-- source_balanced_threshold_0_275_viral_threshold_metrics.json
|   |   |-- fine_tuning
|   |   |   |-- finetuning_metadata.json
|   |   |   `-- history
|   |   |       |-- finetuning_history.csv
|   |   |       `-- finetuning_history.json
|   |   |-- gradcam_preview
|   |   |   `-- gradcam_preview_summary.csv
|   |   |-- inspection
|   |   |   |-- class_distribution.csv
|   |   |   |-- image_dimensions.csv
|   |   |   |-- invalid_images.csv
|   |   |   |-- label_errors.csv
|   |   |   |-- missing_images.csv
|   |   |   `-- split_leakage.csv
|   |   |-- lung_ablation_preview
|   |   |-- lung_background_classifier_ablation
|   |   |   |-- ablation_predictions.csv
|   |   |   |-- ablation_summary.csv
|   |   |   |-- baseline_verification.csv
|   |   |   |-- classification_reports.json
|   |   |   |-- confusion_matrices.json
|   |   |   `-- source_wise_accuracy.csv
|   |   |-- lung_only_classifier_ablation
|   |   |   |-- baseline_verification.json
|   |   |   |-- full_validation_mask_quality_audit
|   |   |   |   |-- audit_summary.json
|   |   |   |   |-- processing_failures.csv
|   |   |   |   |-- validation_mask_quality_by_class.csv
|   |   |   |   |-- validation_mask_quality_by_class_and_source.csv
|   |   |   |   |-- validation_mask_quality_by_source.csv
|   |   |   |   `-- validation_mask_quality_per_image.csv
|   |   |   |-- low_mask_audit_samples.csv
|   |   |   |-- low_mask_visual_audit
|   |   |   |   |-- Bacterial
|   |   |   |   |-- COVID
|   |   |   |   |-- Normal
|   |   |   |   |-- Viral
|   |   |   |   `-- mask_audit_measurements.csv
|   |   |   |-- lung_only_ablation_summary.json
|   |   |   |-- lung_only_detailed_metrics.json
|   |   |   |-- lung_only_metrics.csv
|   |   |   |-- lung_only_predictions.csv
|   |   |   |-- preprocessing_domain_audit
|   |   |   |   |-- audit_summary.json
|   |   |   |   |-- classification_summary_by_class_and_source.csv
|   |   |   |   |-- image_geometry_intensity_per_image.csv
|   |   |   |   |-- image_read_failures.csv
|   |   |   |   `-- summary_by_image_group.csv
|   |   |   `-- targeted_mask_visual_audit
|   |   |       |-- most_fragmented
|   |   |       |-- most_imbalanced
|   |   |       |-- near_empty
|   |   |       `-- selected_image_audit.csv
|   |   |-- lung_threshold_comparison
|   |   |   `-- threshold_mask_area_summary.csv
|   |   |-- model
|   |   |   `-- model_configuration.json
|   |   |-- model_comparison
|   |   |   |-- class_wise_comparison.csv
|   |   |   |-- finetuned
|   |   |   |   |-- classification_report.csv
|   |   |   |   |-- confusion_matrix.csv
|   |   |   |   `-- validation_predictions.csv
|   |   |   |-- original
|   |   |   |   |-- classification_report.csv
|   |   |   |   |-- confusion_matrix.csv
|   |   |   |   `-- validation_predictions.csv
|   |   |   |-- overall_model_comparison.csv
|   |   |   `-- viral_to_bacterial_comparison.csv
|   |   |-- pipeline
|   |   |   |-- class_weights.json
|   |   |   `-- pipeline_summary.json
|   |   |-- pneumonia_classifier_ablation
|   |   |   |-- ablation_predictions.csv
|   |   |   |-- ablation_summary.csv
|   |   |   |-- baseline_verification.json
|   |   |   |-- classification_reports.json
|   |   |   |-- confusion_matrices.json
|   |   |   `-- source_wise_accuracy.csv
|   |   |-- single_image_inference
|   |   |   |-- 0011f2ca9802c97b1f1b83f5aa8458a3a3a39239809aefdbd251f3a026058e13_img_734533885643596249
|   |   |   |   `-- inference_result.json
|   |   |   |-- 0014f9bc0e7e24fb8215d9c42125fb5644f6b063a153acc9f193a1f1c36e6b9c_Normal-3931
|   |   |   |   `-- inference_result.json
|   |   |   |-- 0017de0c673890f89620777f6a0d7dac109dce68f26bead45678269f68c00f8d_Normal-1748
|   |   |   |   `-- inference_result.json
|   |   |   |-- 003b3eb1a66a243a284a6bcfc549f84ead1f422880480618328b406479276e60_COVID-1738
|   |   |   |   `-- inference_result.json
|   |   |   |-- 005a0b4068bec81136091a3b37cb6475399daaa86dd740e339c1fd22242cd9d0_COVID-300
|   |   |   |   `-- inference_result.json
|   |   |   |-- 00f1c591d29afd7b596a6e52299e8122e84e9fd6fee4d41148d1b2a5eb110fb0_Viral Pneumonia-206
|   |   |   |   `-- inference_result.json
|   |   |   |-- 0156be93b8157bf2f165ad84d5bf9cefef5512ab692c1db67975d1bb0bd68713_img_7396259257736670072
|   |   |   |   `-- inference_result.json
|   |   |   `-- 01e15f5723767c0aff217288d2e5ef4a5205ca75cf31267047edb6f65146cb4c_Viral Pneumonia-127
|   |   |       `-- inference_result.json
|   |   |-- source_balanced_model_comparison
|   |   |   |-- class_wise_comparison.csv
|   |   |   |-- confusion_matrices.csv
|   |   |   |-- overall_model_comparison.csv
|   |   |   `-- validation_predictions.csv
|   |   |-- source_balanced_threshold_analysis
|   |   |   |-- selected_threshold_comparison.csv
|   |   |   |-- source_wise_selected_threshold_metrics.csv
|   |   |   `-- viral_threshold_sweep.csv
|   |   |-- source_balanced_threshold_multiclass_comparison
|   |   |   |-- class_wise_multiclass_comparison.csv
|   |   |   |-- multiclass_confusion_matrices.csv
|   |   |   |-- overall_multiclass_comparison.csv
|   |   |   |-- selected_rule_predictions.csv
|   |   |   `-- source_wise_multiclass_metrics.csv
|   |   |-- source_balanced_threshold_sourcewise_multiclass
|   |   |   |-- source_wise_class_metrics.csv
|   |   |   |-- source_wise_confusion_matrices.csv
|   |   |   `-- source_wise_overall_metrics.csv
|   |   |-- source_balanced_training
|   |   |   |-- training_history.csv
|   |   |   `-- training_summary.json
|   |   |-- source_bias_analysis
|   |   |   |-- analysis_summary.json
|   |   |   |-- average_probabilities_by_source.csv
|   |   |   |-- class_distribution_by_source.csv
|   |   |   |-- class_performance_by_source.csv
|   |   |   |-- confusion_matrices_by_source.csv
|   |   |   |-- source_accuracy.csv
|   |   |   `-- top_confusions_by_source.csv
|   |   |-- source_bias_classifier
|   |   |   |-- ablation
|   |   |   |   |-- ablation_comparison.csv
|   |   |   |   |-- ablation_summary.json
|   |   |   |   |-- border_masked_classification_report.csv
|   |   |   |   |-- border_masked_confusion_matrix.csv
|   |   |   |   |-- border_masked_predictions.csv
|   |   |   |   |-- central_masked_classification_report.csv
|   |   |   |   |-- central_masked_confusion_matrix.csv
|   |   |   |   |-- central_masked_predictions.csv
|   |   |   |   |-- original_classification_report.csv
|   |   |   |   |-- original_confusion_matrix.csv
|   |   |   |   `-- original_predictions.csv
|   |   |   |-- ablation_corrected
|   |   |   |   |-- ablation_comparison.csv
|   |   |   |   |-- ablation_summary.json
|   |   |   |   |-- border_masked_classification_report.csv
|   |   |   |   |-- border_masked_confusion_matrix.csv
|   |   |   |   |-- border_masked_predictions.csv
|   |   |   |   |-- central_masked_classification_report.csv
|   |   |   |   |-- central_masked_confusion_matrix.csv
|   |   |   |   |-- central_masked_predictions.csv
|   |   |   |   `-- original_predictions.csv
|   |   |   |-- gradcam
|   |   |   |   `-- gradcam_selected_examples.csv
|   |   |   |-- source_classification_report.csv
|   |   |   |-- source_classifier_summary.json
|   |   |   |-- source_confusion_matrix.csv
|   |   |   |-- source_predictions.csv
|   |   |   `-- training_history.csv
|   |   |-- source_wise_class_confusion
|   |   |   |-- source_classification_metrics.csv
|   |   |   |-- source_confusion_matrices.csv
|   |   |   `-- source_overall_accuracy.csv
|   |   |-- source_wise_confusion_analysis
|   |   |   |-- bacterial_viral_dataset_background_only_confusion_matrix.csv
|   |   |   |-- bacterial_viral_dataset_lung_only_confusion_matrix.csv
|   |   |   |-- bacterial_viral_dataset_original_confusion_matrix.csv
|   |   |   |-- covid19_radiography_background_only_confusion_matrix.csv
|   |   |   |-- covid19_radiography_lung_only_confusion_matrix.csv
|   |   |   |-- covid19_radiography_original_confusion_matrix.csv
|   |   |   |-- source_wise_classification_reports.json
|   |   |   |-- source_wise_confusion_matrices.json
|   |   |   `-- source_wise_metrics_summary.csv
|   |   |-- split_summary.json
|   |   |-- training
|   |   |   |-- training_history.csv
|   |   |   `-- training_history.json
|   |   |-- training_data_distribution_audit
|   |   |   |-- training_class_source_counts.csv
|   |   |   |-- training_image_level_statistics.csv
|   |   |   |-- training_image_read_failures.csv
|   |   |   |-- training_resolution_frequencies.csv
|   |   |   `-- training_source_class_image_statistics.csv
|   |   |-- unet_input_distribution_analysis
|   |   |   |-- unet_input_class_summary.csv
|   |   |   |-- unet_input_dataset_summary.csv
|   |   |   |-- unet_input_dimension_summary.csv
|   |   |   `-- unet_input_per_image.csv
|   |   |-- unet_resize_method_comparison
|   |   |   |-- area_vs_bilinear.csv
|   |   |   |-- bicubic_vs_bilinear.csv
|   |   |   |-- resize_method_per_image.csv
|   |   |   `-- resize_method_summary.csv
|   |   |-- validation_evaluation
|   |   |   |-- classification_report.csv
|   |   |   |-- confusion_matrix.csv
|   |   |   |-- evaluation_summary.json
|   |   |   `-- validation_predictions.csv
|   |   |-- validation_gradcam_analysis
|   |   |   |-- analysis_quality_summary.csv
|   |   |   |-- invalid_mask_inspection
|   |   |   |   `-- invalid_mask_inspection_summary.csv
|   |   |   |-- summary_by_class.csv
|   |   |   |-- summary_by_class_and_source.csv
|   |   |   `-- validation_gradcam_per_image.csv
|   |   |-- viral_source_threshold_analysis
|   |   |   |-- overall_threshold_metrics.csv
|   |   |   `-- source_wise_threshold_metrics.csv
|   |   |-- viral_threshold_analysis
|   |   |   |-- best_threshold_by_viral_f1.csv
|   |   |   |-- best_threshold_recall_precision_at_least_080.csv
|   |   |   |-- source_wise_viral_recall.csv
|   |   |   |-- standard_argmax_baselines.csv
|   |   |   `-- threshold_sweep_all_models.csv
|   |   |-- viral_weight_experiments
|   |   |   |-- all_experiments_summary.json
|   |   |   |-- experiment_1_2x
|   |   |   |   |-- experiment_class_weights.json
|   |   |   |   |-- experiment_summary.json
|   |   |   |   |-- training_history.csv
|   |   |   |   `-- training_history.json
|   |   |   `-- experiment_1_4x
|   |   |       |-- experiment_class_weights.json
|   |   |       |-- experiment_summary.json
|   |   |       |-- training_history.csv
|   |   |       `-- training_history.json
|   |   `-- viral_weight_model_comparison
|   |       |-- class_wise_comparison.csv
|   |       |-- original
|   |       |   |-- classification_report.csv
|   |       |   |-- confusion_matrix.csv
|   |       |   `-- validation_predictions.csv
|   |       |-- overall_model_comparison.csv
|   |       |-- viral_1_2x
|   |       |   |-- classification_report.csv
|   |       |   |-- confusion_matrix.csv
|   |       |   `-- validation_predictions.csv
|   |       |-- viral_1_4x
|   |       |   |-- classification_report.csv
|   |       |   |-- confusion_matrix.csv
|   |       |   `-- validation_predictions.csv
|   |       |-- viral_misclassification_comparison.csv
|   |       `-- viral_source_wise_comparison.csv
|   |-- dataset_audit
|   |   |-- classification_split_counts.csv
|   |   |-- directory_inventory.csv
|   |   |-- exact_duplicate_images.csv
|   |   |-- invalid_images.csv
|   |   |-- mask_samples
|   |   |-- potential_split_leakage.csv
|   |   |-- segmentation_external_duplicates.csv
|   |   |-- segmentation_internal_duplicates.csv
|   |   `-- segmentation_mask_audit.csv
|   `-- segmentation
|       |-- balanced_training_history.csv
|       |-- balanced_validation_metrics.csv
|       |-- clahe_validation_evaluation
|       |   |-- paired_comparison.csv
|       |   |-- validation_per_image_results.csv
|       |   `-- validation_summary.csv
|       |-- clinical_readings_audit.csv
|       |-- error_analysis
|       |-- final_test_metrics.csv
|       |-- lung_region_intensity_analysis
|       |   |-- lung_background_region_summary.csv
|       |   `-- lung_region_per_image.csv
|       |-- original_vs_balanced_comparison.csv
|       |-- prediction_samples
|       |-- segmentation_results_summary.csv
|       |-- spatial_intensity_analysis
|       |   |-- spatial_intensity_group_summary.csv
|       |   `-- spatial_intensity_per_image.csv
|       |-- training_history.csv
|       |-- training_history.json
|       |-- training_vs_failed_image_analysis
|       |   |-- group_distribution_summary.csv
|       |   `-- image_distribution_statistics.csv
|       |-- validation_metrics.csv
|       `-- visual_comparison
|-- saved_models
|   |-- dataset_source_classifier.keras
|   |-- efficientnetb0_pneumonia_classifier.keras
|   |-- efficientnetb0_pneumonia_classifier_best.keras
|   |-- efficientnetb0_pneumonia_classifier_finetuned_best.keras
|   |-- efficientnetb0_source_balanced_best.keras
|   |-- efficientnetb0_source_balanced_last.keras
|   |-- efficientnetb0_viral_weight_1.2x_best.keras
|   |-- efficientnetb0_viral_weight_1.4x_best.keras
|   |-- unet_lung_segmentation.keras
|   |-- unet_lung_segmentation_balanced.keras
|   |-- unet_lung_segmentation_balanced_last.keras
|   `-- unet_lung_segmentation_last.keras
|-- scripts
|   |-- 01_inspect_and_prepare_data.py
|   |-- 01b_visualize_segmentation_masks.py
|   |-- 01c_audit_segmentation_duplicates.py
|   |-- 02_prepare_classification_data.py
|   |-- 02_prepare_segmentation_data.py
|   |-- 03_inspect_classification_data.py
|   |-- 03_train_segmentation_model.py
|   |-- 04_build_classification_pipeline.py
|   |-- 04_visualize_segmentation_predictions.py
|   |-- 05_build_efficientnet_model.py
|   |-- 05_check_segmentation_data.py
|   |-- 06_audit_original_segmentation_pairs.py
|   |-- 06_train_efficientnet.py
|   |-- 07_compare_segmentation_thresholds.py
|   |-- 07_evaluate_validation.py
|   |-- 08_analyze_classification_errors.py
|   |-- 08_evaluate_validation_segmentation.py
|   |-- 09_inspect_worst_segmentation.py
|   |-- 09_visualize_classification_errors.py
|   |-- 10_analyze_source_bias.py
|   |-- 10_check_problem_image.py
|   |-- 11_test_dataset_source_bias.py
|   |-- 11_visualize_preprocessing.py
|   |-- 12_compare_dataset_image_statistics.py
|   |-- 12_visualize_source_classifier_gradcam.py
|   |-- 13_compare_dataset_predictions.py
|   |-- 13_test_source_classifier_ablation.py
|   |-- 14_check_dataset_distribution.py
|   |-- 14_verify_source_classifier_checkpoint.py
|   |-- 15_evaluate_training_by_source.py
|   |-- 15_test_pneumonia_classifier_ablation.py
|   |-- 16_compare_mask_statistics.py
|   |-- 16_preview_lung_ablation_masks.py
|   |-- 17_compare_lung_mask_thresholds.py
|   |-- 17_compare_mask_geometry.py
|   |-- 18_check_mask_components.py
|   |-- 18_compare_connected_components.py
|   |-- 19_run_lung_background_classifier_ablation.py
|   |-- 19_train_balanced_segmentation.py
|   |-- 20_evaluate_balanced_segmentation.py
|   |-- 20_source_wise_confusion_analysis.py
|   |-- 21_compare_original_and_balanced.py
|   |-- 22_statistical_comparison.py
|   |-- 23_visual_compare_segmentation.py
|   |-- 24_plot_balanced_training_history.py
|   |-- 25_evaluate_final_test_segmentation.py
|   |-- 26_build_segmentation_results_summary.py
|   |-- 27_generate_gradcam_preview.py
|   |-- 28_analyze_validation_gradcam.py
|   |-- 29_inspect_invalid_attention_masks.py
|   |-- 30_compare_unet_input_distributions.py
|   |-- 31_compare_failed_and_successful_unet_predictions.py
|   |-- 32_compare_unet_resize_methods.py
|   |-- 33_inspect_failed_image_intensity.py
|   |-- 34_test_clahe_segmentation.py
|   |-- 35_evaluate_clahe_on_segmentation_validation.py
|   |-- 36_compare_unet_training_and_failed_images.py
|   |-- 37_analyze_spatial_intensity.py
|   |-- 38_analyze_lung_region_intensity.py
|   |-- 39_inspect_best_classifier.py
|   |-- 40_finetune_efficientnet.py
|   |-- 41_compare_classifier_models.py
|   |-- 42_viral_weight_experiment.py
|   |-- 43_compare_viral_weight_models.py
|   |-- 44_viral_threshold_analysis.py
|   |-- 45_source_wise_viral_threshold_analysis.py
|   |-- 46_source_wise_class_confusion_analysis.py
|   |-- 47_training_data_distribution_audit.py
|   |-- 48_prepare_source_balanced_training_subset.py
|   |-- 49_calculate_balanced_class_weights.py
|   |-- 50_train_source_balanced_efficientnet.py
|   |-- 51_compare_source_balanced_model.py
|   |-- 52_source_balanced_viral_threshold_analysis.py
|   |-- 53_compare_thresholds_multiclass.py
|   |-- 54_sourcewise_threshold_multiclass_analysis.py
|   |-- 55_final_test_classifier_evaluation.py
|   |-- 56_single_image_unet_efficientnet_gradcam.py
|   |-- 57_lung_only_classifier_ablation.py
|   |-- 58_visualize_low_mask_segmentation_audit.py
|   |-- 59_full_validation_mask_quality_audit.py
|   |-- 60_targeted_mask_visual_audit.py
|   `-- 61_preprocessing_domain_audit.py
`-- venv311
    |-- pyvenv.cfg
    `-- share
        `-- man
            `-- man1
                `-- ttx.1
```
