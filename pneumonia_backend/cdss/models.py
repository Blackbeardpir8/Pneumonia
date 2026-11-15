
from django.db import models


class Prediction(models.Model):

    CLASS_CHOICES = [
        ("Normal", "Normal"),
        ("Viral", "Viral"),
        ("Bacterial", "Bacterial"),
        ("COVID", "COVID"),
    ]

    # Original uploaded image
    image_name = models.CharField(
        max_length=255
    )

    # Classification results
    predicted_class = models.CharField(
        max_length=20,
        choices=CLASS_CHOICES,
    )

    confidence = models.FloatField()

    normal_probability = models.FloatField()
    viral_probability = models.FloatField()
    bacterial_probability = models.FloatField()
    covid_probability = models.FloatField()

    decision_rule = models.CharField(
        max_length=50,
    )

    # Lung segmentation
    lung_threshold = models.FloatField(default=0.30)
    lung_mask_area_pixels = models.PositiveIntegerField()
    lung_mask_area_percent = models.FloatField()

    # Infection segmentation
    infection_threshold = models.FloatField(default=0.50)
    infection_mask_area_pixels = models.PositiveIntegerField()
    infection_mask_area_percent = models.FloatField()

    # Grad-CAM measurements
    gradcam_inside_lung_percent = models.FloatField(
        null=True,
        blank=True,
    )

    gradcam_outside_lung_percent = models.FloatField(
        null=True,
        blank=True,
    )

    # Generated visualization paths
    input_image_path = models.CharField(
        max_length=500,
        blank=True,
    )

    lung_mask_path = models.CharField(
        max_length=500,
        blank=True,
    )

    infection_mask_path = models.CharField(
        max_length=500,
        blank=True,
    )

    lung_overlay_path = models.CharField(
        max_length=500,
        blank=True,
    )

    infection_overlay_path = models.CharField(
        max_length=500,
        blank=True,
    )

    gradcam_overlay_path = models.CharField(
        max_length=500,
        blank=True,
    )

    gradcam_heatmap_path = models.CharField(
        max_length=500,
        blank=True,
    )

    combined_visualization_path = models.CharField(
        max_length=500,
        blank=True,
    )

    # Research limitations
    infection_segmentation_status = models.CharField(
        max_length=50,
        default="experimental",
    )

    limitations = models.JSONField(
        default=list,
        blank=True,
    )

    # Timestamp
    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "CDSS Prediction"
        verbose_name_plural = "CDSS Predictions"

    def __str__(self):
        return (
            f"{self.image_name} - "
            f"{self.predicted_class} "
            f"({self.confidence:.2f}%)"
        )
