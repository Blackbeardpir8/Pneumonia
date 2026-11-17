from pathlib import Path
from urllib.parse import quote

from django.conf import settings
from rest_framework import serializers

from .models import Prediction


class ChestXrayUploadSerializer(serializers.Serializer):

    image = serializers.ImageField(
        required=True,
        allow_empty_file=False,
    )

    def validate_image(self, image):

        allowed_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        ]

        extension = ""
        if image.name:
            extension = Path(image.name).suffix.lower()

        if extension not in allowed_extensions:
            raise serializers.ValidationError(
                "Unsupported image format. "
                "Upload PNG, JPG, JPEG, BMP, TIF or TIFF."
            )

        return image


class PredictionHistorySerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = "__all__"
        read_only_fields = tuple(
            field.name for field in Prediction._meta.fields
        )

    def build_assessment(self, instance):
        """
        Build a patient-friendly assessment from the saved prediction.
        """
        predicted_class = instance.predicted_class
        confidence = instance.confidence

        probabilities = {
            "Normal": instance.normal_probability,
            "Viral": instance.viral_probability,
            "Bacterial": instance.bacterial_probability,
            "COVID": instance.covid_probability,
        }

        if predicted_class == "Normal":
            assessment_type = "normal"
            pneumonia_prediction = False
            summary = (
                "The AI model predicts the Normal category. "
                "No pneumonia-related category was predicted."
            )
            predicted_type = None
        else:
            assessment_type = "pneumonia_pattern_predicted"
            pneumonia_prediction = True
            summary = (
                f"The AI model predicts a pneumonia-related pattern "
                f"in the {predicted_class} category."
            )
            predicted_type = predicted_class

        if predicted_class == "COVID":
            infection_extent = {
                "available": True,
                "value_percent": instance.infection_mask_area_percent,
            }
        else:
            infection_extent = {
                "available": False,
                "value_percent": None,
            }

        return {
            "assessment_type": assessment_type,
            "pneumonia_prediction": pneumonia_prediction,
            "predicted_pneumonia_type": predicted_type,
            "summary": summary,
            "confidence_percent": instance.confidence,
            "class_probabilities_percent": probabilities,
            "infection_extent": infection_extent,
            "disclaimer": (
                "This is an AI-generated research prediction, "
                "not a confirmed medical diagnosis."
            ),
        }

    def build_prediction_transparency(self, instance):
        probabilities = {
            "Normal": float(instance.normal_probability),
            "Viral": float(instance.viral_probability),
            "Bacterial": float(instance.bacterial_probability),
            "COVID": float(instance.covid_probability),
        }

        highest_probability_class = max(
            probabilities,
            key=probabilities.get
        )

        highest_probability_percent = probabilities[
            highest_probability_class
        ]

        return {
            "highest_probability_class": highest_probability_class,
            "highest_probability_percent": highest_probability_percent,
            "decision_rule_overrode_argmax": (
                instance.predicted_class != highest_probability_class
            ),
        }

    def to_representation(self, instance):
        data = super().to_representation(instance)

        # Convert stored media paths into absolute URLs
        request = self.context.get("request")
        image_fields = [
            "input_image_path",
            "lung_mask_path",
            "infection_mask_path",
            "lung_overlay_path",
            "infection_overlay_path",
            "gradcam_overlay_path",
            "gradcam_heatmap_path",
            "combined_visualization_path",
        ]

        for field in image_fields:
            file_path = data.get(field)
            if file_path and request:
                media_path = quote(file_path, safe="/")
                data[field] = request.build_absolute_uri(
                    f"{settings.MEDIA_URL}{media_path}"
                )

        # Add prediction transparency
        data.update(
            self.build_prediction_transparency(instance)
        )

        # Add patient-friendly assessment
        data["assessment"] = self.build_assessment(instance)

        return data