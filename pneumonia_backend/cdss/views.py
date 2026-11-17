import logging
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import ChestXrayUploadSerializer
from .models import Prediction
from .services.inference_service import run_inference, OUTPUT_ROOT


logger = logging.getLogger(__name__)


def convert_paths_to_urls(data, request):
    """Convert generated media file paths into browser-accessible URLs."""

    if isinstance(data, dict):
        return {
            key: convert_paths_to_urls(value, request)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            convert_paths_to_urls(item, request)
            for item in data
        ]

    if isinstance(data, str):
        try:
            file_path = Path(data)

            relative_path = file_path.resolve().relative_to(
                settings.MEDIA_ROOT.resolve()
            )

            media_path = quote(relative_path.as_posix(), safe="/")

            return request.build_absolute_uri(
                f"{settings.MEDIA_URL}{media_path}"
            )

        except (ValueError, OSError):
            return data

    return data


def get_media_relative_path(file_path):
    """Store a media-relative path instead of an absolute local path."""

    if not file_path:
        return ""

    try:
        return Path(file_path).resolve().relative_to(
            settings.MEDIA_ROOT.resolve()
        ).as_posix()

    except (ValueError, OSError):
        return ""


def build_assessment(result):
    """
    Build a patient-friendly assessment from existing model outputs.
    """
    predicted_class = result["predicted_class"]
    confidence = result["selected_class_probability_percent"]
    probabilities = result["class_probabilities_percent"]

    if predicted_class == "Normal":
        assessment_type = "normal"
        pneumonia_prediction = False
        summary = (
            "The AI model predicts the Normal category. "
            "No pneumonia-related category was predicted."
        )
        predicted_pneumonia_type = None
    else:
        assessment_type = "pneumonia_pattern_predicted"
        pneumonia_prediction = True
        summary = (
            f"The AI model predicts a pneumonia-related pattern "
            f"in the {predicted_class} category."
        )
        predicted_pneumonia_type = predicted_class

    infection = result.get("infection_segmentation", {})

    if predicted_class == "COVID":
        infection_extent = {
            "available": True,
            "value_percent": infection.get(
                "mask_area_percent_of_256_input"
            ),
        }
    else:
        infection_extent = {
            "available": False,
            "value_percent": None,
        }

    # The pneumonia_pattern_predicted field is a model-derived result,
    # not proof that the patient has pneumonia or that pneumonia is absent.
    return {
        "assessment_type": assessment_type,
        "pneumonia_prediction": pneumonia_prediction,
        "predicted_pneumonia_type": predicted_pneumonia_type,
        "summary": summary,
        "confidence_percent": confidence,
        "class_probabilities_percent": probabilities,
        "infection_extent": infection_extent,
        "disclaimer": (
            "The pneumonia_pattern_predicted field is a model-derived result, "
            "not proof that the patient has pneumonia or that pneumonia is absent."
        ),
    }


def save_prediction(image_name, result):
    """Save inference results in the database."""

    probabilities = result["class_probabilities_percent"]

    lung = result["lung_segmentation"]
    infection = result["infection_segmentation"]
    gradcam = result["gradcam"]
    visualizations = result["visualizations"]

    prediction = Prediction.objects.create(

        # Image information
        image_name=image_name,

        # Classification
        predicted_class=result["predicted_class"],
        confidence=result["selected_class_probability_percent"],

        normal_probability=probabilities["Normal"],
        viral_probability=probabilities["Viral"],
        bacterial_probability=probabilities["Bacterial"],
        covid_probability=probabilities["COVID"],

        decision_rule=result["decision_rule"],

        # Lung segmentation
        lung_threshold=lung["threshold"],
        lung_mask_area_pixels=lung["mask_area_pixels"],
        lung_mask_area_percent=lung["mask_area_percent_of_224_input"],

        # Infection segmentation
        infection_threshold=infection["threshold"],
        infection_mask_area_pixels=infection["mask_area_pixels"],
        infection_mask_area_percent=infection["mask_area_percent_of_256_input"],

        # Grad-CAM
        gradcam_inside_lung_percent=gradcam.get(
            "activation_inside_lung_percent"
        ),
        gradcam_outside_lung_percent=gradcam.get(
            "activation_outside_lung_percent"
        ),

        # Generated image paths
        input_image_path=get_media_relative_path(
            visualizations.get("input")
        ),

        lung_mask_path=get_media_relative_path(
            lung.get("mask_path")
        ),

        infection_mask_path=get_media_relative_path(
            infection.get("mask_path")
        ),

        lung_overlay_path=get_media_relative_path(
            visualizations.get("lung_overlay")
        ),

        infection_overlay_path=get_media_relative_path(
            visualizations.get("infection_overlay")
        ),

        gradcam_overlay_path=get_media_relative_path(
            gradcam.get("overlay_path")
        ),

        gradcam_heatmap_path=get_media_relative_path(
            visualizations.get("gradcam_heatmap")
        ),

        combined_visualization_path=get_media_relative_path(
            visualizations.get("combined")
        ),

        # Research limitations
        infection_segmentation_status=infection.get(
            "status", "experimental"
        ),

        limitations=result.get("limitations", []),
    )

    return prediction


class ChestXrayInferenceAPIView(APIView):

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):

        serializer = ChestXrayUploadSerializer(
            data=request.data
        )

        if not serializer.is_valid():

            return Response(
                {
                    "success": False,
                    "message": "Invalid image upload.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded_image = serializer.validated_data["image"]

        temp_path = None

        try:

            suffix = Path(uploaded_image.name).suffix.lower()

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:

                for chunk in uploaded_image.chunks():
                    temp_file.write(chunk)

                temp_path = Path(temp_file.name)

            output_dir = OUTPUT_ROOT / uuid.uuid4().hex

            # Run existing CDSS inference pipeline
            result = run_inference(
                image_path=temp_path,
                output_dir=output_dir,
            )

            result["image_name"] = uploaded_image.name

            # Build patient-friendly assessment
            result["assessment"] = build_assessment(result)

            # Save prediction in database
            prediction = save_prediction(
                uploaded_image.name,
                result,
            )

            # Add database ID to response
            result["prediction_id"] = str(prediction.id)

            # Convert local media paths into browser URLs
            result = convert_paths_to_urls(
                result,
                request,
            )

            return Response(
                {
                    "success": True,
                    "message": "Chest X-ray processed successfully.",
                    "data": result,
                },
                status=status.HTTP_200_OK,
            )

        except Exception:

            logger.exception(
                "Chest X-ray inference or database saving failed."
            )

            return Response(
                {
                    "success": False,
                    "message": (
                        "An error occurred while processing "
                        "the chest X-ray."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        finally:

            if temp_path and temp_path.exists():
                temp_path.unlink()


from rest_framework.generics import ListAPIView, RetrieveAPIView
from .serializers import PredictionHistorySerializer


class PredictionHistoryAPIView(ListAPIView):
    """
    GET: Retrieve all saved CDSS predictions.
    """

    serializer_class = PredictionHistorySerializer

    def get_queryset(self):
        return Prediction.objects.all()


class PredictionDetailAPIView(RetrieveAPIView):
    """
    GET: Retrieve one CDSS prediction using its ID.
    """

    serializer_class = PredictionHistorySerializer
    queryset = Prediction.objects.all()

    lookup_field = "id"
    lookup_url_kwarg = "pk"