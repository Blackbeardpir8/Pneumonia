
from rest_framework import serializers


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
            from pathlib import Path
            extension = Path(image.name).suffix.lower()

        if extension not in allowed_extensions:
            raise serializers.ValidationError(
                "Unsupported image format. "
                "Upload PNG, JPG, JPEG, BMP, TIF or TIFF."
            )

        return image



from rest_framework import serializers
from .models import Prediction


class PredictionHistorySerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = "__all__"
        read_only_fields = tuple(
        field.name for field in Prediction._meta.fields
    )
