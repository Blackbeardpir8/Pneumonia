
from django.urls import path

from .views import (
    ChestXrayInferenceAPIView,
    PredictionHistoryAPIView,
    PredictionDetailAPIView,
)

urlpatterns = [
    # Upload X-ray and run CDSS inference
    path(
        "predict/",
        ChestXrayInferenceAPIView.as_view(),
        name="chest-xray-predict",
    ),

    # Retrieve all prediction records
    path(
        "history/",
        PredictionHistoryAPIView.as_view(),
        name="prediction-history",
    ),

    # Retrieve one prediction by ID
    path(
        "history/<int:pk>/",
        PredictionDetailAPIView.as_view(),
        name="prediction-detail",
    ),
]
