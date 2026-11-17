
from django.urls import path

from .views import (
    ChestXrayInferenceAPIView,
    PredictionHistoryAPIView,
    PredictionDetailAPIView,
    dashboard_view,
    upload_page_view,
    prediction_result_view,
    prediction_history_view,
)

urlpatterns = [
    # Frontend dashboard
    path(
        "",
        dashboard_view,
        name="cdss-dashboard",
    ),

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

    # Frontend upload page
    path(
        "upload/",
        upload_page_view,
        name="cdss-upload",
    ),

    path(
    "result/<int:pk>/",
    prediction_result_view,
    name="cdss-result",
    ),

    # Frontend prediction history page
    path(
        "records/",
        prediction_history_view,
        name="cdss-history-page",
    ),
]
