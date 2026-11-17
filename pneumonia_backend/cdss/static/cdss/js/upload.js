const imageInput = document.getElementById("xrayImage");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const analyzeButton = document.getElementById("analyzeButton");
const removeImage = document.getElementById("removeImage");
const fileName = document.getElementById("fileName");
const uploadError = document.getElementById("uploadError");
let previewUrl = null;

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    uploadError.hidden = true;
    if (!file) return;

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/bmp",
        "image/tiff"
    ];
    if (!allowedTypes.includes(file.type)) {
        showError("Please select a valid chest X-ray image.");
        resetImage();
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError("Image size must not exceed 10 MB.");
        resetImage();
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    fileName.textContent = file.name;
    uploadPlaceholder.hidden = true;
    previewContainer.hidden = false;
    analyzeButton.disabled = false;
});

removeImage.addEventListener("click", resetImage);

function resetImage() {
    imageInput.value = "";
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
    imagePreview.src = "";
    previewContainer.hidden = true;
    uploadPlaceholder.hidden = false;
    analyzeButton.disabled = true;
}

function showError(message) {
    uploadError.textContent = message;
    uploadError.hidden = false;
}

// =====================================================
// SUBMIT CHEST X-RAY TO DJANGO INFERENCE API
// =====================================================
const uploadForm = document.getElementById("xrayUploadForm");
const loadingMessage = document.getElementById("loadingMessage");
uploadForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) {
        showError("Please select a chest X-ray image first.");
        return;
    }
    const formData = new FormData();
    formData.append("image", file);
    uploadError.hidden = true;
    analyzeButton.disabled = true;
    loadingMessage.hidden = false;
    loadingMessage.textContent = "Processing chest X-ray. Please wait...";
    try {
        const response = await fetch(uploadForm.dataset.apiUrl, {
            method: "POST",
            body: formData
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(
                result.message || "Unable to process the chest X-ray."
            );
        }
        const predictionId = result.data.prediction_id;
        if (!predictionId) {
            throw new Error(
                "Prediction was saved, but its ID was not returned."
            );
        }
        const resultUrl = uploadForm.dataset.resultUrlTemplate.replace(
            "/0/",
            `/${predictionId}/`
        );
        window.location.href = resultUrl;
    } catch (error) {
        showError(error.message);
        analyzeButton.disabled = false;
        loadingMessage.hidden = true;
    }
});