import pytest
import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve 'api' module
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

# Initialize FastAPI test client
client = TestClient(app)

# Path to a valid test image (update with a real image from your dataset)
TEST_IMAGE_PATH = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Apple___Cedar_apple_rust/0a41c25a-f9a6-4c34-8e5c-7f89a6ac4c40___FREC_C.Rust 9807_new30degFlipTB.JPG"
@pytest.fixture
def test_image():
    """Fixture to ensure test image exists."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH

def test_predict_valid_image(test_image):
    """Test prediction endpoint with a valid image."""
    with open(test_image, "rb") as image_file:
        response = client.post(
            "/api/v1/predict",
            files={"file": (Path(test_image).name, image_file, "image/jpeg")}
        )
    
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
    result = response.json()
    assert "predicted_class" in result, "Missing 'predicted_class' in response"
    assert "confidence" in result, "Missing 'confidence' in response"
    assert "message" in result, "Missing 'message' in response"
    assert isinstance(result["confidence"], float), "Confidence is not a float"
    assert 0 <= result["confidence"] <= 1, f"Confidence {result['confidence']} out of range"
    assert result["predicted_class"] in os.listdir(
        "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    ), f"Predicted class {result['predicted_class']} not in class labels"

def test_predict_invalid_file():
    """Test prediction endpoint with a non-image file."""
    invalid_file_path = __file__  # Use this test script as an invalid file
    with open(invalid_file_path, "rb") as invalid_file:
        response = client.post(
            "/api/v1/predict",
            files={"file": (Path(invalid_file_path).name, invalid_file, "text/python")}
        )
    
    assert response.status_code == 500, f"Expected status 500, got {response.status_code}"
    assert "detail" in response.json(), "Missing 'detail' in error response"
    assert "cannot identify image file" in response.json()["detail"], "Unexpected error message"