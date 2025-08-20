import pytest
import sys
import os
import sqlite3
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

TEST_IMAGE_PATH = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Apple___healthy/YOUR_IMAGE_NAME_HERE.JPG"  # Replace with valid image
DB_PATH = "data/db/predictions.db"

@pytest.fixture
def test_image():
    """Fixture to ensure test image exists."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH

@pytest.fixture
def setup_db():
    """Initialize database for testing."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommendation TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def test_predict_valid_image(test_image, setup_db):
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
    assert "recommendation" in result, "Missing 'recommendation' in response"
    assert isinstance(result["confidence"], float), "Confidence is not a float"
    assert 0 <= result["confidence"] <= 1, f"Confidence {result['confidence']} out of range"
    assert result["predicted_class"] in os.listdir(
        "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    ), f"Predicted class {result['predicted_class']} not in class labels"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE image_name = ?", (Path(test_image).name,))
    saved_prediction = cursor.fetchone()
    conn.close()
    assert saved_prediction is not None, "Prediction not saved to database"
    assert saved_prediction[2] == result["predicted_class"], "Saved predicted_class mismatch"
    assert abs(saved_prediction[3] - result["confidence"]) < 1e-5, "Saved confidence mismatch"

def test_predict_invalid_file(setup_db):
    """Test prediction endpoint with a non-image file."""
    invalid_file_path = __file__
    with open(invalid_file_path, "rb") as invalid_file:
        response = client.post(
            "/api/v1/predict",
            files={"file": (Path(invalid_file_path).name, invalid_file, "text/python")}
        )
    
    assert response.status_code == 500, f"Expected status 500, got {response.status_code}"
    assert "detail" in response.json(), "Missing 'detail' in error response"
    assert "cannot identify image file" in response.json()["detail"], "Unexpected error message"