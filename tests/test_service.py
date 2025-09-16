import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from PIL import Image
import io

from src.tbank_logo_detector.service import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock model fixture"""
    with patch("src.tbank_logo_detector.service.model") as mock:
        yield mock


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    image = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


def test_detect_logo_success(client, mock_model, sample_image):
    """Test successful logo detection"""
    mock_model.predict.return_value = []
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


def test_detect_logo_no_boxes(client, mock_model, sample_image):
    """Test logo detection when no boxes are found"""
    mock_model.predict.return_value = []
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 200


def test_detect_logo_invalid_file_type(client):
    """Test detection with non-image file"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/detect", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_detect_logo_corrupted_image(client):
    """Test detection with corrupted image"""
    files = {"file": ("test.png", b"invalid image data", "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_detect_logo_model_error(client, mock_model, sample_image):
    """Test detection when model prediction fails"""
    mock_model.predict.side_effect = Exception("Model error")

    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 500
    data = response.json()
    assert "error" in data


def test_detect_logo_empty_file(client):
    """Test detection with empty file"""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
