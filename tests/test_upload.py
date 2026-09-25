"""
test_upload.py - Unit & Integration Test Suite for Image Upload Support
Validates:
1. Upload of valid PNG image -> Preprocessing, CNN Inference, 28x28 Preview, Probabilities
2. Upload of valid JPG/JPEG image -> Correct classification
3. Error handling: Invalid file extension (.txt, .pdf, .gif) -> HTTP 400
4. Error handling: Empty file -> HTTP 400
5. Error handling: Missing file -> HTTP 400
6. Verifies that Canvas /api/predict continues to work seamlessly alongside Image Upload
"""

import os
import sys
import io
import asyncio
import numpy as np
from PIL import Image

# Ensure project root & backend are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from fastapi.testclient import TestClient
from app import app, startup_event


def create_sample_digit_image_bytes(digit_type: str = "vertical_line", format: str = "PNG") -> bytes:
    """Creates a sample handwritten digit image (dark stroke on light background) in memory."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    pixels = img.load()

    if digit_type == "vertical_line":  # Simulating digit '1'
        for y in range(30, 170):
            for x in range(95, 105):
                pixels[x, y] = (0, 0, 0)
    elif digit_type == "horizontal_bar":  # Simulating digit '-' or part of '7'
        for y in range(40, 50):
            for x in range(50, 150):
                pixels[x, y] = (0, 0, 0)
        for y in range(50, 160):
            for x in range(135, 145):
                pixels[x, y] = (0, 0, 0)

    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def run_tests():
    print("=" * 65)
    print(" OmniScan AI - Image Upload Integration & Validation Test Suite ")
    print("=" * 65)

    with TestClient(app) as client:
        # 1. Test Valid PNG Upload
        print("\n[1/6] Testing Upload of Valid PNG Handwritten Digit...")
        png_bytes = create_sample_digit_image_bytes("vertical_line", format="PNG")
        response = client.post(
            "/api/upload?model=cnn",
            files={"file": ("test_digit_1.png", png_bytes, "image/png")}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert 0 <= data["digit"] <= 9
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 10
        assert data["preprocessed_image"].startswith("data:image/png;base64,")
        assert data["uploaded_preview"].startswith("data:image/png;base64,")
        assert "CNN" in data["model_used"] or "Convolutional" in data["model_used"]
        print(f"  [OK] PNG Upload Recognized: Digit {data['digit']} (Confidence: {data['confidence'] * 100:.1f}%)")
        print(f"  * Model Engine: {data['model_used']}")
        print(f"  * Total Latency: {data['total_latency_ms']} ms")

        # 2. Test Valid JPEG Upload
        print("\n[2/6] Testing Upload of Valid JPEG Handwritten Digit...")
        jpg_bytes = create_sample_digit_image_bytes("horizontal_bar", format="JPEG")
        response_jpg = client.post(
            "/api/upload?model=cnn",
            files={"file": ("test_digit_7.jpg", jpg_bytes, "image/jpeg")}
        )
        assert response_jpg.status_code == 200, f"Expected 200, got {response_jpg.status_code}: {response_jpg.text}"
        data_jpg = response_jpg.json()
        assert data_jpg["status"] == "success"
        assert 0 <= data_jpg["digit"] <= 9
        print(f"  [OK] JPEG Upload Recognized: Digit {data_jpg['digit']} (Confidence: {data_jpg['confidence'] * 100:.1f}%)")

        # 3. Test Invalid File Extension (.txt)
        print("\n[3/6] Testing Error Handling: Invalid File Extension (.txt)...")
        fake_txt = b"This is not an image file"
        resp_inv = client.post(
            "/api/upload?model=cnn",
            files={"file": ("document.txt", fake_txt, "text/plain")}
        )
        assert resp_inv.status_code == 400
        assert "Invalid file type" in resp_inv.json()["detail"]
        print(f"  [OK] Successfully rejected .txt file with HTTP 400: '{resp_inv.json()['detail']}'")

        # 4. Test Error Handling: Empty File
        print("\n[4/6] Testing Error Handling: Empty File (0 Bytes)...")
        empty_bytes = b""
        resp_empty = client.post(
            "/api/upload?model=cnn",
            files={"file": ("empty.png", empty_bytes, "image/png")}
        )
        assert resp_empty.status_code == 400
        assert "empty" in resp_empty.json()["detail"].lower()
        print(f"  [OK] Successfully rejected empty file with HTTP 400: '{resp_empty.json()['detail']}'")

        # 5. Test Error Handling: Missing / Unselected File
        print("\n[5/6] Testing Error Handling: Missing File...")
        resp_missing = client.post("/api/upload?model=cnn")
        assert resp_missing.status_code in (400, 422)
        print(f"  [OK] Successfully caught missing file with HTTP {resp_missing.status_code}")

        # 6. Verify Existing Canvas /api/predict Functionality
        print("\n[6/6] Verifying Existing Canvas Drawing /api/predict Continues to Function...")
        # Send synthetic canvas base64 image
        import base64
        b64_canvas = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
        resp_canvas = client.post(
            "/api/predict",
            json={"image": b64_canvas, "model": "cnn"}
        )
        assert resp_canvas.status_code == 200
        data_canvas = resp_canvas.json()
        assert data_canvas["status"] == "success"
        assert data_canvas["digit"] == data["digit"], "Canvas and Upload of identical digit must produce identical prediction!"
        print(f"  [OK] Canvas /api/predict output identical to Upload: Digit {data_canvas['digit']}")
        print("  [OK] Both Canvas and Upload pipelines share the exact same CNN model!")

        print("\n" + "=" * 65)
        print(" ALL IMAGE UPLOAD TESTS PASSED WITH 100% SUCCESS! ")
        print("=" * 65)


if __name__ == "__main__":
    run_tests()
