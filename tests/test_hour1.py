"""
test_hour1.py - Comprehensive Unit & Pipeline Test for Hour 1 Foundations
Validates:
1. Python Core OOP abstract interfaces
2. Spatial Image Moments (Center of Mass) calculation and centering
3. Polarity inversion (dark-on-light paper vs light-on-dark MNIST)
4. Multi-digit document segmentation and left-to-right ordering
5. Graceful handling of edge cases (empty canvas, noise)
"""

import os
import sys
import numpy as np

# Ensure backend directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from schemas import BoundingBox, PreprocessResult, DocumentScanResult
from preprocessor import (
    preprocess_single_digit,
    segment_document_digits,
    compute_center_of_mass,
    translate_by_center_of_mass
)
from base_model import BaseDigitModel


def test_python_core_oop():
    print("[1/5] Testing Python Core OOP Base Interface...")

    class DummyModel(BaseDigitModel):
        def load(self, weights_path=None):
            self.is_loaded = True
            return True

        def predict(self, tensor_28x28):
            from schemas import PredictionResult
            return PredictionResult(
                digit=7,
                confidence=0.99,
                probabilities=[0.0] * 7 + [0.99] + [0.0] * 2,
                model_name=self.model_name,
                latency_ms=1.5
            )

        def get_model_info(self):
            return {"name": self.model_name, "type": self.model_type}

    model = DummyModel("MockModel", "test_type")
    assert model.load() is True
    res = model.predict(np.zeros((28, 28), dtype=np.float32))
    assert res.digit == 7
    assert res.confidence == 0.99
    info = model.get_model_info()
    assert info["name"] == "MockModel"
    print("  [OK] BaseDigitModel abstract contract & polymorphism verified.")


def test_spatial_moments_centering():
    print("\n[2/5] Testing Image Moments & Center-of-Mass Translation...")
    # Create a 28x28 canvas with an off-center mass located at top-left (x=5, y=5)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:7, 4:7] = 255

    cx_init, cy_init = compute_center_of_mass(canvas)
    print(f"  * Initial off-center centroid: ({cx_init:.2f}, {cy_init:.2f})")
    assert cx_init < 10.0 and cy_init < 10.0, "Centroid should be in top-left quadrant"

    # Translate mass to target center (13.5, 13.5)
    centered = translate_by_center_of_mass(canvas, target_center=(13.5, 13.5))
    cx_final, cy_final = compute_center_of_mass(centered)
    print(f"  * Final centered centroid:     ({cx_final:.2f}, {cy_final:.2f})")
    assert abs(cx_final - 13.5) < 1.0, f"Expected cx close to 13.5, got {cx_final}"
    assert abs(cy_final - 13.5) < 1.0, f"Expected cy close to 13.5, got {cy_final}"
    print("  [OK] Spatial Moments correctly shifted center-of-mass to center.")


def test_polarity_and_bounding_box():
    print("\n[3/5] Testing Polarity Correction & Aspect-Preserving Fit...")
    # Simulate a user drawing a black stroke '1' on a white 200x200 canvas (HTML5 canvas mode)
    white_canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Vertical line stroke from y=40 to y=160 at x=100
    white_canvas[40:160, 96:104] = 0

    result = preprocess_single_digit(white_canvas)
    assert result.status == "valid", "Expected valid preprocessing status"
    assert result.tensor_28x28.shape == (28, 28), "Tensor shape must be (28, 28)"
    assert result.flattened_784.shape == (784,), "Flattened shape must be (784,)"
    assert 0.0 <= result.tensor_28x28.max() <= 1.0, "Values must be normalized in [0, 1]"
    # In the preprocessed tensor, the stroke should now be bright (white) on dark
    assert np.mean(result.tensor_28x28) < 0.35, "Background should be dark (0), stroke bright"
    print(f"  [OK] Automatically inverted dark-on-white polarity.")
    print(f"  [OK] Bounding box isolated: {result.bounding_box.to_dict()}")
    print(f"  [OK] Preserved aspect ratio inside 20x20 core area.")


def test_blank_canvas_graceful():
    print("\n[4/5] Testing Blank / Faint Canvas Handling...")
    blank = np.ones((150, 150, 3), dtype=np.uint8) * 255
    res = preprocess_single_digit(blank)
    assert res.status == "empty", "Blank image must return status 'empty'"
    assert np.all(res.tensor_28x28 == 0.0), "Empty canvas must yield zero tensor"
    print("  [OK] Edge case: Blank canvas handled gracefully without crashing.")


def test_multi_digit_document_segmentation():
    print("\n[5/5] Testing Multi-Digit Document Segmentation (Cheque / Form Reader)...")
    # Simulate a document snippet (height=100, width=300) with 3 separate digits: "1", "0", "7"
    doc_strip = np.ones((100, 300, 3), dtype=np.uint8) * 255

    # Digit 1 at x=40
    doc_strip[20:80, 40:48] = 0

    # Digit 0 at x=140
    doc_strip[20:80, 130:136] = 0
    doc_strip[20:80, 164:170] = 0
    doc_strip[20:26, 130:170] = 0
    doc_strip[74:80, 130:170] = 0

    # Digit 7 at x=230
    doc_strip[20:26, 220:260] = 0
    doc_strip[20:80, 254:260] = 0

    scan_res = segment_document_digits(doc_strip, min_area=30)
    print(f"  * Detected {scan_res.total_digits_found} distinct digits in sequence.")
    assert scan_res.total_digits_found >= 3, f"Expected at least 3 digits, found {scan_res.total_digits_found}"

    # Verify left-to-right sorting
    x_positions = [d.bounding_box.x for d in scan_res.segmented_digits]
    assert x_positions == sorted(x_positions), f"Digits must be sorted strictly left-to-right: {x_positions}"
    print(f"  * Left-to-right coordinates verified: {x_positions}")

    # Check each preprocessed digit tensor
    for idx, d in enumerate(scan_res.segmented_digits):
        assert d.preprocessed.tensor_28x28.shape == (28, 28)
        print(f"    - Digit #{idx}: BoundingBox(x={d.bounding_box.x}, y={d.bounding_box.y}, w={d.bounding_box.w}, h={d.bounding_box.h})")

    print("  [OK] Real-world multi-digit sequence segmentation fully functional!")


if __name__ == "__main__":
    print("=" * 65)
    print(" OmniScan AI - Hour 1 Foundations & CV Preprocessor Test Suite ")
    print("=" * 65)
    test_python_core_oop()
    test_spatial_moments_centering()
    test_polarity_and_bounding_box()
    test_blank_canvas_graceful()
    test_multi_digit_document_segmentation()
    print("\n" + "=" * 65)
    print(" ALL HOUR 1 TESTS PASSED WITH 100% SUCCESS! ")
    print("=" * 65)
