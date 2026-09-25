"""
test_hour3.py - Comprehensive Test Suite for Hour 3 Deep Learning Models (ANN & CNN)
Validates:
1. Deep Artificial Neural Network (MLP) inference & parameter count
2. Deep Convolutional Neural Network (CNN) inference & Conv2D spatial features
3. Convolutional Feature Maps Extraction (visualizing edge/stroke filters)
4. Zero-dependency Pure NumPy MLP forward-pass fallback
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

from models_dl import ANNDigitModel, CNNDigitModel, NumPyANNRunner, TORCH_AVAILABLE


def test_ann_model():
    print("[1/4] Testing Deep Artificial Neural Network (ANN)...")
    ann = ANNDigitModel()
    assert ann.load() is True
    assert ann.is_loaded is True

    # Test dummy 28x28 digit tensor
    dummy = np.zeros((28, 28), dtype=np.float32)
    dummy[8:22, 13:16] = 1.0  # simulate stroke '1'

    res = ann.predict(dummy)
    assert 0 <= res.digit <= 9, f"Predicted digit must be 0-9, got {res.digit}"
    assert 0.0 <= res.confidence <= 1.0, f"Confidence must be in [0, 1], got {res.confidence}"
    assert len(res.probabilities) == 10, "Must return 10 class probabilities"
    assert abs(sum(res.probabilities) - 1.0) < 0.05, "Probabilities must sum to ~1.0"
    assert res.latency_ms > 0, "Latency must be measured"

    info = ann.get_model_info()
    assert info["total_parameters"] == 566410
    print(f"  [OK] ANN Prediction: Digit {res.digit} (Confidence: {res.confidence * 100:.1f}%)")
    print(f"  * Active Runtime: {res.metadata.get('runtime')}")
    print(f"  * Latency: {res.latency_ms:.2f} ms | Parameters: {info['total_parameters']:,}")


def test_cnn_model():
    print("\n[2/4] Testing Deep Convolutional Neural Network (CNN)...")
    cnn = CNNDigitModel()
    assert cnn.load() is True
    assert cnn.is_loaded is True

    dummy = np.zeros((28, 28), dtype=np.float32)
    dummy[6:22, 12:15] = 1.0

    res = cnn.predict(dummy)
    assert 0 <= res.digit <= 9
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 10
    print(f"  [OK] CNN Prediction: Digit {res.digit} (Confidence: {res.confidence * 100:.1f}%)")
    print(f"  * Latency: {res.latency_ms:.2f} ms")


def test_cnn_feature_maps_extraction():
    print("\n[3/4] Testing CNN Feature Maps Visualizer (XAI Filter Inspection)...")
    cnn = CNNDigitModel()
    cnn.load()

    dummy = np.zeros((28, 28), dtype=np.float32)
    dummy[6:22, 12:15] = 1.0

    fmaps = cnn.get_feature_map_previews(dummy, num_filters=4)
    assert len(fmaps) == 4, f"Expected 4 filter previews, got {len(fmaps)}"
    for f in fmaps:
        assert "filter_index" in f
        assert "layer" in f
        assert f["feature_map_base64"].startswith("data:image/png;base64,")
    print(f"  [OK] Extracted {len(fmaps)} Conv2D intermediate activation maps for visual inspection.")


def test_pure_numpy_ann_fallback():
    print("\n[4/4] Testing Zero-Dependency Pure NumPy ANN Runner...")
    runner = NumPyANNRunner()
    dummy_input = np.random.rand(28, 28).astype(np.float32)

    probs = runner.forward(dummy_input)
    assert probs.shape == (10,), f"Expected shape (10,), got {probs.shape}"
    assert 0.0 <= probs.max() <= 1.0
    assert abs(np.sum(probs) - 1.0) < 1e-4, "Probabilities must sum to 1.0"
    pred_digit = int(np.argmax(probs))
    print(f"  [OK] Pure NumPy forward pass verified: Predicted {pred_digit} with sum(p)={np.sum(probs):.4f}")


if __name__ == "__main__":
    print("=" * 65)
    print(" OmniScan AI - Hour 3 Deep Learning Models (ANN & CNN) Test Suite ")
    print("=" * 65)
    test_ann_model()
    test_cnn_model()
    test_cnn_feature_maps_extraction()
    test_pure_numpy_ann_fallback()
    print("\n" + "=" * 65)
    print(" ALL HOUR 3 TESTS PASSED WITH 100% SUCCESS! ")
    print("=" * 65)
