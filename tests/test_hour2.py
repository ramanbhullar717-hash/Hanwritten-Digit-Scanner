"""
test_hour2.py - Comprehensive Test Suite for Hour 2 Classical Machine Learning Models
Validates:
1. Multi-Class Logistic Regression (Softmax probabilities & decision boundary)
2. K-Nearest Neighbors (KNN prediction & Explainable AI top-k visual prototypes)
3. K-Means Clustering (Handwriting style cluster assignment & centroid visualization)
4. BaseDigitModel Polymorphism & Latency Telemetry
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

from models_classical import LogisticRegressionDigitModel, KNNDigitModel, KMeansDigitClusterer


def test_logistic_regression():
    print("[1/4] Testing Logistic Regression Model...")
    lr_model = LogisticRegressionDigitModel()
    assert lr_model.load() is True
    assert lr_model.is_loaded is True

    # Test dummy 28x28 digit tensor
    dummy_digit = np.zeros((28, 28), dtype=np.float32)
    dummy_digit[10:20, 13:15] = 1.0  # simulate stroke '1'

    res = lr_model.predict(dummy_digit)
    assert 0 <= res.digit <= 9, f"Predicted digit must be 0-9, got {res.digit}"
    assert 0.0 <= res.confidence <= 1.0, f"Confidence must be in [0, 1], got {res.confidence}"
    assert len(res.probabilities) == 10, f"Must return 10 class probabilities, got {len(res.probabilities)}"
    assert abs(sum(res.probabilities) - 1.0) < 0.05, f"Probabilities must sum to ~1.0, got {sum(res.probabilities)}"
    assert res.latency_ms > 0, "Latency must be measured"

    info = lr_model.get_model_info()
    assert info["name"] == "Multinomial Logistic Regression"
    print(f"  [OK] Prediction: Digit {res.digit} (Confidence: {res.confidence * 100:.1f}%)")
    print(f"  * Latency: {res.latency_ms:.2f} ms")
    print(f"  * Parameters: {info['parameters']}")


def test_knn_explainability():
    print("\n[2/4] Testing KNN Model & Explainable AI (XAI)...")
    knn_model = KNNDigitModel(n_neighbors=5)
    assert knn_model.load() is True

    dummy_digit = np.zeros((28, 28), dtype=np.float32)
    dummy_digit[5:25, 14] = 1.0

    res = knn_model.predict(dummy_digit)
    assert 0 <= res.digit <= 9
    assert len(res.probabilities) == 10
    print(f"  [OK] KNN Predicted: Digit {res.digit} (Confidence: {res.confidence * 100:.1f}%)")

    # Test XAI: Explain prediction with top-5 visual prototypes
    explanations = knn_model.explain_prediction(dummy_digit, k=5)
    assert len(explanations) == 5, f"Expected 5 neighbors, got {len(explanations)}"
    for neighbor in explanations:
        assert "rank" in neighbor
        assert "label" in neighbor
        assert "distance" in neighbor
        assert neighbor["image_base64"].startswith("data:image/png;base64,")
    print(f"  [OK] Retrieved {len(explanations)} nearest prototype images with distances: {[n['distance'] for n in explanations]}")
    print("  [OK] Explainable AI feature successfully verified.")


def test_kmeans_clustering():
    print("\n[3/4] Testing K-Means Clustering & Style Archetypes...")
    clusterer = KMeansDigitClusterer(n_clusters=20)
    assert clusterer.load() is True

    dummy_digit = np.zeros((28, 28), dtype=np.float32)
    dummy_digit[10:18, 10:18] = 0.8

    cluster_info = clusterer.assign_cluster(dummy_digit)
    assert 0 <= cluster_info["cluster_id"] < 20
    assert 0 <= cluster_info["dominant_digit"] <= 9
    assert cluster_info["distance_to_centroid"] >= 0.0
    assert cluster_info["centroid_preview_base64"].startswith("data:image/png;base64,")
    print(f"  [OK] Assigned to {cluster_info['style_description']}")
    print(f"  * Distance to centroid: {cluster_info['distance_to_centroid']}")

    # Test previews of all 20 cluster centroids
    previews = clusterer.get_cluster_centroids_previews()
    assert len(previews) == 20
    print(f"  [OK] Generated visual previews for all {len(previews)} style cluster centroids.")


def test_polymorphic_batch_inference():
    print("\n[4/4] Testing Multi-Model Polymorphic Execution...")
    models = [
        LogisticRegressionDigitModel(),
        KNNDigitModel(n_neighbors=5)
    ]
    for m in models:
        m.load()

    # Pass a simulated 3-digit sequence (e.g. from segmented document)
    tensors = [np.random.rand(28, 28).astype(np.float32) for _ in range(3)]
    for m in models:
        results = m.batch_predict(tensors)
        assert len(results) == 3
        print(f"  [OK] Model [{m.model_name}] processed batch of 3 digits successfully.")


if __name__ == "__main__":
    print("=" * 65)
    print(" OmniScan AI - Hour 2 Classical ML Models Test Suite ")
    print("=" * 65)
    test_logistic_regression()
    test_knn_explainability()
    test_kmeans_clustering()
    test_polymorphic_batch_inference()
    print("\n" + "=" * 65)
    print(" ALL HOUR 2 TESTS PASSED WITH 100% SUCCESS! ")
    print("=" * 65)
