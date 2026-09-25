"""
test_hour4.py - Comprehensive Integration Test Suite for Hour 4 FastAPI Backend
Validates:
1. GET  /api/health
2. GET  /api/model-info
3. POST /api/predict (Single Digit)
4. POST /api/benchmark (Side-by-side comparison across all 4 models)
5. POST /api/explain/knn (Explainable AI Top-5 Neighbors)
6. POST /api/cluster/inspect (K-Means Style Archetypes)
7. POST /api/feature-maps (CNN Conv2D Activation Maps)
8. POST /api/scan/document (Multi-Digit Cheque/Form Reader)
"""

import os
import sys
import asyncio
import numpy as np

# Ensure project root & backend are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from preprocessor import encode_numpy_to_base64
from app import (
    app,
    startup_event,
    health_check,
    model_info,
    predict_single_digit,
    benchmark_all_models,
    explain_with_knn,
    inspect_handwriting_cluster,
    inspect_feature_maps,
    scan_document_digits,
    SinglePredictRequest,
    BenchmarkRequest,
    KNNExplainRequest,
    DocumentScanRequest
)


def create_synthetic_digit_b64(digit_type: str = "vertical_line") -> str:
    """Creates a base64 image simulating a user drawing on canvas."""
    canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    if digit_type == "vertical_line":  # Digit '1'
        canvas[30:170, 95:105] = 0
    elif digit_type == "box":          # Digit '0'
        canvas[40:160, 60:70] = 0
        canvas[40:160, 130:140] = 0
        canvas[40:50, 60:140] = 0
        canvas[150:160, 60:140] = 0
    return encode_numpy_to_base64(canvas)


def create_synthetic_document_b64() -> str:
    """Creates a document strip with 3 digits: '1', '0', '1'."""
    doc = np.ones((100, 300, 3), dtype=np.uint8) * 255
    # Digit 1 at x=50
    doc[20:80, 48:54] = 0
    # Digit 0 at x=150
    doc[20:80, 130:136] = 0
    doc[20:80, 164:170] = 0
    doc[20:26, 130:170] = 0
    doc[74:80, 130:170] = 0
    # Digit 1 at x=250
    doc[20:80, 248:254] = 0
    return encode_numpy_to_base64(doc)


async def run_hour4_tests():
    print("=" * 65)
    print(" OmniScan AI - Hour 4 FastAPI Backend Endpoints Test Suite ")
    print("=" * 65)

    # Trigger model registry startup
    await startup_event()

    # 1. Test Health
    print("\n[1/8] Testing GET /api/health...")
    h = await health_check()
    assert h["status"] == "healthy"
    assert h["models_ready"] is True
    print(f"  [OK] Service status: {h['status']} (Models loaded: {h['models_ready']})")

    # 2. Test Model Info
    print("\n[2/8] Testing GET /api/model-info...")
    info = await model_info()
    assert "active_models" in info
    assert len(info["active_models"]) == 4
    print(f"  [OK] Active Models: {list(info['active_models'].keys())}")

    # 3. Test Single Predict
    print("\n[3/8] Testing POST /api/predict (CNN)...")
    digit_b64 = create_synthetic_digit_b64("vertical_line")
    pred = await predict_single_digit(SinglePredictRequest(image=digit_b64, model="cnn"))
    assert pred["status"] == "success"
    assert 0 <= pred["digit"] <= 9
    assert pred["confidence"] > 0
    assert len(pred["probabilities"]) == 10
    print(f"  [OK] Prediction: Digit {pred['digit']} (Confidence: {pred['confidence'] * 100:.1f}%)")
    print(f"  * Latency: {pred['total_latency_ms']} ms")

    # 4. Test Benchmark All Models
    print("\n[4/8] Testing POST /api/benchmark (Side-by-Side Model Comparison)...")
    bench = await benchmark_all_models(BenchmarkRequest(image=digit_b64))
    assert bench["status"] == "success"
    assert "benchmark_results" in bench
    assert len(bench["benchmark_results"]) == 4
    for model_name, res in bench["benchmark_results"].items():
        print(f"  * {res['model_name']:<35}: Digit {res['digit']} (Conf: {res['confidence'] * 100:.1f}%, Latency: {res['latency_ms']:.2f}ms)")
    print(f"  [OK] Consensus Vote across all models: Digit {bench['consensus_digit']}")

    # 5. Test KNN Explainability
    print("\n[5/8] Testing POST /api/explain/knn (XAI Top-5 Prototype Retrieval)...")
    knn_exp = await explain_with_knn(KNNExplainRequest(image=digit_b64, k=5))
    assert knn_exp["status"] == "success"
    assert len(knn_exp["neighbors"]) == 5
    print(f"  [OK] Retrieved {len(knn_exp['neighbors'])} nearest training prototypes.")
    for n in knn_exp["neighbors"]:
        print(f"    - Rank #{n['rank']}: Label={n['label']}, Distance={n['distance']}")

    # 6. Test K-Means Style Clustering
    print("\n[6/8] Testing POST /api/cluster/inspect (Unsupervised Style Archetype)...")
    cluster = await inspect_handwriting_cluster(BenchmarkRequest(image=digit_b64))
    assert cluster["status"] == "success"
    c_diag = cluster["cluster_diagnostics"]
    print(f"  [OK] Assigned Style: {c_diag['style_description']}")
    print(f"  * Distance to Centroid: {c_diag['distance_to_centroid']}")

    # 7. Test Feature Maps
    print("\n[7/8] Testing POST /api/feature-maps (CNN Intermediate Conv Activations)...")
    fmaps = await inspect_feature_maps(BenchmarkRequest(image=digit_b64))
    assert fmaps["status"] == "success"
    print(f"  [OK] Extracted {len(fmaps['feature_maps'])} feature maps.")

    # 8. Test Multi-Digit Document Reader
    print("\n[8/8] Testing POST /api/scan/document (Multi-Digit Cheque/Form Reader)...")
    doc_b64 = create_synthetic_document_b64()
    doc_scan = await scan_document_digits(DocumentScanRequest(image=doc_b64, model="cnn"))
    assert doc_scan["status"] == "success"
    assert doc_scan["total_digits"] >= 3
    print(f"  [OK] Extracted Sequence: \"{doc_scan['recognized_sequence']}\" ({doc_scan['total_digits']} digits detected)")
    print(f"  * Average Confidence: {doc_scan['average_confidence'] * 100:.1f}%")
    print(f"  * Total Processing Time: {doc_scan['total_latency_ms']} ms")

    print("\n" + "=" * 65)
    print(" ALL HOUR 4 FASTAPI BACKEND TESTS PASSED WITH 100% SUCCESS! ")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_hour4_tests())
