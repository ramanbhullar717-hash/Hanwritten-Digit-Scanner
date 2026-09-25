"""
app.py - High-Throughput FastAPI Asynchronous Service for OmniScan AI
Exposes REST endpoints for:
1. /api/predict          - Real-time single-digit classification
2. /api/benchmark        - Side-by-side comparison of Logistic Regression, KNN, ANN & CNN
3. /api/explain/knn      - Visual top-k nearest neighbor prototypes (Explainable AI)
4. /api/cluster/inspect  - K-Means unsupervised style archetype categorization
5. /api/feature-maps     - CNN Conv2D filter activation inspection
6. /api/scan/document    - Multi-digit cheque/form scanner with contour segmentation
7. /api/upload           - Multipart file upload
8. /api/model-info       - Parameter counts and runtime diagnostics
9. /api/health           - Backend health check
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocessor import (
    preprocess_single_digit,
    segment_document_digits,
    decode_image_to_numpy,
    encode_numpy_to_base64
)
from model_registry import registry

app = FastAPI(
    title="OmniScan AI - Applied ML & Deep Learning Scanner",
    description="Real-time Handwritten Digit & Document Recognition unifying Logistic Regression, KNN, K-Means, ANN, and CNN",
    version="2.0.0"
)

# CORS middleware for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initializes models upon server launch."""
    registry.initialize_all()


# Pydantic Request & Response Models
class SinglePredictRequest(BaseModel):
    image: str = Field(..., description="Base64 Data URL or raw base64 string")
    model: str = Field(default="cnn", description="Selected model: 'cnn', 'ann', 'knn', or 'logistic_regression'")


class BenchmarkRequest(BaseModel):
    image: str = Field(..., description="Base64 Data URL or raw base64 string")


class KNNExplainRequest(BaseModel):
    image: str = Field(..., description="Base64 Data URL or raw base64 string")
    k: int = Field(default=5, ge=1, le=10, description="Number of visual neighbors to retrieve")


class DocumentScanRequest(BaseModel):
    image: str = Field(..., description="Base64 Data URL of document/cheque with multiple digits")
    model: str = Field(default="cnn", description="Inference model to recognize individual digits")


# 1. Health Endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "OmniScan AI Backend",
        "timestamp": time.time(),
        "models_ready": registry._is_initialized
    }


# 2. Model Diagnostics
@app.get("/api/model-info")
async def model_info():
    return registry.get_system_summary()


# 3. Single Digit Prediction
@app.post("/api/predict")
async def predict_single_digit(payload: SinglePredictRequest):
    t0 = time.perf_counter()
    prep = preprocess_single_digit(payload.image)

    if prep.status == "empty":
        return {
            "status": "empty",
            "message": "Canvas is blank or stroke is too faint",
            "digit": None,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "preprocessed_preview": prep.preview_base64
        }
    elif prep.status == "error":
        raise HTTPException(status_code=400, detail=prep.diagnostics.get("error", "Preprocessing failed"))

    try:
        res = registry.predict(payload.model, prep.tensor_28x28)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_latency = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "success",
        "digit": res.digit,
        "confidence": res.confidence,
        "probabilities": res.probabilities,
        "model_used": res.model_name,
        "preprocessed_image": prep.preview_base64,
        "center_of_mass": prep.center_of_mass,
        "inference_latency_ms": res.latency_ms,
        "total_latency_ms": round(total_latency, 2),
        "metadata": res.metadata
    }


# 4. Side-by-Side Model Benchmark
@app.post("/api/benchmark")
async def benchmark_all_models(payload: BenchmarkRequest):
    prep = preprocess_single_digit(payload.image)
    if prep.status == "empty":
        return {"status": "empty", "message": "Canvas is blank"}

    benchmark_data = registry.benchmark_all(prep.tensor_28x28)
    benchmark_data["preprocessed_image"] = prep.preview_base64
    benchmark_data["center_of_mass"] = prep.center_of_mass
    return benchmark_data


# 5. Explainable AI: KNN Top-K Prototypes
@app.post("/api/explain/knn")
async def explain_with_knn(payload: KNNExplainRequest):
    prep = preprocess_single_digit(payload.image)
    if prep.status == "empty":
        return {"status": "empty", "neighbors": []}

    knn_res = registry.predict("knn", prep.tensor_28x28)
    neighbors = registry.explain_with_knn(prep.tensor_28x28, k=payload.k)

    return {
        "status": "success",
        "predicted_digit": knn_res.digit,
        "confidence": knn_res.confidence,
        "k": payload.k,
        "neighbors": neighbors
    }


# 6. Unsupervised Clustering: K-Means Style Inspector
@app.post("/api/cluster/inspect")
async def inspect_handwriting_cluster(payload: BenchmarkRequest):
    prep = preprocess_single_digit(payload.image)
    if prep.status == "empty":
        return {"status": "empty"}

    cluster_info = registry.inspect_cluster(prep.tensor_28x28)
    cluster_info["preprocessed_image"] = prep.preview_base64
    return {"status": "success", "cluster_diagnostics": cluster_info}


# 7. CNN Feature Map Activations
@app.post("/api/feature-maps")
async def inspect_feature_maps(payload: BenchmarkRequest):
    prep = preprocess_single_digit(payload.image)
    if prep.status == "empty":
        return {"status": "empty", "feature_maps": []}

    maps = registry.get_cnn_feature_maps(prep.tensor_28x28)
    return {"status": "success", "feature_maps": maps}


# 8. Real-World Problem Solver: Multi-Digit Cheque/Document Reader
@app.post("/api/scan/document")
async def scan_document_digits(payload: DocumentScanRequest):
    t0 = time.perf_counter()
    doc_result = segment_document_digits(payload.image)

    if doc_result.status == "empty" or doc_result.total_digits_found == 0:
        return {
            "status": "empty",
            "message": "No valid digit strokes detected on document.",
            "recognized_sequence": "",
            "total_digits_found": 0,
            "annotated_preview": doc_result.annotated_preview_base64
        }

    sequence_result = registry.scan_multi_digit_sequence(
        doc_result.segmented_digits,
        model_name=payload.model
    )

    total_latency = (time.perf_counter() - t0) * 1000.0
    sequence_result["annotated_preview"] = doc_result.annotated_preview_base64
    sequence_result["total_latency_ms"] = round(total_latency, 2)

    return sequence_result


# 9. Multipart Image File Upload
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}

@app.post("/api/upload")
async def upload_image_file(file: UploadFile = File(None), model: str = Query("cnn")):
    # 1. Model initialization check
    if not registry._is_initialized:
        raise HTTPException(status_code=503, detail="Model engine is not loaded or still initializing.")

    # 2. File presence check
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No image selected. Please choose a JPG, JPEG, or PNG file.")

    # 3. File extension validation
    filename_lower = file.filename.lower()
    ext = os.path.splitext(filename_lower)[1]
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only JPG, JPEG, and PNG images are supported."
        )

    # 4. Content reading & empty check
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The selected file is empty. Please upload a valid image.")

    # 5. Decode image
    try:
        raw_numpy = decode_image_to_numpy(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image cannot be processed or is corrupted: {str(e)}")

    # 6. Generate Base64 preview of raw uploaded image
    b64_upload_preview = encode_numpy_to_base64(raw_numpy)

    # 7. Apply the EXACT same mathematical preprocessing pipeline as canvas
    prep = preprocess_single_digit(raw_numpy)

    if prep.status == "empty":
        return {
            "status": "empty",
            "message": "No handwritten digit stroke detected in the image. Please ensure the digit is visible.",
            "digit": None,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "uploaded_preview": b64_upload_preview,
            "preprocessed_image": prep.preview_base64,
            "center_of_mass": prep.center_of_mass
        }
    elif prep.status == "error":
        raise HTTPException(status_code=400, detail=prep.diagnostics.get("error", "Image preprocessing failed."))

    # 8. Send to the SAME existing model (PyTorch CNN by default)
    t0 = time.perf_counter()
    try:
        res = registry.predict(model, prep.tensor_28x28)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    total_latency = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "success",
        "digit": res.digit,
        "confidence": res.confidence,
        "probabilities": res.probabilities,
        "model_used": res.model_name,
        "uploaded_preview": b64_upload_preview,
        "preprocessed_image": prep.preview_base64,
        "center_of_mass": prep.center_of_mass,
        "inference_latency_ms": res.latency_ms,
        "total_latency_ms": round(total_latency, 2),
        "metadata": res.metadata
    }


# Mount Frontend UI if directory exists
FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "OmniScan AI API is running. Visit /docs for Swagger UI."}
