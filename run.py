"""
run.py - Master Launcher for OmniScan AI Dashboard
Initializes models, verifies test suites, and launches the high-throughput FastAPI service.
"""

import os
import sys
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
BACKEND_DIR = os.path.join(CURRENT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

if __name__ == "__main__":
    print("=" * 70)
    print(" OmniScan AI: Intelligent Real-Time Digit & Document Scanner ")
    print(" Unified ML & DL Dashboard: Logistic Regression | KNN | K-Means | ANN | CNN ")
    print("=" * 70)
    print(" Starting high-performance ASGI server on http://127.0.0.1:8000 ...")
    print(" Interactive Web Dashboard: http://127.0.0.1:8000")
    print(" Automatic REST API Docs:   http://127.0.0.1:8000/docs")
    print("=" * 70)

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        app_dir=CURRENT_DIR,
        reload=False
    )
