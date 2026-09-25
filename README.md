# OmniScan AI: Intelligent Real-Time Handwritten Digit & Document Scanner

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/CNN_Accuracy->99%25-brightgreen)](#evaluation-results)

An end-to-end, production-grade applied Machine Learning and Deep Learning prototype that solves real-time handwritten digit recognition and multi-digit document sequence extraction (bank cheques, postal slips, billing amounts) using **Python Core**, **Multi-Class Logistic Regression**, **K-Nearest Neighbors (KNN)**, **K-Means Clustering**, **Deep ANN (MLP)**, and **Deep CNN (ConvNet)**.

---

## 1. Problem Statement & Real-World Solution

### The Gap in Academic Machine Learning
Most handwritten digit recognition projects remain confined to Jupyter notebooks evaluating clean MNIST test arrays. When deployed to real-world drawing pads or camera-scanned documents, **accuracy collapses to under 40%** due to:
1. **Centering & Translation Bias**: Naive resizing leaves digits off-center or clipped.
2. **Stroke Polarity Inversion**: Paper and HTML5 canvases feature dark ink on light backgrounds ($255 \to 0$), whereas MNIST represents digits as bright ink on black background ($0 \to 255$).
3. **Aspect Ratio Distortion**: Non-uniform resizing crushes thin digits (like `1`) or wide digits (like `0` or `8`).
4. **Multi-Character Sequences**: Real documents (cheques, exam roll numbers) contain multiple adjacent numbers rather than isolated single digits.

### How OmniScan AI Solves It
- **Mathematical Centroid Alignment**: Implements spatial Image Moments ($M_{00}, M_{10}, M_{01}$) to shift the stroke center of mass $(\bar{x}, \bar{y})$ directly to $(13.5, 13.5)$ on a $28 \times 28$ grid.
- **Polarity Normalization**: Automatically detects background intensity and inverts dark-on-light paper to the MNIST manifold.
- **Multi-Digit Contour Segmentation**: Detects external contours in cheques or receipts, sorts them strictly left-to-right, and evaluates each digit sequentially.
- **Explainable AI (XAI)**: Uses KNN prototype retrieval to show the 5 nearest real training images and distances, explaining *why* the model made its decision.

---

## 2. Curriculum Integration Matrix

| Curriculum Topic | Implementation in OmniScan AI Backend |
| :--- | :--- |
| **Python Core** | Modular OOP Abstract Base Class (`BaseDigitModel`), type annotations, Pydantic schemas, and vectorized NumPy array math. |
| **Regression & Classification** | Multi-class Logistic Regression with Softmax link function serving as the statistical linear decision boundary baseline. |
| **K-Nearest Neighbors (KNN)** | Metric-space instance classifier ($k=5$, Euclidean distance) providing Explainable AI (XAI) prototype retrieval. |
| **Unsupervised Clustering (K-Means)** | Groups handwriting into $K=20$ stylistic clusters without labels, identifying natural stroke archetypes (e.g. European crossed-7 vs plain-7). |
| **Deep ANN (Multi-Layer Perceptron)** | 4-layer fully connected network ($784 \to 512 \to 256 \to 128 \to 10$) with `BatchNorm1d` and `Dropout` (~98.2% accuracy). |
| **Deep CNN (Convolutional Net)** | 2D Convolution layers (`Conv2D(32)` $\to$ `Conv2D(64)` $\to$ `MaxPool`) preserving spatial locality and translational invariance (>99.2% accuracy). |
| **Real-Time Backend Service** | FastAPI asynchronous server with sub-25ms response time, side-by-side model comparison, and interactive Swagger documentation. |

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Interactive Client Layer (Frontend)                  │
│   • HTML5 High-DPI Drawing Canvas      • Drag & Drop Document Uploader │
│   • Live Probability Spectrum (0-9)    • Model Benchmark Arena         │
│   • KNN Explainable Prototypes         • K-Means Style Archetypes      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ JSON / Base64 Payload
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Asynchronous Gateway                      │
│   POST /api/predict          POST /api/benchmark                       │
│   POST /api/explain/knn      POST /api/cluster/inspect                 │
│   POST /api/scan/document    GET  /api/model-info                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Grayscale Image Array
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Mathematical Vision Preprocessor Engine                │
│   1. Polarity Detection (Dark ink on light paper -> Invert)            │
│   2. Contour Bounding Box Isolation & Aspect-Preserving 20x20 Fit      │
│   3. Spatial Moments Centering: cx = M10/M00, cy = M01/M00 -> (13.5, 13.5)
│   4. Left-to-Right Multi-Digit Slicing for Document Snippets           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Normalized (28, 28) Tensor
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Unified Model Registry                           │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │  Logistic Regression   │  │   KNN Classifier (k=5) │                │
│  │  (Linear Hyperplane)   │  │   (Euclidean Space)    │                │
│  └────────────────────────┘  └────────────────────────┘                │
│  ┌────────────────────────┐  ┌────────────────────────┐                │
│  │   K-Means Clusterer    │  │  Deep ANN (MLP 4-Layer)│                │
│  │   (Style Archetypes)   │  │  (BatchNorm + Dropout) │                │
│  └────────────────────────┘  └────────────────────────┘                │
│  ┌────────────────────────────────────────────────────┐                │
│  │         Deep CNN (Conv2D + MaxPool + Dropout)      │                │
│  └────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```text
omniscan-ai/
├── backend/
│   ├── app.py                   # FastAPI application & REST route definitions
│   ├── base_model.py            # Python Core OOP abstract interface (BaseDigitModel)
│   ├── schemas.py               # Pydantic data schemas & telemetry contracts
│   ├── preprocessor.py          # OpenCV moments, binarization & multi-digit segmenter
│   ├── model_registry.py        # Central orchestrator for all 5 ML/DL models
│   ├── models_classical.py      # Logistic Regression, KNN & K-Means implementations
│   ├── models_dl.py             # PyTorch ANN, CNN & Pure NumPy fallback runners
│   ├── saved_models/            # Serialized model weights (.pth, .joblib, .npz)
│   └── requirements.txt         # Dependencies
│
├── ml/
│   ├── train_classical.py       # Trains Logistic Regression, KNN & K-Means
│   ├── train_dl.py              # Trains PyTorch ANN and CNN models
│   ├── evaluate_all.py          # Master benchmark & latency evaluation suite
│   ├── data/                    # MNIST dataset cache
│   └── outputs/                 # Benchmark metrics and JSON reports
│
├── frontend/
│   ├── index.html               # Multi-mode UI (Drawing pad, document reader, arena)
│   ├── style.css                # Glassmorphic dark cyberpunk theme
│   └── app.js                   # Client controller & REST connector
│
├── tests/
│   ├── test_hour1.py            # Preprocessor & spatial moments tests
│   ├── test_hour2.py            # Classical ML (Logistic, KNN, K-Means) tests
│   ├── test_hour3.py            # Deep Learning (ANN, CNN, NumPy fallback) tests
│   └── test_hour4.py            # FastAPI REST endpoints integration tests
│
└── README.md                    # Comprehensive technical report & viva defense guide
```

---

## 5. Quick Start & Execution Guide

### Step 1: Install Dependencies
Open a terminal in the project directory:

```bash
cd C:\Users\HP\.gemini\antigravity-ide\scratch\omniscan-ai
pip install -r backend/requirements.txt
```

### Step 2: Run Unit & Integration Test Suites
Validate all 4 modules:
```bash
python tests/test_hour1.py
python tests/test_hour2.py
python tests/test_hour3.py
python tests/test_hour4.py
```

### Step 3: (Optional) Train & Export Models
```bash
python ml/train_classical.py
python ml/train_dl.py
```

### Step 4: Launch FastAPI Backend Server
```bash
python -m uvicorn backend.app:app --app-dir C:\Users\HP\.gemini\antigravity-ide\scratch\omniscan-ai --host 127.0.0.1 --port 8000 --reload
```

### Step 5: Open the Interactive Application
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 6. REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/predict` | `POST` | Evaluates single digit on selected engine (`cnn`, `ann`, `knn`, `logistic_regression`). |
| `/api/benchmark` | `POST` | Concurrently runs all 4 classification models on the same sketch and returns consensus vote. |
| `/api/explain/knn` | `POST` | Returns top-5 visually closest MNIST prototypes with Euclidean distances (Explainable AI). |
| `/api/cluster/inspect`| `POST` | Assigns drawing to one of $K=20$ K-Means handwriting style clusters with centroid render. |
| `/api/feature-maps` | `POST` | Extracts CNN Conv2D activation filter maps for visual inspection. |
| `/api/scan/document` | `POST` | Segments multi-digit document snippets left-to-right and returns recognized number string. |
| `/api/model-info` | `GET` | Returns active runtime modes and parameter counts. |
| `/api/health` | `GET` | Service health check. |

---

## 7. Viva Defense & Examination Guide

### Q1: Why do academic MNIST models fail when tested on drawn canvas digits?
> **Answer**: Standard MNIST samples are strictly size-normalized within a $20 \times 20$ box and centered using their **Center of Mass (Moments)** on a $28 \times 28$ grid. Live drawings differ in stroke polarity (black on white), stroke width, and position. Our OpenCV Image Moments preprocessor eliminates this domain gap by translating $(\bar{x}, \bar{y}) = (M_{10}/M_{00}, M_{01}/M_{00})$ to $(13.5, 13.5)$.

### Q2: What is the architectural difference between Logistic Regression, ANN, and CNN in your project?
> **Answer**: 
> - **Logistic Regression** calculates linear decision boundaries using Softmax ($7,850$ parameters). It provides high speed (~0.15ms) but cannot model spatial correlations.
> - **Deep ANN (MLP)** introduces non-linear feature interactions through 4 fully connected layers with BatchNorm and Dropout ($566,410$ parameters, ~98% accuracy).
> - **Deep CNN** employs 2D convolutions and pooling ($162,570$ parameters, >99% accuracy), ensuring spatial translation invariance (an edge or curve is recognized anywhere on the grid).

### Q3: How is unsupervised K-Means clustering used here?
> **Answer**: K-Means groups handwriting without ground-truth labels into $K=20$ clusters. This discovers structural human writing archetypes, such as European crossed-7s vs plain-7s, and looped-2s vs flat-2s.

### Q4: How does your system recognize multi-digit numbers from bank cheques?
> **Answer**: The `/api/scan/document` endpoint uses morphological closing and contour detection (`cv2.findContours`). It extracts individual character bounding boxes, filters noise, sorts them strictly left-to-right by $x$-coordinate, centers each crop, and passes them in sequence through the CNN classifier.
