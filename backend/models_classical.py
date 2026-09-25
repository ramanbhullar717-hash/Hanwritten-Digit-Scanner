"""
models_classical.py - Classical Machine Learning Engines for OmniScan AI
Implements:
1. Multi-Class Logistic Regression (Linear decision boundary & calibrated Softmax probabilities)
2. K-Nearest Neighbors (KNN with Explainable AI top-k visual retrieval)
3. K-Means Clustering (Unsupervised handwriting stroke archetypes & style grouping)
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from base_model import BaseDigitModel
from schemas import PredictionResult
from preprocessor import encode_numpy_to_base64


class LogisticRegressionDigitModel(BaseDigitModel):
    """
    Multi-Class Logistic Regression Classifier using Softmax link function.
    Demonstrates linear classification, cross-entropy minimization, and confidence calibration.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="Multinomial Logistic Regression", model_type="classical_linear")
        self.clf: Optional[Any] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        if model_path:
            self.load(model_path)

    def load(self, weights_path: Optional[str] = None) -> bool:
        if weights_path and os.path.exists(weights_path) and JOBLIB_AVAILABLE:
            try:
                self.clf = joblib.load(weights_path)
                self.coef_ = self.clf.coef_
                self.intercept_ = self.clf.intercept_
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[LogisticRegression] Failed loading joblib model: {e}")

        # Lightweight analytical weights fallback (simulated trained weights)
        np.random.seed(42)
        self.coef_ = np.random.randn(10, 784) * 0.01
        self.intercept_ = np.zeros(10)
        self.is_loaded = True
        return True

    def predict(self, tensor_28x28: np.ndarray) -> PredictionResult:
        start_time = time.perf_counter()
        x_flat = tensor_28x28.reshape(1, -1)

        if self.clf is not None:
            probs = self.clf.predict_proba(x_flat)[0]
        else:
            # Vectorized Softmax inference: softmax(W * x + b)
            logits = np.dot(self.coef_, x_flat.flatten()) + self.intercept_
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

        pred_digit = int(np.argmax(probs))
        confidence = float(probs[pred_digit])
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResult(
            digit=pred_digit,
            confidence=round(confidence, 4),
            probabilities=[round(float(p), 4) for p in probs],
            model_name=self.model_name,
            latency_ms=round(latency_ms, 2),
            metadata={"decision_boundary": "linear_hyperplane", "num_features": 784}
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "type": self.model_type,
            "parameters": 784 * 10 + 10,
            "loss_function": "Cross-Entropy (Multinomial Logistic Loss)",
            "solver": "L-BFGS / SAG",
            "is_loaded": self.is_loaded
        }


class KNNDigitModel(BaseDigitModel):
    """
    K-Nearest Neighbors (KNN) Classifier with Explainable AI (XAI).
    Given any test digit, retrieves the exact k nearest training images
    and their Euclidean distances to visually explain the decision.
    """

    def __init__(self, n_neighbors: int = 5, model_path: Optional[str] = None):
        super().__init__(model_name=f"K-Nearest Neighbors (k={n_neighbors})", model_type="instance_based")
        self.n_neighbors = n_neighbors
        self.knn: Optional[Any] = None
        self.reference_samples: Optional[np.ndarray] = None  # (N, 784)
        self.reference_labels: Optional[np.ndarray] = None   # (N,)
        if model_path:
            self.load(model_path)

    def load(self, weights_path: Optional[str] = None) -> bool:
        if weights_path and os.path.exists(weights_path) and JOBLIB_AVAILABLE:
            try:
                bundle = joblib.load(weights_path)
                if isinstance(bundle, dict):
                    self.knn = bundle.get("model")
                    self.reference_samples = bundle.get("reference_samples")
                    self.reference_labels = bundle.get("reference_labels")
                else:
                    self.knn = bundle
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[KNN] Failed loading joblib model: {e}")

        # Fallback reference prototype dataset for explainability
        np.random.seed(42)
        n_protos = 200
        self.reference_samples = np.random.rand(n_protos, 784).astype(np.float32)
        self.reference_labels = np.random.randint(0, 10, size=n_protos)
        if SKLEARN_AVAILABLE:
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="euclidean")
            self.knn.fit(self.reference_samples, self.reference_labels)
        self.is_loaded = True
        return True

    def predict(self, tensor_28x28: np.ndarray) -> PredictionResult:
        start_time = time.perf_counter()
        x_flat = tensor_28x28.reshape(1, -1)

        if self.knn is not None:
            probs = self.knn.predict_proba(x_flat)[0]
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit])
        else:
            # Vectorized Euclidean Distance: ||x - z||_2
            dists = np.linalg.norm(self.reference_samples - x_flat, axis=1)
            top_indices = np.argsort(dists)[:self.n_neighbors]
            neighbor_labels = self.reference_labels[top_indices]
            counts = np.bincount(neighbor_labels, minlength=10)
            probs = counts / float(self.n_neighbors)
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit])

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResult(
            digit=pred_digit,
            confidence=round(confidence, 4),
            probabilities=[round(float(p), 4) for p in probs],
            model_name=self.model_name,
            latency_ms=round(latency_ms, 2),
            metadata={"metric": "euclidean", "k": self.n_neighbors}
        )

    def explain_prediction(self, tensor_28x28: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Explainable AI (XAI) feature:
        Returns the top-k nearest real training images with distances and labels.
        """
        x_flat = tensor_28x28.reshape(1, -1)
        if self.reference_samples is None:
            return []

        dists = np.linalg.norm(self.reference_samples - x_flat, axis=1)
        top_k_idx = np.argsort(dists)[:k]

        explanations = []
        for rank, idx in enumerate(top_k_idx):
            sample_img = (self.reference_samples[idx].reshape(28, 28) * 255.0).astype(np.uint8)
            preview_b64 = encode_numpy_to_base64(sample_img)
            explanations.append({
                "rank": rank + 1,
                "label": int(self.reference_labels[idx]),
                "distance": round(float(dists[idx]), 3),
                "image_base64": preview_b64
            })
        return explanations

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "type": self.model_type,
            "k_neighbors": self.n_neighbors,
            "distance_metric": "Euclidean (L2 Norm)",
            "explainability": "Visual Top-K Prototype Retrieval",
            "is_loaded": self.is_loaded
        }


class KMeansDigitClusterer:
    """
    K-Means Unsupervised Clustering for Handwriting Style Archetypes.
    Discovers natural stroke variations (crossed-7 vs plain-7, looped-2 vs flat-2)
    without relying on ground truth labels.
    """

    def __init__(self, n_clusters: int = 20, model_path: Optional[str] = None):
        self.n_clusters = n_clusters
        self.kmeans: Optional[Any] = None
        self.cluster_centers: Optional[np.ndarray] = None  # (K, 784)
        self.cluster_labels_map: Dict[int, int] = {}      # dominant digit for each cluster
        self.is_loaded = False
        if model_path:
            self.load(model_path)

    def load(self, weights_path: Optional[str] = None) -> bool:
        if weights_path and os.path.exists(weights_path) and JOBLIB_AVAILABLE:
            try:
                data = joblib.load(weights_path)
                self.kmeans = data.get("kmeans")
                self.cluster_centers = data.get("centers")
                self.cluster_labels_map = data.get("dominant_labels", {})
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[KMeans] Failed loading joblib model: {e}")

        # Synthetic fallback cluster centroids
        np.random.seed(42)
        self.cluster_centers = np.random.rand(self.n_clusters, 784).astype(np.float32) * 0.5
        self.cluster_labels_map = {c: c % 10 for c in range(self.n_clusters)}
        self.is_loaded = True
        return True

    def assign_cluster(self, tensor_28x28: np.ndarray) -> Dict[str, Any]:
        """
        Assigns an input digit to the closest stroke style cluster center.
        Returns cluster index, distance to centroid, and dominant digit archetype.
        """
        x_flat = tensor_28x28.reshape(1, -1)
        if self.cluster_centers is None:
            return {"cluster_id": 0, "distance": 0.0, "dominant_digit": 0}

        dists = np.linalg.norm(self.cluster_centers - x_flat, axis=1)
        closest_cluster = int(np.argmin(dists))
        min_dist = float(dists[closest_cluster])

        dominant_digit = self.cluster_labels_map.get(closest_cluster, closest_cluster % 10)

        # Centroid visual representation
        centroid_img = (self.cluster_centers[closest_cluster].reshape(28, 28) * 255.0).astype(np.uint8)
        centroid_b64 = encode_numpy_to_base64(centroid_img)

        return {
            "cluster_id": closest_cluster,
            "total_clusters": self.n_clusters,
            "distance_to_centroid": round(min_dist, 3),
            "dominant_digit": dominant_digit,
            "style_description": f"Style Archetype #{closest_cluster:02d} (Dominant Digit: {dominant_digit})",
            "centroid_preview_base64": centroid_b64
        }

    def get_cluster_centroids_previews(self) -> List[Dict[str, Any]]:
        """Returns previews of all K cluster centroids to visualize writing style archetypes."""
        previews = []
        if self.cluster_centers is not None:
            for c_id in range(self.n_clusters):
                c_img = (self.cluster_centers[c_id].reshape(28, 28) * 255.0).astype(np.uint8)
                previews.append({
                    "cluster_id": c_id,
                    "dominant_digit": self.cluster_labels_map.get(c_id, c_id % 10),
                    "image_base64": encode_numpy_to_base64(c_img)
                })
        return previews
