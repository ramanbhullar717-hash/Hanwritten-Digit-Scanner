"""
model_registry.py - Centralized Model Registry & Inference Orchestrator for OmniScan AI
Manages lifecycle, lazy-loading, and polymorphic dispatch for:
1. Logistic Regression (Baseline linear model)
2. K-Nearest Neighbors (Instance-based explainable model)
3. K-Means Clustering (Unsupervised style archetypes)
4. Deep ANN (Multi-Layer Perceptron)
5. Deep CNN (Convolutional Neural Network)
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from base_model import BaseDigitModel
from models_classical import LogisticRegressionDigitModel, KNNDigitModel, KMeansDigitClusterer
from models_dl import ANNDigitModel, CNNDigitModel
from schemas import PredictionResult, SegmentedDigit


class ModelRegistry:
    """Singleton registry orchestrating all machine learning and deep learning models."""

    def __init__(self, saved_models_dir: Optional[str] = None):
        self.saved_models_dir = saved_models_dir or os.path.join(CURRENT_DIR, "saved_models")
        os.makedirs(self.saved_models_dir, exist_ok=True)

        # Initialize models
        self.models: Dict[str, BaseDigitModel] = {
            "logistic_regression": LogisticRegressionDigitModel(),
            "knn": KNNDigitModel(n_neighbors=5),
            "ann": ANNDigitModel(),
            "cnn": CNNDigitModel()
        }
        self.clusterer = KMeansDigitClusterer(n_clusters=20)
        self._is_initialized = False

    def initialize_all(self):
        """Loads weights and state dictionaries for all registered models."""
        print("[ModelRegistry] Initializing models from:", self.saved_models_dir)

        # 1. Logistic Regression
        lr_path = os.path.join(self.saved_models_dir, "logistic_regression.joblib")
        self.models["logistic_regression"].load(lr_path)

        # 2. KNN
        knn_path = os.path.join(self.saved_models_dir, "knn_model.joblib")
        self.models["knn"].load(knn_path)

        # 3. K-Means
        kmeans_path = os.path.join(self.saved_models_dir, "kmeans_clusterer.joblib")
        self.clusterer.load(kmeans_path)

        # 4. Deep ANN
        ann_path = os.path.join(self.saved_models_dir, "digit_ann.pth")
        self.models["ann"].load(ann_path)

        # 5. Deep CNN
        cnn_path = os.path.join(self.saved_models_dir, "digit_cnn.pth")
        self.models["cnn"].load(cnn_path)

        self._is_initialized = True
        print("[ModelRegistry] All 5 ML/DL models successfully initialized!")

    def get_model(self, key: str) -> BaseDigitModel:
        key = key.lower().strip()
        # Aliases
        alias_map = {
            "logistic": "logistic_regression",
            "lr": "logistic_regression",
            "linear": "logistic_regression",
            "mlp": "ann",
            "convnet": "cnn"
        }
        key = alias_map.get(key, key)
        if key not in self.models:
            raise ValueError(f"Unknown model '{key}'. Available: {list(self.models.keys())}")
        return self.models[key]

    def predict(self, model_name: str, tensor_28x28: np.ndarray) -> PredictionResult:
        model = self.get_model(model_name)
        return model.predict(tensor_28x28)

    def benchmark_all(self, tensor_28x28: np.ndarray) -> Dict[str, Any]:
        """
        Runs the exact same preprocessed tensor concurrently across ALL classification models.
        Returns a side-by-side comparison table of predictions, confidences, and latencies.
        """
        results = {}
        for key, model in self.models.items():
            res = model.predict(tensor_28x28)
            results[key] = {
                "model_name": res.model_name,
                "model_type": model.model_type,
                "digit": res.digit,
                "confidence": res.confidence,
                "probabilities": res.probabilities,
                "latency_ms": res.latency_ms,
                "metadata": res.metadata
            }

        # Determine consensus (majority voting among models)
        votes = [r["digit"] for r in results.values()]
        consensus_digit = max(set(votes), key=votes.count)

        return {
            "status": "success",
            "consensus_digit": consensus_digit,
            "models_evaluated": len(results),
            "benchmark_results": results
        }

    def explain_with_knn(self, tensor_28x28: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Returns top-k visual prototypes from the KNN model."""
        knn_model: KNNDigitModel = self.models["knn"]  # type: ignore
        return knn_model.explain_prediction(tensor_28x28, k=k)

    def inspect_cluster(self, tensor_28x28: np.ndarray) -> Dict[str, Any]:
        """Returns K-Means unsupervised style archetype for this handwriting."""
        return self.clusterer.assign_cluster(tensor_28x28)

    def get_cnn_feature_maps(self, tensor_28x28: np.ndarray) -> List[Dict[str, Any]]:
        """Returns internal Conv2D activation maps."""
        cnn_model: CNNDigitModel = self.models["cnn"]  # type: ignore
        return cnn_model.get_feature_map_previews(tensor_28x28)

    def scan_multi_digit_sequence(
        self,
        segmented_digits: List[SegmentedDigit],
        model_name: str = "cnn"
    ) -> Dict[str, Any]:
        """
        Solves real-world document problem:
        Takes segmented digit sequence and predicts the complete numerical string.
        """
        model = self.get_model(model_name)
        digits_str = ""
        total_conf = 0.0
        details = []

        for item in segmented_digits:
            res = model.predict(item.preprocessed.tensor_28x28)
            digits_str += str(res.digit)
            total_conf += res.confidence
            details.append({
                "index": item.index,
                "digit": res.digit,
                "confidence": res.confidence,
                "bounding_box": item.bounding_box.to_dict(),
                "probabilities": res.probabilities
            })

        avg_conf = (total_conf / len(segmented_digits)) if segmented_digits else 0.0

        return {
            "status": "success",
            "recognized_sequence": digits_str,
            "total_digits": len(segmented_digits),
            "average_confidence": round(avg_conf, 4),
            "model_used": model.model_name,
            "digit_details": details
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Returns system-wide diagnostics of all registered models."""
        return {
            "active_models": {k: m.get_model_info() for k, m in self.models.items()},
            "unsupervised_clusterer": {
                "clusters": self.clusterer.n_clusters,
                "is_loaded": self.clusterer.is_loaded
            },
            "saved_models_dir": self.saved_models_dir
        }


# Global singleton instance
registry = ModelRegistry()
