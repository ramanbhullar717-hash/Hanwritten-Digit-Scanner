"""
base_model.py - Python Core OOP Abstract Interface for OmniScan AI Models
Demonstrates clean object-oriented architecture, abstraction, and polymorphism.
Enables hot-swapping between Logistic Regression, KNN, K-Means, ANN, and CNN.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from schemas import PredictionResult


class BaseDigitModel(ABC):
    """
    Abstract Base Class for all digit recognition models in OmniScan AI.
    
    Guarantees that regardless of whether the underlying algorithm is
    classical (Logistic Regression, KNN) or deep (ANN, CNN), the consumer
    (FastAPI route handlers) interacts with identical method signatures.
    """

    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type  # e.g., 'classical_linear', 'instance_based', 'deep_mlp', 'convnet'
        self.is_loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> bool:
        """Loads weights, state dictionaries, or serialized model artifacts."""
        pass

    @abstractmethod
    def predict(self, tensor_28x28: np.ndarray) -> PredictionResult:
        """
        Executes inference on a single preprocessed 28x28 grayscale image tensor (values in [0.0, 1.0]).
        Returns standardized PredictionResult with class, confidence, and probabilities.
        """
        pass

    def batch_predict(self, tensors: List[np.ndarray]) -> List[PredictionResult]:
        """
        Processes a sequence of segmented digit tensors (e.g. from a multi-digit document).
        Subclasses may override this with vectorized batch matrix operations.
        """
        return [self.predict(t) for t in tensors]

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata such as parameter count, training accuracy, and architecture description."""
        pass
