"""
schemas.py - Data Contracts and Pydantic Schemas for OmniScan AI
Defines unified types for image preprocessing, single-digit and multi-digit detection,
and model predictions across classical ML and Deep Learning.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def aspect_ratio(self) -> float:
        return float(self.w) / float(self.h) if self.h > 0 else 1.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class PreprocessResult:
    """Output of single-digit mathematical preprocessing."""
    tensor_28x28: np.ndarray             # Normalized (28, 28) float32 in [0, 1]
    flattened_784: np.ndarray            # Flattened (784,) float32 in [0, 1]
    preview_base64: str                  # Diagnostic Data URL of what model sees
    center_of_mass: Tuple[float, float]  # (cx, cy) after alignment
    bounding_box: Optional[BoundingBox]  # Detected stroke crop coordinates
    status: str                          # 'valid', 'empty', or 'error'
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentedDigit:
    """A single isolated digit from a multi-digit document or cheque."""
    index: int                           # Left-to-right reading order (0, 1, 2...)
    bounding_box: BoundingBox            # Location on original document
    preprocessed: PreprocessResult       # Scaled & centered 28x28 tensor


@dataclass
class DocumentScanResult:
    """Output of multi-digit document segmentation."""
    total_digits_found: int
    annotated_preview_base64: str        # Image with color-coded bounding boxes
    segmented_digits: List[SegmentedDigit]
    status: str


@dataclass
class PredictionResult:
    """Unified prediction format returned by any model in the registry."""
    digit: int
    confidence: float
    probabilities: List[float]           # 10 class probabilities (0-9)
    model_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
