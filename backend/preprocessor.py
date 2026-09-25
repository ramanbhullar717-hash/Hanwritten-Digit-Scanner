"""
preprocessor.py - Mathematical Computer Vision Engine for OmniScan AI
Implements:
1. Polarity and Contrast Normalization (Dark-on-light vs Light-on-dark)
2. Aspect-Preserving 20x20 Bounding Box Scaling on a 28x28 Grid
3. Spatial Moments Centroid (Center-of-Mass) Affine Translation
4. Multi-Digit Document Segmentation (Cheques, IDs, Postal slips, Account numbers)
"""

import io
import re
import base64
from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from schemas import BoundingBox, PreprocessResult, SegmentedDigit, DocumentScanResult


def decode_image_to_numpy(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Decodes diverse input types (Base64 data URL, raw bytes, PIL Image, or NumPy array)
    into a standardized uint8 NumPy array (H, W, C) or (H, W).
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    if isinstance(image_input, Image.Image):
        return np.array(image_input)

    if isinstance(image_input, str):
        # Strip Data URL header if present (e.g., 'data:image/png;base64,...')
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        image_bytes = base64.b64decode(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        image_bytes = bytes(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    pil_img = Image.open(io.BytesIO(image_bytes))
    return np.array(pil_img)


def encode_numpy_to_base64(img_array: np.ndarray, format: str = "PNG") -> str:
    """Encodes a uint8 NumPy image array into a Base64 PNG Data URL."""
    # Ensure values are uint8
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255.0).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"


def compute_center_of_mass(img_28x28: np.ndarray) -> Tuple[float, float]:
    """
    Computes spatial image moments to determine the visual center of mass:
        M00 = sum(I)
        M10 = sum(x * I)
        M01 = sum(y * I)
        cx = M10 / M00, cy = M01 / M00
    """
    if CV2_AVAILABLE:
        moments = cv2.moments(img_28x28.astype(np.float32))
        m00 = moments["m00"]
        if m00 > 1e-5:
            return float(moments["m10"] / m00), float(moments["m01"] / m00)
    else:
        # Vectorized Pure NumPy moments
        total_mass = np.sum(img_28x28)
        if total_mass > 1e-5:
            y_indices, x_indices = np.indices(img_28x28.shape)
            cx = float(np.sum(x_indices * img_28x28) / total_mass)
            cy = float(np.sum(y_indices * img_28x28) / total_mass)
            return cx, cy

    return 13.5, 13.5


def translate_by_center_of_mass(img_28x28: np.ndarray, target_center: Tuple[float, float] = (13.5, 13.5)) -> np.ndarray:
    """
    Shifts the image using an affine transformation matrix so that its center of mass
    aligns exactly with the target coordinate (typically 13.5, 13.5 on a 28x28 canvas).
    """
    cx, cy = compute_center_of_mass(img_28x28)
    shift_x = target_center[0] - cx
    shift_y = target_center[1] - cy

    if CV2_AVAILABLE:
        trans_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(
            img_28x28,
            trans_mat,
            (28, 28),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return np.clip(shifted, 0, 255).astype(np.uint8)
    else:
        # Fallback NumPy affine shift using PIL
        pil_img = Image.fromarray(img_28x28)
        # Affine transform matrix in PIL expects inverse mapping: (a, b, c, d, e, f)
        shifted = pil_img.transform(
            (28, 28),
            Image.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.BILINEAR
        )
        return np.array(shifted, dtype=np.uint8)


def preprocess_single_digit(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> PreprocessResult:
    """
    Transforms any arbitrary user sketch or cropped digit image into an exact
    28x28 normalized MNIST-manifold tensor using:
    1. Polarity Correction (Dark-on-light vs Light-on-dark)
    2. Bounding Box Isolation
    3. Aspect-Preserving Fit into 20x20 Box
    4. Centering by Spatial Image Moments
    """
    try:
        raw_img = decode_image_to_numpy(image_input)

        # 1. Convert to Grayscale
        if raw_img.ndim == 3:
            if raw_img.shape[2] == 4:  # RGBA
                # Handle alpha transparency
                alpha = raw_img[:, :, 3]
                rgb = raw_img[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.array(Image.fromarray(rgb).convert("L"))
                # Blend with white background where alpha is 0
                gray = np.where(alpha < 50, 255, gray).astype(np.uint8)
            else:
                gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.array(Image.fromarray(raw_img).convert("L"))
        else:
            gray = raw_img.copy()

        # 2. Polarity Detection & Contrast Correction
        # In MNIST, digits are bright (white) on dark (black) background.
        # If the average pixel intensity > 127, it's dark ink on white paper -> Invert.
        if np.mean(gray) > 127:
            gray = 255 - gray

        # 3. Noise Reduction & Thresholding
        if CV2_AVAILABLE:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Robust NumPy thresholding: Otsu-like midpoint threshold
            positive_pixels = gray[gray > 20]
            if len(positive_pixels) > 0:
                thresh_val = max(20, int(np.mean(positive_pixels) * 0.5))
            else:
                thresh_val = 30
            thresh = np.where(gray >= thresh_val, 255, 0).astype(np.uint8)

        # Check for empty canvas
        if np.sum(thresh > 0) < 15:
            empty_tensor = np.zeros((28, 28), dtype=np.float32)
            return PreprocessResult(
                tensor_28x28=empty_tensor,
                flattened_784=empty_tensor.flatten(),
                preview_base64=encode_numpy_to_base64((empty_tensor * 255).astype(np.uint8)),
                center_of_mass=(13.5, 13.5),
                bounding_box=None,
                status="empty",
                diagnostics={"message": "Canvas is blank or stroke is too faint"}
            )

        # 4. Find Bounding Box around the stroke
        coords = cv2.findNonZero(thresh) if CV2_AVAILABLE else np.argwhere(thresh > 0)
        if CV2_AVAILABLE and coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
        elif not CV2_AVAILABLE and len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)
        else:
            x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]

        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        cropped_stroke = thresh[y:y+h, x:x+w]

        # 5. Aspect-Preserving Resize to fit inside a 20x20 box
        aspect = float(w) / float(h) if h > 0 else 1.0
        if aspect > 1.0:
            new_w = 20
            new_h = max(1, int(round(20.0 / aspect)))
        else:
            new_h = 20
            new_w = max(1, int(round(20.0 * aspect)))

        if CV2_AVAILABLE:
            resized_stroke = cv2.resize(cropped_stroke, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_stroke = np.array(Image.fromarray(cropped_stroke).resize((new_w, new_h), Image.BILINEAR))

        # 6. Paste into a 28x28 canvas (4-pixel margins around 20x20 center)
        canvas_28x28 = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas_28x28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_stroke

        # 7. Fine-Tune Alignment via Center of Mass
        centered_28x28 = translate_by_center_of_mass(canvas_28x28, target_center=(13.5, 13.5))

        # 8. Normalize to [0.0, 1.0] float32
        tensor_28x28 = centered_28x28.astype(np.float32) / 255.0
        flattened_784 = tensor_28x28.flatten()
        final_cx, final_cy = compute_center_of_mass(centered_28x28)

        preview_b64 = encode_numpy_to_base64(centered_28x28)

        return PreprocessResult(
            tensor_28x28=tensor_28x28,
            flattened_784=flattened_784,
            preview_base64=preview_b64,
            center_of_mass=(round(final_cx, 2), round(final_cy, 2)),
            bounding_box=bbox,
            status="valid",
            diagnostics={
                "cropped_size": [w, h],
                "fitted_size": [new_w, new_h],
                "centroid": [round(final_cx, 2), round(final_cy, 2)]
            }
        )

    except Exception as e:
        empty = np.zeros((28, 28), dtype=np.float32)
        return PreprocessResult(
            tensor_28x28=empty,
            flattened_784=empty.flatten(),
            preview_base64=encode_numpy_to_base64((empty * 255).astype(np.uint8)),
            center_of_mass=(13.5, 13.5),
            bounding_box=None,
            status="error",
            diagnostics={"error": str(e)}
        )


def segment_document_digits(
    image_input: Union[str, bytes, np.ndarray, Image.Image],
    min_area: int = 40,
    max_area_ratio: float = 0.95
) -> DocumentScanResult:
    """
    Real-world problem solver:
    Segments multiple handwritten digits from cheques, billing amounts, or document slips.
    Sorts detected digits strictly in left-to-right reading order and preprocesses each.
    """
    raw_img = decode_image_to_numpy(image_input)
    img_h, img_w = raw_img.shape[:2]
    total_pixels = img_h * img_w

    # Convert to Grayscale
    if raw_img.ndim == 3:
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.array(Image.fromarray(raw_img).convert("L"))
    else:
        gray = raw_img.copy()

    # Polarity correction: ensure dark ink becomes white foreground
    if np.mean(gray) > 127:
        inverted = 255 - gray
    else:
        inverted = gray.copy()

    # Thresholding & Morphological closing to bridge fragmented strokes
    if CV2_AVAILABLE:
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Fallback: Treat the whole image as a single digit if OpenCV is absent
        single = preprocess_single_digit(raw_img)
        bbox = single.bounding_box or BoundingBox(0, 0, img_w, img_h)
        return DocumentScanResult(
            total_digits_found=1 if single.status == "valid" else 0,
            annotated_preview_base64=encode_numpy_to_base64(raw_img),
            segmented_digits=[SegmentedDigit(index=0, bounding_box=bbox, preprocessed=single)],
            status="valid" if single.status == "valid" else "empty"
        )

    # Filter bounding boxes
    valid_boxes: List[BoundingBox] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter noise speckles and whole-image borders
        if area >= min_area and (w * h) < (total_pixels * max_area_ratio) and h > 10 and w > 4:
            valid_boxes.append(BoundingBox(x=x, y=y, w=w, h=h))

    # Sort bounding boxes left-to-right (primary: x coordinate)
    valid_boxes.sort(key=lambda b: b.x)

    # Annotated preview for visual telemetry
    if raw_img.ndim == 2:
        annotated = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    elif raw_img.shape[2] == 4:
        annotated = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
    else:
        annotated = raw_img.copy()

    segmented_results: List[SegmentedDigit] = []

    for idx, box in enumerate(valid_boxes):
        # Add slight 2px padding around the crop if within image boundaries
        pad = 2
        crop_x = max(0, box.x - pad)
        crop_y = max(0, box.y - pad)
        crop_w = min(img_w - crop_x, box.w + 2 * pad)
        crop_h = min(img_h - crop_y, box.h + 2 * pad)

        digit_crop = raw_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        preprocessed = preprocess_single_digit(digit_crop)

        if preprocessed.status == "valid":
            segmented_results.append(SegmentedDigit(
                index=len(segmented_results),
                bounding_box=box,
                preprocessed=preprocessed
            ))

            # Draw green bounding box & index label on preview
            cv2.rectangle(annotated, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 220, 100), 2)
            cv2.putText(annotated, f"#{len(segmented_results)}", (box.x, max(12, box.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1, cv2.LINE_AA)

    preview_b64 = encode_numpy_to_base64(annotated)

    return DocumentScanResult(
        total_digits_found=len(segmented_results),
        annotated_preview_base64=preview_b64,
        segmented_digits=segmented_results,
        status="valid" if len(segmented_results) > 0 else "empty"
    )
