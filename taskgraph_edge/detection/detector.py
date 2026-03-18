"""
YOLOv8-nano Object Detector with ONNX Runtime support.
Optimized for edge deployment with INT8 quantization support.
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from taskgraph_edge.detection.coco_classes import COCO_CLASSES, get_class_name


@dataclass
class Detection:
    """Represents a single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    visual_features: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.area = max(0, (x2 - x1)) * max(0, (y2 - y1))


class ObjectDetector:
    """
    YOLOv8-nano detector using either ultralytics or ONNX Runtime.
    
    Supports two backends:
    1. Ultralytics (default) - easiest setup, auto-downloads model
    2. ONNX Runtime - for edge deployment with INT8 quantization
    """

    def __init__(self, config=None):
        """
        Initialize detector.
        
        Args:
            config: DetectionConfig dataclass or None for defaults
        """
        from taskgraph_edge.config import DetectionConfig
        self.config = config or DetectionConfig()
        self.model = None
        self.backend = "ultralytics"  # or "onnx"
        self._feature_dim = self.config.feature_dim
        self._load_model()

    def _load_model(self):
        """Load the YOLOv8-nano model."""
        try:
            from ultralytics import YOLO
            model_name = self.config.model
            if not model_name.endswith('.pt') and not model_name.endswith('.onnx'):
                model_name = model_name + '.pt'

            self.model = YOLO(model_name)
            self.backend = "ultralytics"
            print(f"[Detector] Loaded {model_name} via ultralytics")
        except ImportError:
            self._load_onnx_model()

    def _load_onnx_model(self):
        """Fallback: load ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort
            model_path = self.config.model
            if not model_path.endswith('.onnx'):
                model_path += '.onnx'

            self.model = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.backend = "onnx"
            print(f"[Detector] Loaded {model_path} via ONNX Runtime")
        except Exception as e:
            raise RuntimeError(
                f"Could not load object detection model. "
                f"Install ultralytics: pip install ultralytics\n"
                f"Error: {e}"
            )

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run object detection on an image.
        
        Args:
            image: BGR image (H, W, 3) numpy array
            
        Returns:
            List of Detection objects sorted by confidence
        """
        if self.backend == "ultralytics":
            return self._detect_ultralytics(image)
        else:
            return self._detect_onnx(image)

    def _detect_ultralytics(self, image: np.ndarray) -> List[Detection]:
        """Detection using ultralytics backend."""
        results = self.model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = get_class_name(cls_id)

                # Generate visual features from bbox region
                visual_feat = self._extract_visual_features(image, bbox)

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=tuple(bbox),
                    visual_features=visual_feat,
                )
                detections.append(det)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:self.config.max_detections]

    def _detect_onnx(self, image: np.ndarray) -> List[Detection]:
        """Detection using ONNX Runtime backend."""
        # Preprocess
        input_tensor = self._preprocess(image)

        # Inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})

        # Postprocess
        detections = self._postprocess_onnx(outputs[0], image.shape)
        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        size = self.config.input_size
        # Resize
        img = cv2.resize(image, (size, size))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess_onnx(self, output: np.ndarray, orig_shape: Tuple) -> List[Detection]:
        """Postprocess ONNX model output with NMS."""
        # YOLOv8 output: (1, 84, N) where 84 = 4 bbox + 80 classes
        if output.ndim == 3:
            output = output[0]  # Remove batch dim

        # Transpose if needed (84, N) -> (N, 84)
        if output.shape[0] == 84:
            output = output.T

        num_detections = output.shape[0]
        h_orig, w_orig = orig_shape[:2]
        size = self.config.input_size
        scale_x = w_orig / size
        scale_y = h_orig / size

        boxes = []
        scores = []
        class_ids = []

        for i in range(num_detections):
            # Extract class scores (columns 4-83)
            class_scores = output[i, 4:]
            max_score = np.max(class_scores)

            if max_score < self.config.confidence_threshold:
                continue

            cls_id = int(np.argmax(class_scores))

            # Extract bbox (cx, cy, w, h) -> (x1, y1, x2, y2)
            cx, cy, w, h = output[i, :4]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            # Clip to image bounds
            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(0, min(x2, w_orig - 1))
            y2 = max(0, min(y2, h_orig - 1))

            boxes.append([x1, y1, x2, y2])
            scores.append(float(max_score))
            class_ids.append(cls_id)

        # Apply NMS
        if len(boxes) > 0:
            indices = self._nms(
                np.array(boxes),
                np.array(scores),
                self.config.iou_threshold
            )

            detections = []
            for idx in indices[:self.config.max_detections]:
                bbox = tuple(boxes[idx])
                visual_feat = self._extract_visual_features(
                    np.zeros((h_orig, w_orig, 3), dtype=np.uint8),  # placeholder
                    np.array(bbox)
                )
                det = Detection(
                    class_id=class_ids[idx],
                    class_name=get_class_name(class_ids[idx]),
                    confidence=scores[idx],
                    bbox=bbox,
                    visual_features=visual_feat,
                )
                detections.append(det)

            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        return []

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def _extract_visual_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract visual features from a detected object region.
        Uses color histograms + spatial features for a lightweight representation.
        """
        x1, y1, x2, y2 = bbox[:4]
        h, w = image.shape[:2]

        # Ensure valid bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self._feature_dim, dtype=np.float32)

        # Crop region
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(self._feature_dim, dtype=np.float32)

        # Resize to fixed size for consistent features
        crop_resized = cv2.resize(crop, (32, 32))

        # Color histogram features (RGB channels, 16 bins each = 48 dims)
        features = []
        for ch in range(3):
            hist = cv2.calcHist([crop_resized], [ch], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            features.append(hist)

        # Spatial features (normalized bbox coordinates)
        spatial = np.array([
            x1 / w, y1 / h, x2 / w, y2 / h,  # normalized bbox
            (x2 - x1) / w,  # width ratio
            (y2 - y1) / h,  # height ratio
            ((x1 + x2) / 2) / w,  # center x
            ((y1 + y2) / 2) / h,  # center y
        ], dtype=np.float32)

        # Texture features: gradient magnitude stats
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        texture = np.array([
            grad_mag.mean(),
            grad_mag.std(),
            grad_mag.max(),
        ], dtype=np.float32)
        texture = texture / (texture.max() + 1e-8)

        # Combine all features
        combined = np.concatenate([
            np.concatenate(features),     # 48 dims
            spatial,                       # 8 dims
            texture,                       # 3 dims
        ])  # Total: 59 dims

        # Pad or truncate to feature_dim
        if len(combined) < self._feature_dim:
            combined = np.pad(combined, (0, self._feature_dim - len(combined)))
        else:
            combined = combined[:self._feature_dim]

        return combined.astype(np.float32)

    def export_onnx(self, output_path: str = "yolov8n.onnx"):
        """Export model to ONNX format for edge deployment."""
        if self.backend == "ultralytics":
            self.model.export(
                format="onnx",
                simplify=True,
                dynamic=False,
                imgsz=self.config.input_size,
            )
            print(f"[Detector] Exported to ONNX: {output_path}")
        else:
            print("[Detector] Model is already in ONNX format")
