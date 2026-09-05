"""Ultralytics YOLO Model Loader and Inference Engine for Graal Mining Macro."""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from app.core.logger import setup_logger

logger = setup_logger("YoloDetector")


@dataclass
class YoloDetection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]


class YoloDetector:
    """YOLO Object Detector for Player, Mining Wall, Spider, and Rocks."""

    def __init__(self, model_path: str = "models/graal_mining.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
        self.status_message = "NOT LOADED"
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            self.status_message = "MODEL NOT FOUND"
            logger.info(f"YOLO model path '{self.model_path}' not found. Passive observation mode will operate with OpenCV heuristics.")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            self.status_message = "LOADED"
            logger.info(f"YOLO model '{self.model_path}' successfully loaded on {self.device}.")
        except Exception as e:
            self.is_loaded = False
            self.status_message = f"ERROR: {e}"
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[YoloDetection]:
        if not self.is_loaded or self.model is None or frame is None:
            return []

        try:
            results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
            detections: List[YoloDetection] = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    detections.append(
                        YoloDetection(
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(x1, y1, w, h),
                            center=(x1 + w // 2, y1 + h // 2),
                        )
                    )

            return detections
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []
