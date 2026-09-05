"""Color-based Computer Vision Detectors for Graal Mining Macro."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from app.vision.preprocessing import ImagePreprocessor


@dataclass
class YellowGlowDetection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    area: int
    confidence: float


class ColorDetector:
    """Detects glowing yellow rocks and visual color indicators using OpenCV HSV."""

    def __init__(self, yellow_hsv_min: Tuple[int, int, int] = (15, 120, 150), yellow_hsv_max: Tuple[int, int, int] = (35, 255, 255)):
        self.yellow_hsv_min = yellow_hsv_min
        self.yellow_hsv_max = yellow_hsv_max

    def detect_yellow_glow_rocks(self, frame: np.ndarray, min_area: int = 40) -> List[YellowGlowDetection]:
        """Detects yellow glowing completed rocks in frame."""
        if frame is None or frame.size == 0:
            return []

        hsv = ImagePreprocessor.bgr_to_hsv(frame)
        mask = ImagePreprocessor.apply_color_mask(hsv, self.yellow_hsv_min, self.yellow_hsv_max)

        # Apply morphological opening & closing to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[YellowGlowDetection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
                # Calculate confidence score based on saturation & area density
                confidence = min(1.0, area / 200.0)
                detections.append(
                    YellowGlowDetection(
                        bbox=(x, y, w, h),
                        center=(center_x, center_y),
                        area=int(area),
                        confidence=confidence,
                    )
                )

        return detections
