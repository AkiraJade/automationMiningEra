"""Image Preprocessing and OpenCV Helper Utilities for Graal Mining Macro."""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """OpenCV helper functions for frame preprocessing."""

    @staticmethod
    def crop_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crops Region of Interest given bbox (x, y, w, h)."""
        x, y, w, h = bbox
        height, width = frame.shape[:2]

        x1 = max(0, min(x, width - 1))
        y1 = max(0, min(y, height - 1))
        x2 = max(x1 + 1, min(x + w, width))
        y2 = max(y1 + 1, min(y + h, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    @staticmethod
    def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    @staticmethod
    def apply_color_mask(hsv_frame: np.ndarray, lower_hsv: Tuple[int, int, int], upper_hsv: Tuple[int, int, int]) -> np.ndarray:
        lower = np.array(lower_hsv, dtype=np.uint8)
        upper = np.array(upper_hsv, dtype=np.uint8)
        return cv2.inRange(hsv_frame, lower, upper)
