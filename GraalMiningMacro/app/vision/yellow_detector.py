"""Yellow Glow Rock Completion Detector with Multi-Frame Temporal Confirmation."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from app.vision.preprocessing import ImagePreprocessor


@dataclass
class YellowGlowDetectionResult:
    detected_raw: bool = False
    is_confirmed: bool = False               # Requires N consecutive frame confirmations
    consecutive_frames: int = 0
    required_frames: int = 3
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    area: int = 0
    confidence: float = 0.0
    matched_reference_name: str = ""

    def summary_text(self) -> str:
        if not self.is_confirmed:
            if self.detected_raw:
                return f"YELLOW GLOW (ACCUMULATING {self.consecutive_frames}/{self.required_frames})"
            return "YELLOW GLOW: NONE"
        ref_text = f" REF:{self.matched_reference_name}" if self.matched_reference_name else ""
        return f"★ ROCK COMPLETED (YELLOW GLOW {self.confidence * 100:.0f}%{ref_text})"


class YellowGlowDetector:
    """Detects yellow glowing completed rocks using HSV thresholding, reference template matching & temporal confirmation."""

    def __init__(
        self,
        hsv_min: Tuple[int, int, int] = (15, 120, 150),
        hsv_max: Tuple[int, int, int] = (35, 255, 255),
        min_area: int = 40,
        required_frames: int = 3
    ):
        self.hsv_min = list(hsv_min)
        self.hsv_max = list(hsv_max)
        self.min_area = min_area
        self.required_frames = required_frames

        self._consecutive_count = 0
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_center: Optional[Tuple[int, int]] = None

    def set_hsv_bounds(self, h_min: int, s_min: int, v_min: int, h_max: int, s_max: int, v_max: int) -> None:
        self.hsv_min = [h_min, s_min, v_min]
        self.hsv_max = [h_max, s_max, v_max]

    def detect(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> YellowGlowDetectionResult:
        if frame is None or frame.size == 0:
            self._consecutive_count = 0
            return YellowGlowDetectionResult(required_frames=self.required_frames)

        # 1. Reference Template Matching for Completed Yellow Rock
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="rock", subcategory="yellow_complete", gray_image=gray_image, match_cache=match_cache
            )
            if not ref_match or not ref_match.found:
                ref_match = reference_manager.find_best_match(
                    frame, category="rock", gray_image=gray_image, match_cache=match_cache
                )

            if ref_match and ref_match.found and "yellow" in (ref_match.subcategory + ref_match.reference_name).lower():
                self._consecutive_count += 1
                self._last_bbox = ref_match.bbox
                self._last_center = ref_match.center
                is_confirmed = (self._consecutive_count >= self.required_frames)

                return YellowGlowDetectionResult(
                    detected_raw=True,
                    is_confirmed=is_confirmed,
                    consecutive_frames=self._consecutive_count,
                    required_frames=self.required_frames,
                    bbox=ref_match.bbox,
                    center=ref_match.center,
                    area=ref_match.bbox[2] * ref_match.bbox[3] if ref_match.bbox else 100,
                    confidence=ref_match.confidence,
                    matched_reference_name=ref_match.reference_name,
                )

        target_frame = frame
        offset_x, offset_y = 0, 0

        if roi_bbox:
            cropped = ImagePreprocessor.crop_roi(frame, roi_bbox)
            if cropped is not None and cropped.size > 0:
                target_frame = cropped
                offset_x, offset_y = roi_bbox[0], roi_bbox[1]

        hsv = ImagePreprocessor.bgr_to_hsv(target_frame)
        mask = ImagePreprocessor.apply_color_mask(hsv, tuple(self.hsv_min), tuple(self.hsv_max))

        # Morphological opening & dilation to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        best_bbox = None
        best_center = None
        best_conf = 0.0

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                x += offset_x
                y += offset_y
                conf = min(1.0, area / 180.0)

                if area > best_area:
                    best_area = area
                    best_bbox = (x, y, w, h)
                    best_center = (x + w // 2, y + h // 2)
                    best_conf = conf

        if best_bbox:
            self._consecutive_count += 1
            self._last_bbox = best_bbox
            self._last_center = best_center
            is_confirmed = (self._consecutive_count >= self.required_frames)

            return YellowGlowDetectionResult(
                detected_raw=True,
                is_confirmed=is_confirmed,
                consecutive_frames=self._consecutive_count,
                required_frames=self.required_frames,
                bbox=best_bbox,
                center=best_center,
                area=best_area,
                confidence=best_conf,
            )
        else:
            self._consecutive_count = 0
            self._last_bbox = None
            self._last_center = None
            return YellowGlowDetectionResult(
                detected_raw=False,
                is_confirmed=False,
                consecutive_frames=0,
                required_frames=self.required_frames,
            )
