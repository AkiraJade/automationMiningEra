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
    raw_score: float = 0.0
    confidence: float = 0.0                  # Final validated confidence score
    matched_reference_name: str = ""
    rejection_reason: str = ""

    def summary_text(self) -> str:
        if not self.is_confirmed:
            if self.detected_raw:
                return f"YELLOW GLOW (ACCUMULATING {self.consecutive_frames}/{self.required_frames}) RAW:{self.raw_score:.2f}"
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
        player_center: Optional[Tuple[int, int]] = None,
        target_center: Optional[Tuple[int, int]] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        candidate_scales: Optional[List[float]] = None,
    ) -> YellowGlowDetectionResult:
        if frame is None or frame.size == 0:
            self._consecutive_count = 0
            return YellowGlowDetectionResult(required_frames=self.required_frames)

        height, width = frame.shape[:2]

        raw_match = None
        cand_bbox = None
        cand_center = None
        cand_raw_score = 0.0
        ref_name = ""

        # 1. Reference Template Matching for Completed Yellow Rock
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame,
                category="rock",
                subcategory="yellow_complete",
                roi=roi_bbox,
                gray_image=gray_image,
                match_cache=match_cache,
                candidate_scales=candidate_scales,
                use_core=True,
            )
            if ref_match and ref_match.found and "yellow" in (ref_match.subcategory + ref_match.reference_name).lower():
                raw_match = ref_match
                cand_raw_score = getattr(ref_match, "raw_score", ref_match.confidence)
                cand_bbox = ref_match.bbox
                cand_center = ref_match.center
                ref_name = ref_match.reference_name

        # 2. HSV Color Verification Fallback if no reference match
        if not cand_bbox:
            # Color fallback requires a confirmed player or target position to bind spatial relationship
            if not player_center and not target_center:
                self._consecutive_count = 0
                self._last_bbox = None
                self._last_center = None
                return YellowGlowDetectionResult(
                    detected_raw=False,
                    is_confirmed=False,
                    consecutive_frames=0,
                    required_frames=self.required_frames,
                    rejection_reason="UNCONFIRMED_PLAYER_OR_TARGET",
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

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_area = 0
            for cnt in contours:
                area = int(cv2.contourArea(cnt))
                if 30 <= area <= 2500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    abs_x = x + offset_x
                    abs_y = y + offset_y

                    # Exclusion Zone 1: Top HUD (y < 15% height) and Status Panel for full frames (height >= 250)
                    if height >= 250 and (abs_y < int(height * 0.15) or (abs_x < int(width * 0.35) and abs_y < int(height * 0.25))):
                        continue

                    # Exclusion Zone 2: Lower Message Banner / DANGER Sign Area (y > 75% height)
                    if height >= 250 and abs_y > int(height * 0.75):
                        continue

                    # Aspect ratio check: rock completion visual is roughly square (0.4 to 2.2)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    if not (0.4 <= aspect_ratio <= 2.2):
                        continue

                    # Spatial Binding Check: Must be near player/target if player position provided
                    cx, cy = abs_x + w // 2, abs_y + h // 2
                    if player_center and player_center[0] > 0 and player_center[1] > 0:
                        dist_p = np.sqrt((cx - player_center[0])**2 + (cy - player_center[1])**2)
                        if dist_p > 120.0:  # Completed rock must be within immediate mining proximity of player
                            continue

                    conf = min(1.0, area / 180.0)
                    if area > best_area:
                        best_area = area
                        cand_bbox = (abs_x, abs_y, w, h)
                        cand_center = (cx, cy)
                        cand_raw_score = conf

        if cand_bbox and cand_raw_score >= 0.50:
            self._consecutive_count += 1
            self._last_bbox = cand_bbox
            self._last_center = cand_center
            is_confirmed = (self._consecutive_count >= self.required_frames)

            # Separate raw score from final validated confidence score
            final_confidence = min(1.0, cand_raw_score * (0.80 + 0.20 * min(1.0, self._consecutive_count / float(self.required_frames))))

            return YellowGlowDetectionResult(
                detected_raw=True,
                is_confirmed=is_confirmed,
                consecutive_frames=self._consecutive_count,
                required_frames=self.required_frames,
                bbox=cand_bbox,
                center=cand_center,
                area=cand_bbox[2] * cand_bbox[3],
                raw_score=cand_raw_score,
                confidence=final_confidence,
                matched_reference_name=ref_name,
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
