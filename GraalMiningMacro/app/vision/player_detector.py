"""Player Detection Module for Graal Mining Macro."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PlayerDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None         # (x, y)
    raw_score: float = 0.0                           # Raw template match score
    confidence: float = 0.0                          # Final validated confidence
    movement_delta: float = 0.0                      # Distance moved since last frame (px)
    matched_reference_name: str = ""
    matched_subcategory: str = ""
    matched_reference_confidence: float = 0.0
    drill_equipped: Optional[bool] = None            # True if equipped, False if unequipped, None if unknown
    detection_method: str = "TEMPLATE"               # "TEMPLATE", "YOLO", "HEURISTIC"
    player_source: str = "NONE"                      # "REFERENCE", "YOLO", "HEURISTIC", "NONE"
    player_state: str = "PLAYER_NOT_FOUND"           # "PLAYER_CONFIRMED", "PLAYER_UNCERTAIN", "PLAYER_NOT_FOUND"
    is_heuristic: bool = False
    facing_direction: str = "UNKNOWN"                # "LEFT", "RIGHT", "UP", "DOWN", "UNKNOWN"
    rejection_reason: str = ""

    def summary_text(self) -> str:
        if not self.detected or not self.center or self.player_state == "PLAYER_NOT_FOUND":
            return "PLAYER: UNKNOWN"
        if self.is_heuristic or self.player_source == "HEURISTIC":
            return f"PLAYER (HEURISTIC) X:{self.center[0]} Y:{self.center[1]}"
        if self.matched_reference_name:
            pose_str = f" [{self.matched_subcategory.upper()}]" if self.matched_subcategory else ""
            return f"PLAYER (REF {self.confidence * 100:.0f}%) REF: {self.matched_reference_name}{pose_str} X:{self.center[0]} Y:{self.center[1]}"
        return f"PLAYER ({self.confidence * 100:.0f}%) [SOURCE:{self.player_source}] X:{self.center[0]} Y:{self.center[1]}"


class PlayerDetector:
    """Detects player character using reference library template matching, YOLO, dynamic contour scanning, and auto-learning center cropping."""

    PLAYER_MIN_MATCH_SIZE: int = 12
    PLAYER_MAX_MATCH_SIZE: int = 120

    def __init__(
        self,
        confidence_threshold: float = 0.58,
        allow_heuristic_fallback: bool = True,
        enable_auto_learning: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.enable_auto_learning = enable_auto_learning
        self._last_center: Optional[Tuple[int, int]] = None
        self._locked_scale: float = 1.0
        self._dynamic_ref_registered: bool = False
        self._dynamic_ref_name: str = "dynamic_live_player"

    def detect(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[list] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        world_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> PlayerDetection:
        if frame is None or frame.size == 0:
            self._last_center = None
            return PlayerDetection(detected=False, confidence=0.0, player_state="PLAYER_NOT_FOUND")

        height, width = frame.shape[:2]

        # 1. REFERENCE TEMPLATE MATCHING (PRIMARY DETECTOR - INCLUDES DYNAMICALLY LEARNED SPRITES)
        if reference_manager is not None:
            ref_match = None

            # Fast local tracking window if player was previously detected
            if self._last_center is not None:
                lcx, lcy = self._last_center
                track_w, track_h = 320, 320
                track_x = max(0, min(lcx - track_w // 2, width - track_w))
                track_y = max(0, min(lcy - track_h // 2, height - track_h))
                track_roi = (track_x, track_y, min(width - track_x, track_w), min(height - track_y, track_h))
                cand_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=track_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=[self._locked_scale],
                    use_core=True,
                )
                if cand_match and cand_match.found and cand_match.confidence >= self.confidence_threshold:
                    ref_match = cand_match

            # Central viewport search prior (Graal camera keeps player centered in cave)
            if ref_match is None:
                cx, cy = width // 2, height // 2
                cw, ch = int(width * 0.50), int(height * 0.55)
                central_roi = (max(0, cx - cw // 2), max(0, cy - ch // 2), min(width, cw), min(height, ch))

                # Primary scales: locked scale and native 1.0
                primary_scales = [self._locked_scale] if abs(self._locked_scale - 1.0) < 1e-3 else [self._locked_scale, 1.0]

                cand_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=central_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=primary_scales,
                    use_core=True,
                )
                if cand_match and cand_match.found and cand_match.confidence >= self.confidence_threshold:
                    ref_match = cand_match
                    if getattr(cand_match, "scale", 1.0) > 0.1:
                        self._locked_scale = float(cand_match.scale)
                elif self._last_center is None:
                    # Secondary fallback scales if primary missed and lost
                    secondary_scales = [1.05, 0.95]
                    cand_match2 = reference_manager.find_best_match(
                        frame,
                        category="player",
                        roi=central_roi,
                        gray_image=gray_image,
                        match_cache=match_cache,
                        candidate_scales=secondary_scales,
                        use_core=True,
                    )
                    if cand_match2 and cand_match2.found and cand_match2.confidence >= self.confidence_threshold:
                        ref_match = cand_match2
                        if getattr(cand_match2, "scale", 1.0) > 0.1:
                            self._locked_scale = float(cand_match2.scale)

            # Fall back to full world_roi search if tracking missed or lost
            if ref_match is None and world_roi is not None:
                ref_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=world_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=[self._locked_scale, 1.0],
                    use_core=True,
                )

            if ref_match and ref_match.found and ref_match.confidence >= self.confidence_threshold:
                bw, bh = ref_match.bbox[2], ref_match.bbox[3] if ref_match.bbox else (0, 0)
                if self.PLAYER_MIN_MATCH_SIZE <= bw <= self.PLAYER_MAX_MATCH_SIZE and self.PLAYER_MIN_MATCH_SIZE <= bh <= self.PLAYER_MAX_MATCH_SIZE:
                    center = ref_match.center
                    delta = self._calc_delta(center)
                    self._last_center = center

                    # Determine drill equipment evidence & facing direction directly from player reference template
                    ref_name_lower = ref_match.reference_name.lower()
                    sub_lower = ref_match.subcategory.lower()

                    drill_equipped = None
                    if "drillequiped" in ref_name_lower or ref_match.subcategory == "mining":
                        drill_equipped = True
                    elif ref_name_lower in ["left", "down", "right", "up"] or ref_match.subcategory in ["left", "down", "right", "up"]:
                        drill_equipped = False

                    facing = "UNKNOWN"
                    if "left" in ref_name_lower or "left" in sub_lower:
                        facing = "LEFT"
                    elif "right" in ref_name_lower or "right" in sub_lower:
                        facing = "RIGHT"
                    elif "up" in ref_name_lower or "up" in sub_lower:
                        facing = "UP"
                    elif "down" in ref_name_lower or "down" in sub_lower:
                        facing = "DOWN"

                    return PlayerDetection(
                        detected=True,
                        bbox=ref_match.bbox,
                        center=center,
                        raw_score=getattr(ref_match, "raw_score", ref_match.confidence),
                        confidence=ref_match.confidence,
                        movement_delta=delta,
                        matched_reference_name=ref_match.reference_name,
                        matched_subcategory=ref_match.subcategory,
                        matched_reference_confidence=ref_match.confidence,
                        drill_equipped=drill_equipped,
                        facing_direction=facing,
                        detection_method="TEMPLATE",
                        player_source="REFERENCE" if "dynamic" not in ref_name_lower else "DYNAMIC_SCAN",
                        player_state="PLAYER_CONFIRMED",
                        is_heuristic=False,
                    )

        # 2. YOLO DETECTOR FALLBACK
        if yolo_detections:
            for det in yolo_detections:
                if getattr(det, "class_name", "") == "player" and det.confidence >= self.confidence_threshold:
                    center = det.center
                    delta = self._calc_delta(center)
                    self._last_center = center
                    return PlayerDetection(
                        detected=True,
                        bbox=det.bbox,
                        center=center,
                        raw_score=det.confidence,
                        confidence=det.confidence,
                        movement_delta=delta,
                        detection_method="YOLO",
                        player_source="YOLO",
                        player_state="PLAYER_CONFIRMED",
                        is_heuristic=False,
                    )

        # 3. DYNAMIC CONTOUR & SHADOW BLOB SCANNER (Scans camera center for player sprite)
        dyn_center, dyn_bbox = self._detect_dynamic_contour(frame, gray_image)
        if dyn_center and dyn_bbox:
            cx, cy = dyn_center
            delta = self._calc_delta(dyn_center)
            self._last_center = dyn_center

            # AUTO-LEARNING: Register cropped sprite dynamically into reference manager if enabled
            if self.enable_auto_learning and reference_manager is not None and getattr(reference_manager, 'registry', None) is not None:
                has_player_refs = len(reference_manager.registry.get_enabled_by_category("player")) > 0
                if not has_player_refs:
                    self._auto_learn_player_sprite(frame, dyn_center, dyn_bbox, reference_manager)

            return PlayerDetection(
                detected=True,
                bbox=dyn_bbox,
                center=dyn_center,
                raw_score=0.78,
                confidence=0.78,
                movement_delta=delta,
                matched_reference_name="",
                matched_subcategory="dynamic" if self._dynamic_ref_registered else "",
                matched_reference_confidence=0.0,
                detection_method="DYNAMIC_SCAN",
                player_source="DYNAMIC_SCAN",
                player_state="PLAYER_CONFIRMED",
                is_heuristic=False,
            )

        # 4. EXPLICIT CENTER HEURISTIC FALLBACK (Labeled clearly as HEURISTIC with low confidence 0.30)
        if self.allow_heuristic_fallback:
            cx, cy = width // 2, height // 2
            pw, ph = 36, 48
            bbox = (cx - pw // 2, cy - ph // 2, pw, ph)
            center = (cx, cy)
            heuristic_conf = 0.30  # Low confidence marker for unconfirmed heuristic fallback

            delta = self._calc_delta(center)
            self._last_center = center

            return PlayerDetection(
                detected=True,
                bbox=bbox,
                center=center,
                raw_score=0.30,
                confidence=heuristic_conf,
                movement_delta=delta,
                detection_method="HEURISTIC",
                player_source="HEURISTIC",
                player_state="PLAYER_UNCERTAIN",
                is_heuristic=True,
            )

        self._last_center = None
        return PlayerDetection(detected=False, confidence=0.0, player_source="NONE", player_state="PLAYER_NOT_FOUND")

    def _detect_dynamic_contour(
        self, frame: np.ndarray, gray_image: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
        """Scans the central camera region for character sprite contours and ground shadows."""
        import cv2
        h, w = frame.shape[:2]
        cam_cx, cam_cy = w // 2, h // 2

        # Define search ROI centered around camera (width 40%, height 45%)
        roi_w, roi_h = int(w * 0.40), int(h * 0.45)
        rx = max(0, cam_cx - roi_w // 2)
        ry = max(0, cam_cy - roi_h // 2)

        if gray_image is not None:
            roi_gray = gray_image[ry:ry+roi_h, rx:rx+roi_w]
        elif len(frame.shape) == 3:
            roi_gray = cv2.cvtColor(frame[ry:ry+roi_h, rx:rx+roi_w], cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = frame[ry:ry+roi_h, rx:rx+roi_w]

        if roi_gray.size == 0 or roi_gray.shape[0] < 20 or roi_gray.shape[1] < 20:
            return None, None

        # Edge & Intensity contrast thresholding for sprite isolation
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_center = None
        best_bbox = None
        min_dist_to_center = float('inf')

        for c in contours:
            area = cv2.contourArea(c)
            if area < 80 or area > 6000:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            # Character sprite bounding dimensions filter
            if 18 <= bw <= 80 and 22 <= bh <= 90:
                aspect = bw / float(bh)
                if 0.35 <= aspect <= 1.4:
                    abs_x = rx + bx + bw // 2
                    abs_y = ry + by + bh // 2
                    dist = np.hypot(abs_x - cam_cx, abs_y - cam_cy)

                    if dist < min_dist_to_center:
                        min_dist_to_center = dist
                        best_center = (abs_x, abs_y)
                        best_bbox = (rx + bx, ry + by, bw, bh)

        # Accept contour if it lies within 100px of camera center
        if best_center and min_dist_to_center <= 100.0:
            return best_center, best_bbox

        return None, None

    def _auto_learn_player_sprite(
        self, frame: np.ndarray, center: Tuple[int, int], bbox: Tuple[int, int, int, int], reference_manager: object
    ) -> None:
        """Crops current character sprite from frame and registers it dynamically into reference library if texture is real."""
        if self._dynamic_ref_registered:
            return

        try:
            h, w = frame.shape[:2]
            cx, cy = center
            sw, sh = 48, 56
            x1 = max(0, cx - sw // 2)
            y1 = max(0, cy - sh // 2)
            x2 = min(w, x1 + sw)
            y2 = min(h, y1 + sh)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 24 and crop.shape[1] >= 24:
                std_dev = float(np.std(crop))
                if std_dev > 8.0:
                    reference_manager.registry.add_reference(
                        name=self._dynamic_ref_name,
                        category="player",
                        subcategory="dynamic",
                        source_file_or_image=crop,
                        threshold=0.62
                    )
                    self._dynamic_ref_registered = True
        except Exception:
            pass

    def _calc_delta(self, current_center: Tuple[int, int]) -> float:
        if not self._last_center:
            return 0.0
        dx = current_center[0] - self._last_center[0]
        dy = current_center[1] - self._last_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
