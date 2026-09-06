"""Game Status Message & 'Nothing to Mine Here' Detector Module."""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MessageDetection:
    nothing_to_mine_detected: bool = False
    cooldown_until: float = 0.0
    cooldown_remaining: float = 0.0
    confidence: float = 0.0
    message_text: str = ""
    matched_reference_name: str = ""

    def summary_text(self) -> str:
        if self.nothing_to_mine_detected:
            ref_text = f" REF:{self.matched_reference_name}" if self.matched_reference_name else ""
            return f"⚠️ 'NOTHING TO MINE HERE'{ref_text} (COOLDOWN: {self.cooldown_remaining:.1f}s)"
        return "MSG: NONE"


class MessageDetector:
    """Detects game status messages such as 'Nothing to Mine Here' using reference template matching & text heuristics."""

    def __init__(self, default_cooldown_seconds: float = 10.0):
        self.default_cooldown_seconds = default_cooldown_seconds
        self._cooldown_until = 0.0
        self._last_matched_ref = ""

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> MessageDetection:
        if frame is None or frame.size == 0:
            return self._build_result(detected=False)

        now = time.time()
        # Active Cooldown Fast Path: Skip template matching if cooldown is currently active
        if now < self._cooldown_until:
            rem = max(0.0, self._cooldown_until - now)
            return MessageDetection(
                nothing_to_mine_detected=True,
                cooldown_until=self._cooldown_until,
                cooldown_remaining=rem,
                confidence=0.85,
                message_text="Cooldown Active",
                matched_reference_name=self._last_matched_ref,
            )

        h, w = frame.shape[:2]
        # Top HUD / Alert banner search region (excludes lower screen where chat/PM windows appear)
        alert_roi = roi if roi else (int(w * 0.10), int(h * 0.02), int(w * 0.80), int(h * 0.25))

        # 1. Reference Template Matching (Primary, evidence-based)
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="messages", roi=alert_roi, gray_image=gray_image, match_cache=match_cache
            )
            if ref_match and ref_match.found:
                self._cooldown_until = now + self.default_cooldown_seconds
                self._last_matched_ref = ref_match.reference_name

                return self._build_result(
                    detected=True,
                    confidence=ref_match.confidence,
                    text="Nothing to Mine Here",
                    ref_name=ref_match.reference_name
                )
        else:
            # Synthetic fallback only when reference_manager is absent (e.g. standalone unit tests)
            rx, ry, rw, rh = alert_roi
            hud_img = frame[ry:ry+rh, rx:rx+rw]
            gray = cv2.cvtColor(hud_img, cv2.COLOR_BGR2GRAY) if len(hud_img.shape) == 3 else hud_img
            _, text_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 150:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = bw / float(bh) if bh > 0 else 0
                    if aspect_ratio > 3.5 and bw > 100:
                        self._cooldown_until = now + self.default_cooldown_seconds
                        self._last_matched_ref = "test_banner"
                        return self._build_result(detected=True, confidence=min(1.0, area / 400.0), text="Nothing to Mine Here")

        return self._build_result(detected=False)

    def _build_result(self, detected: bool, confidence: float = 0.0, text: str = "", ref_name: str = "") -> MessageDetection:
        now = time.time()
        rem = max(0.0, self._cooldown_until - now)
        is_active = detected or (rem > 0.0)
        return MessageDetection(
            nothing_to_mine_detected=is_active,
            cooldown_until=self._cooldown_until,
            cooldown_remaining=rem,
            confidence=confidence if detected else (0.80 if rem > 0 else 0.0),
            message_text=text if detected else ("Cooldown Active" if rem > 0 else ""),
            matched_reference_name=ref_name,
        )
