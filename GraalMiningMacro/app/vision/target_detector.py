"""Mining Target Detection and Tracking Module for Graal Mining Macro."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from app.mining.mining_target import TargetMemoryBank, MiningTarget, TargetState


@dataclass
class TargetDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    target_id: str = ""
    confidence: float = 0.0
    iteration: int = 0
    max_iterations: int = 3
    is_completed: bool = False
    age_seconds: float = 0.0
    target_state: str = "NO_TARGET"  # "NO_TARGET", "TARGET_CANDIDATE", "TARGET_CONFIRMED", "YELLOW_COMPLETE"

    def summary_text(self) -> str:
        if not self.detected or self.target_state == "NO_TARGET":
            return "TARGET: UNKNOWN"
        if self.is_completed or self.target_state == "YELLOW_COMPLETE":
            return "TARGET: YELLOW COMPLETED"
        iter_str = f"{self.iteration}/3" if self.iteration > 0 else "UNKNOWN"
        return f"TARGET ({self.confidence * 100:.0f}%) [{self.target_state}] ITER: {iter_str}"


class TargetDetector:
    """Detects mineable rock targets, tracks short-term memory bank, age, and iteration status."""

    def __init__(self, memory_bank: Optional[TargetMemoryBank] = None):
        self.memory_bank = memory_bank or TargetMemoryBank()
        self.current_target: Optional[MiningTarget] = None

    def update_target(
        self,
        center: Optional[Tuple[int, int]],
        bbox: Optional[Tuple[int, int, int, int]],
        confidence: float,
        is_yellow_completed: bool = False
    ) -> TargetDetection:
        if not center or not bbox or confidence < 0.40:
            return TargetDetection(detected=False, target_state="NO_TARGET")

        w, h = bbox[2], bbox[3]
        if w < 10 or h < 10 or w > 300 or h > 300:
            return TargetDetection(detected=False, target_state="NO_TARGET")

        target = self.memory_bank.get_or_create(center=center, bbox=bbox, confidence=confidence)
        if is_yellow_completed:
            target.is_yellow_completed = True
            target.iterations = 3
            target.state = TargetState.COMPLETED

        self.current_target = target
        age = time.time() - target.last_seen

        t_state = "YELLOW_COMPLETE" if target.is_yellow_completed else ("TARGET_CONFIRMED" if confidence >= 0.65 else "TARGET_CANDIDATE")

        return TargetDetection(
            detected=True,
            bbox=target.bbox,
            center=target.center,
            target_id=target.target_id,
            confidence=target.confidence,
            iteration=target.iterations,
            max_iterations=target.max_iterations,
            is_completed=target.is_yellow_completed,
            age_seconds=age,
            target_state=t_state,
        )
