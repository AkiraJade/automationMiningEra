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

    def summary_text(self) -> str:
        if not self.detected:
            return "TARGET: UNKNOWN"
        if self.is_completed:
            return "TARGET: YELLOW COMPLETED"
        return f"TARGET ({self.confidence * 100:.0f}%) ITER: {self.iteration}/3"


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
        if not center or not bbox or confidence <= 0.0:
            return TargetDetection(detected=False)

        target = self.memory_bank.get_or_create(center=center, bbox=bbox, confidence=confidence)
        if is_yellow_completed:
            target.is_yellow_completed = True
            target.iterations = 3
            target.state = TargetState.COMPLETED

        self.current_target = target
        age = time.time() - target.last_seen

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
        )
