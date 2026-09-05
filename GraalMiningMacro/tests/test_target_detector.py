"""Unit tests for Target Detector module."""

import pytest
from app.vision.target_detector import TargetDetector, TargetDetection
from app.mining.mining_target import TargetMemoryBank


def test_target_detector_creation_and_memory():
    bank = TargetMemoryBank()
    detector = TargetDetector(memory_bank=bank)

    det1 = detector.update_target(center=(100, 100), bbox=(80, 80, 40, 40), confidence=0.90)
    assert isinstance(det1, TargetDetection)
    assert det1.detected is True
    assert det1.target_id != ""
    assert det1.iteration == 0

    # Simulate yellow rock completion
    det2 = detector.update_target(center=(100, 100), bbox=(80, 80, 40, 40), confidence=0.95, is_yellow_completed=True)
    assert det2.is_completed is True
    assert det2.iteration == 3
