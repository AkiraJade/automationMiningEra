"""Comprehensive Deliverable 4 Perception Accuracy & Multi-Factor Validation Unit Tests."""

import numpy as np
import pytest
from app.core import config
from app.vision.player_detector import PlayerDetector, PlayerDetection
from app.vision.spider_detector import SpiderDetector, SpiderDetection
from app.vision.yellow_detector import YellowGlowDetector, YellowGlowDetectionResult
from app.vision.reference import ReferenceManager, ReferenceImage, ReferenceMatchResult, ReferenceMatcher
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.input.safety import safety


def test_player_detection_primary_template_and_heuristic_fallback(tmp_path):
    """Verifies template match takes priority, and empty frame falls back to PLAYER (HEURISTIC) at 0.30 confidence."""
    manager = ReferenceManager(base_dir=str(tmp_path))
    detector = PlayerDetector()

    # Empty frame -> Heuristic fallback
    frame_empty = np.zeros((200, 200, 3), dtype=np.uint8)
    det_heuristic = detector.detect(frame_empty, reference_manager=manager)

    assert det_heuristic.detected is True
    assert det_heuristic.detection_method == "HEURISTIC"
    assert det_heuristic.is_heuristic is True
    assert det_heuristic.confidence == 0.30
    assert det_heuristic.center == (100, 100)

    # Register player template reference
    rng = np.random.RandomState(123)
    player_patch = rng.randint(150, 255, (30, 30, 3), dtype=np.uint8)
    frame_with_player = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_with_player[50:80, 50:80] = player_patch

    manager.registry.add_reference(
        name="player_front",
        category="player",
        subcategory="down",
        source_file_or_image=player_patch,
        threshold=0.75
    )

    det_template = detector.detect(frame_with_player, reference_manager=manager)
    assert det_template.detected is True
    assert det_template.detection_method == "TEMPLATE"
    assert det_template.is_heuristic is False
    assert det_template.confidence >= 0.75
    assert det_template.matched_reference_name == "player_front"
    assert det_template.center == (65, 65)


def test_spider_multi_factor_validation_and_candidate_state(tmp_path):
    """Verifies size check, spatial distance, and 2-frame candidate -> confirmed spider state machine."""
    manager = ReferenceManager(base_dir=str(tmp_path))
    spider_detector = SpiderDetector(confirm_frames=2, min_size=(10, 10), max_size=(50, 50), max_distance=300.0)

    rng = np.random.RandomState(456)
    frame = rng.randint(10, 40, (300, 300, 3), dtype=np.uint8)
    spider_patch = rng.randint(150, 255, (20, 20, 3), dtype=np.uint8)
    frame[100:120, 100:120] = spider_patch

    manager.registry.add_reference(
        name="spider_ref",
        category="spider",
        subcategory="default",
        source_file_or_image=spider_patch,
        threshold=0.75
    )

    player_center = (120, 120)  # Close to spider -> dist ~28px

    # Tick 1: Candidate State
    res_tick1 = spider_detector.detect(frame, player_center=player_center, reference_manager=manager)
    assert res_tick1.detected is False
    assert res_tick1.is_candidate is True
    assert res_tick1.consecutive_frames == 1

    # Tick 2: Confirmed State
    res_tick2 = spider_detector.detect(frame, player_center=player_center, reference_manager=manager)
    assert res_tick2.detected is True
    assert res_tick2.is_candidate is False
    assert res_tick2.consecutive_frames == 2
    assert res_tick2.matched_reference_name == "spider_ref"
    assert res_tick2.distance_from_player > 0.0

    # Far player center -> Rejection due to DISTANCE_TOO_FAR
    far_player_center = (900, 900)
    res_far = spider_detector.detect(frame, player_center=far_player_center, reference_manager=manager)
    assert res_far.detected is False
    assert res_far.rejection_reason == "DISTANCE_TOO_FAR"


def test_yellow_rock_multi_factor_raw_vs_confidence(tmp_path):
    """Verifies raw_score separation from confidence and 3-frame temporal confirmation."""
    manager = ReferenceManager(base_dir=str(tmp_path))
    yellow_detector = YellowGlowDetector(required_frames=3)

    rng = np.random.RandomState(789)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    yellow_patch = rng.randint(180, 255, (25, 25, 3), dtype=np.uint8)
    yellow_patch[:, :, 0] = 0   # B
    yellow_patch[:, :, 1] = 255 # G
    yellow_patch[:, :, 2] = 255 # R (Yellow)
    frame[50:75, 50:75] = yellow_patch

    manager.registry.add_reference(
        name="yellow_rock_complete",
        category="rock",
        subcategory="yellow_complete",
        source_file_or_image=yellow_patch,
        threshold=0.80
    )

    # Tick 1: Accumulating 1/3
    res1 = yellow_detector.detect(frame, reference_manager=manager)
    assert res1.detected_raw is True
    assert res1.is_confirmed is False
    assert res1.consecutive_frames == 1
    assert res1.raw_score >= 0.80

    # Tick 2: Accumulating 2/3
    res2 = yellow_detector.detect(frame, reference_manager=manager)
    assert res2.detected_raw is True
    assert res2.is_confirmed is False
    assert res2.consecutive_frames == 2

    # Tick 3: Confirmed 3/3
    res3 = yellow_detector.detect(frame, reference_manager=manager)
    assert res3.detected_raw is True
    assert res3.is_confirmed is True
    assert res3.consecutive_frames == 3
    assert res3.confidence > res1.confidence
    assert res3.matched_reference_name == "yellow_rock_complete"


def test_multi_scale_template_matching_caching(tmp_path):
    """Verifies multi-scale matching evaluates scaled templates and caches scaled grayscale images."""
    matcher = ReferenceMatcher()
    
    rng = np.random.RandomState(999)
    tpl_bgr = rng.randint(100, 255, (30, 30, 3), dtype=np.uint8)
    
    ref_path = str(tmp_path / "test_tpl.png")
    import cv2
    cv2.imwrite(ref_path, tpl_bgr)

    ref = ReferenceImage(
        id="tpl1",
        name="Test Template",
        category="player",
        subcategory="down",
        file_path=ref_path,
        threshold=0.70
    )

    # Create search frame with 1.1x scaled template
    scaled_patch = cv2.resize(tpl_bgr, (33, 33))
    frame = np.zeros((150, 150, 3), dtype=np.uint8)
    frame[40:73, 40:73] = scaled_patch

    match_res = matcher.match_single(frame, ref, multi_scale=True)
    assert match_res.found is True
    assert match_res.raw_score >= 0.70
    assert abs(match_res.scale - 1.10) <= 0.06
    assert len(matcher._scaled_cache) > 0


def test_dry_run_mode_strictly_preserved():
    """Confirms dry-run mode remains active and zero game inputs are generated."""
    assert safety.dry_run is True
    assert safety.emergency_stopped is False
