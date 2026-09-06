"""Comprehensive Unit Tests for Reference-Based Vision System and Calibration Library."""

import os
import time
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.vision.reference import (
    ReferenceImage, ReferenceMatchResult, ReferenceRegistry, ReferenceMatcher, ReferenceManager, CATEGORIES
)
from app.vision.player_detector import PlayerDetector, PlayerDetection
from app.vision.spider_detector import SpiderDetector, SpiderDetection
from app.vision.yellow_detector import YellowGlowDetector, YellowGlowDetectionResult
from app.vision.message_detector import MessageDetector, MessageDetection
from app.vision.status_detectors import StatusDetector, StatusDetectionResult, DrillState, BatteryState, MineLocationState
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.input.safety import safety


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_ref_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_ref_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_safety():
    safety.emergency_stopped = False
    yield
    safety.emergency_stopped = False


def test_reference_model_creation_and_serialization():
    ref = ReferenceImage(
        id="ref123",
        name="test_player",
        category="player",
        subcategory="right",
        file_path="reference/player/test_player.png",
        enabled=True,
        threshold=0.85,
        notes="Testing player image"
    )

    data = ref.to_dict()
    assert data["id"] == "ref123"
    assert data["name"] == "test_player"
    assert data["category"] == "player"
    assert data["threshold"] == 0.85

    restored = ReferenceImage.from_dict(data)
    assert restored.id == ref.id
    assert restored.name == ref.name
    assert restored.category == ref.category
    assert restored.threshold == ref.threshold


def test_reference_registry_directories_and_persistence(temp_ref_dir):
    registry = ReferenceRegistry(base_dir=temp_ref_dir)

    # Check directory structure creation
    assert os.path.exists(temp_ref_dir)
    for cat in CATEGORIES:
        assert os.path.exists(os.path.join(temp_ref_dir, cat))

    # Add a sample reference image
    sample_img = np.full((30, 30, 3), 150, dtype=np.uint8)
    ref = registry.add_reference(
        name="sample_rock",
        category="rock",
        subcategory="normal",
        source_file_or_image=sample_img,
        threshold=0.75,
        notes="Sample rock test"
    )

    assert ref is not None
    assert ref.name == "sample_rock"
    assert os.path.exists(ref.file_path)

    # Verify reloading from disk
    registry2 = ReferenceRegistry(base_dir=temp_ref_dir)
    assert len(registry2.get_all()) == 1
    loaded_ref = registry2.get_all()[0]
    assert loaded_ref.name == "sample_rock"
    assert loaded_ref.threshold == 0.75


def test_missing_reference_and_corrupt_file_handling(temp_ref_dir):
    registry = ReferenceRegistry(base_dir=temp_ref_dir)
    matcher = ReferenceMatcher()

    # 1. Missing File Reference
    missing_ref = ReferenceImage(
        id="missing1",
        name="non_existent",
        category="player",
        subcategory="left",
        file_path=os.path.join(temp_ref_dir, "player", "does_not_exist.png"),
        enabled=True
    )

    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = matcher.match_single(test_frame, missing_ref)
    assert res.found is False
    assert "missing" in res.error_message.lower()

    # 2. Corrupt File Reference
    corrupt_path = os.path.join(temp_ref_dir, "player", "corrupt.png")
    with open(corrupt_path, "w") as f:
        f.write("NOT_AN_IMAGE_FILE_DATA")

    corrupt_ref = ReferenceImage(
        id="corrupt1",
        name="corrupt_img",
        category="player",
        subcategory="left",
        file_path=corrupt_path,
        enabled=True
    )

    res_corrupt = matcher.match_single(test_frame, corrupt_ref)
    assert res_corrupt.found is False


def test_template_matcher_basic_match_and_threshold(temp_ref_dir):
    registry = ReferenceRegistry(base_dir=temp_ref_dir)
    matcher = ReferenceMatcher()

def test_template_matcher_basic_match_and_threshold(temp_ref_dir):
    registry = ReferenceRegistry(base_dir=temp_ref_dir)
    matcher = ReferenceMatcher()

    # Create synthetic frame with a distinct textured patch at (50, 40)
    rng = np.random.RandomState(42)
    frame = rng.randint(10, 40, (200, 200, 3), dtype=np.uint8)
    patch = rng.randint(150, 255, (20, 20, 3), dtype=np.uint8)
    frame[40:60, 50:70] = patch

    # Save patch as reference
    ref = registry.add_reference(
        name="red_patch",
        category="spider",
        subcategory="threat",
        source_file_or_image=patch,
        threshold=0.85
    )

    assert ref is not None

    # Match against target frame
    match_res = matcher.match_single(frame, ref)
    assert match_res.found is True
    assert match_res.confidence >= 0.85
    assert match_res.bbox == (50, 40, 20, 20)
    assert match_res.center == (60, 50)

    # Test threshold behavior: set threshold to 0.999 (should fail if noise present or lower)
    ref.threshold = 0.999
    match_fail = matcher.match_single(np.full((200, 200, 3), 50, dtype=np.uint8), ref)
    assert match_fail.found is False


def test_player_detector_with_reference_and_fallback(temp_ref_dir):
    registry = ReferenceRegistry(base_dir=temp_ref_dir)
    manager = ReferenceManager(base_dir=temp_ref_dir)
    detector = PlayerDetector()

    rng = np.random.RandomState(123)
    frame = rng.randint(10, 40, (200, 200, 3), dtype=np.uint8)
    player_patch = rng.randint(150, 255, (30, 30, 3), dtype=np.uint8)
    frame[80:110, 80:110] = player_patch

    # 1. Fallback behavior (no references registered)
    res_fallback = detector.detect(frame, reference_manager=manager)
    assert res_fallback.detected is True
    assert res_fallback.matched_reference_name == ""

    # 2. Reference match behavior
    registry.add_reference(
        name="player_front",
        category="player",
        subcategory="down",
        source_file_or_image=player_patch,
        threshold=0.80
    )
    # Reload manager registry
    manager.registry.load()

    res_ref = detector.detect(frame, reference_manager=manager)
    assert res_ref.detected is True
    assert res_ref.matched_reference_name == "player_front"
    assert res_ref.bbox == (80, 80, 30, 30)


def test_spider_detector_with_reference_and_spatial_dist(temp_ref_dir):
    manager = ReferenceManager(base_dir=temp_ref_dir)
    detector = SpiderDetector()

    rng = np.random.RandomState(456)
    frame = rng.randint(10, 40, (300, 300, 3), dtype=np.uint8)
    spider_patch = rng.randint(150, 255, (15, 15, 3), dtype=np.uint8)
    frame[50:65, 50:65] = spider_patch

    # Register spider reference
    manager.registry.add_reference(
        name="spider_model_1",
        category="spider",
        subcategory="default",
        source_file_or_image=spider_patch,
        threshold=0.75
    )

    player_center = (150, 150)
    _ = detector.detect(frame, player_center=player_center, reference_manager=manager)
    res = detector.detect(frame, player_center=player_center, reference_manager=manager)

    assert res.detected is True
    assert res.matched_reference_name == "spider_model_1"
    assert res.distance_from_player > 0.0


def test_yellow_rock_detector_with_reference_and_temporal_confirmation(temp_ref_dir):
    manager = ReferenceManager(base_dir=temp_ref_dir)
    detector = YellowGlowDetector(required_frames=2)

    rng = np.random.RandomState(789)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    yellow_patch = rng.randint(180, 255, (25, 25, 3), dtype=np.uint8)
    yellow_patch[:, :, 0] = 0   # B
    yellow_patch[:, :, 1] = 255 # G
    yellow_patch[:, :, 2] = 255 # R (Yellow)
    frame[60:85, 60:85] = yellow_patch

    manager.registry.add_reference(
        name="yellow_complete_rock",
        category="rock",
        subcategory="yellow_complete",
        source_file_or_image=yellow_patch,
        threshold=0.80
    )

    # Frame 1: Accumulating
    res1 = detector.detect(frame, reference_manager=manager)
    assert res1.detected_raw is True
    assert res1.is_confirmed is False
    assert res1.consecutive_frames == 1

    # Frame 2: Confirmed
    res2 = detector.detect(frame, reference_manager=manager)
    assert res2.detected_raw is True
    assert res2.is_confirmed is True
    assert res2.matched_reference_name == "yellow_complete_rock"


def test_status_detector_with_drill_battery_mine_references(temp_ref_dir):
    manager = ReferenceManager(base_dir=temp_ref_dir)
    detector = StatusDetector()

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bat_empty_patch = np.full((20, 20, 3), 50, dtype=np.uint8)
    bat_empty_patch[:, :] = [0, 0, 255]
    frame[10:30, 10:30] = bat_empty_patch

    manager.registry.add_reference(
        name="battery_empty_icon",
        category="status",
        subcategory="battery_empty",
        source_file_or_image=bat_empty_patch,
        threshold=0.80
    )

    res = detector.detect(frame, reference_manager=manager)
    assert res.battery_state == BatteryState.BATTERY_EMPTY
    assert res.matched_reference_name == "battery_empty_icon"


def test_mining_perception_engine_reference_integration(temp_ref_dir):
    engine = MiningPerceptionEngine(reference_dir=temp_ref_dir)
    frame = np.full((200, 200, 3), 80, dtype=np.uint8)

    res = engine.process_frame(frame)
    assert isinstance(res, MiningPerceptionResult)
    assert isinstance(res.reference_matches, list)


def test_dry_run_safety_remains_enforced():
    assert safety.dry_run is True
    assert safety.emergency_stopped is False
