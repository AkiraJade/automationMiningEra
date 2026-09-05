"""Unit tests for Window Detector with focus-independent target lifetime locking and Era.exe recognition."""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.window.models import WindowInfo, CandidateWindowInfo
from app.window.detector import WindowDetector
from app.capture.screen_capture import ScreenCaptureEngine


def test_window_info_validity():
    valid_info = WindowInfo(
        hwnd=12345,
        title="GraalOnline Era",
        pid=9999,
        process_name="Era.exe",
        outer_rect=(100, 100, 900, 700),
        client_rect=(105, 130, 895, 695),
        client_width=790,
        client_height=565,
        is_visible=True,
        is_minimized=False,
    )
    assert valid_info.is_valid is True


def test_era_exe_and_graalonline_era_recognition():
    detector = WindowDetector(title_pattern="GraalOnline Era", target_executable="Era.exe")

    cand_era_game = CandidateWindowInfo(
        hwnd=5001,
        title="GraalOnline Era",
        pid=7777,
        process_name="Era.exe",
        client_width=1280,
        client_height=720,
        is_visible=True,
        is_minimized=False,
        is_own_process=False,
        score=850,
        rejection_reason=None,
    )

    fake_selected_info = WindowInfo(
        hwnd=5001,
        title="GraalOnline Era",
        pid=7777,
        process_name="Era.exe",
        outer_rect=(0, 0, 1280, 720),
        client_rect=(0, 0, 1280, 720),
        client_width=1280,
        client_height=720,
        is_visible=True,
        is_minimized=False,
    )

    with patch.object(detector, "list_candidate_windows", return_value=[cand_era_game]):
        with patch.object(detector, "get_window_info_by_hwnd", return_value=fake_selected_info):
            selected = detector.find_window()
            assert selected is not None
            assert selected.hwnd == 5001
            assert selected.process_name == "Era.exe"
            assert selected.title == "GraalOnline Era"
            assert detector.target_hwnd == 5001


def test_target_hwnd_persists_when_macro_focused():
    """Verifies existing Era HWND remains locked as target even if macro application gains foreground focus."""
    detector = WindowDetector(title_pattern="GraalOnline Era", target_executable="Era.exe")
    detector._target_hwnd = 5001

    fake_era_info = WindowInfo(
        hwnd=5001,
        title="GraalOnline Era",
        pid=7777,
        process_name="Era.exe",
        outer_rect=(0, 0, 1280, 720),
        client_rect=(0, 0, 1280, 720),
        client_width=1280,
        client_height=720,
        is_visible=True,
        is_minimized=False,
    )

    with patch.object(detector, "get_window_info_by_hwnd", return_value=fake_era_info):
        # Even if candidate search would run or focus changed, refresh() MUST lock onto target_hwnd 5001
        refreshed = detector.refresh()
        assert refreshed is not None
        assert refreshed.hwnd == 5001
        assert detector.target_hwnd == 5001


def test_re_detection_triggers_only_on_target_invalidation():
    """Verifies candidate re-detection runs ONLY when locked target HWND closes or becomes invalid."""
    detector = WindowDetector(title_pattern="GraalOnline Era", target_executable="Era.exe")
    detector._target_hwnd = 5001

    # Simulate get_window_info_by_hwnd returning None (target HWND closed)
    with patch.object(detector, "get_window_info_by_hwnd", return_value=None):
        with patch.object(detector, "find_window", return_value=None) as mock_find:
            refreshed = detector.refresh()
            assert refreshed is None
            assert detector.target_hwnd is None
            mock_find.assert_called_once()


def test_own_process_and_macro_title_rejection():
    detector = WindowDetector(title_pattern="GraalOnline Era")
    own_pid = os.getpid()

    candidates = [
        CandidateWindowInfo(
            hwnd=1001,
            title="Graal Mining Macro",
            pid=own_pid,
            process_name="python.exe",
            client_width=1280,
            client_height=800,
            is_visible=True,
            is_minimized=False,
            is_own_process=True,
            score=0,
            rejection_reason="Excluded: Belongs to own macro process ID",
        ),
    ]

    with patch.object(detector, "list_candidate_windows", return_value=candidates):
        result = detector.find_window()
        assert result is None, "WindowDetector MUST return None (DISCONNECTED) when only macro windows exist!"


def test_no_matching_game_produces_disconnected():
    detector = WindowDetector(title_pattern="NonExistentWindow9999", target_executable="NonExistent.exe")
    with patch.object(detector, "list_candidate_windows", return_value=[]):
        result = detector.find_window()
        assert result is None


def test_capture_engine_hard_safety_guard():
    own_pid = os.getpid()
    engine = ScreenCaptureEngine()

    macro_own_window = WindowInfo(
        hwnd=99999,
        title="Graal Mining Macro",
        pid=own_pid,
        process_name="python.exe",
        outer_rect=(10, 10, 500, 500),
        client_rect=(10, 10, 500, 500),
        client_width=490,
        client_height=490,
        is_visible=True,
        is_minimized=False,
    )

    frame = engine.capture_window_frame(macro_own_window)
    assert frame is None, "ScreenCaptureEngine MUST return None (refuse capture) when given macro window!"
    engine.close()
