"""Unit tests for Screen Capture abstraction and WindowGraphicsCaptureBackend."""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.capture.capture_target import CaptureTarget
from app.capture.window_graphics_capture import WindowGraphicsCaptureBackend
from app.capture.screen_capture import ScreenCaptureEngine
from app.window.models import WindowInfo


def test_capture_target_creation():
    win_info = WindowInfo(
        hwnd=12345,
        title="GraalOnline Era",
        pid=7777,
        process_name="Era.exe",
        outer_rect=(10, 10, 800, 600),
        client_rect=(10, 10, 800, 600),
        client_width=790,
        client_height=590,
        is_visible=True,
        is_minimized=False,
    )

    target = CaptureTarget.from_window_info(win_info)
    assert target is not None
    assert target.hwnd == 12345
    assert target.pid == 7777
    assert target.process_name == "Era.exe"
    assert target.title == "GraalOnline Era"
    assert target.width == 790
    assert target.height == 590
    assert target.is_valid is True
    assert target.is_minimized is False


def test_bgr_to_qimage_conversion():
    fake_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_bgr[:, :, 0] = 255  # Blue channel

    qimg = ScreenCaptureEngine.bgr_to_qimage(fake_bgr)
    assert qimg is not None
    assert qimg.width() == 100
    assert qimg.height() == 100


def test_window_graphics_capture_safety_rejection():
    backend = WindowGraphicsCaptureBackend()
    own_pid = os.getpid()

    own_macro_target = CaptureTarget(
        hwnd=9999,
        pid=own_pid,
        process_name="python.exe",
        title="Graal Mining Macro",
        client_rect=(0, 0, 800, 600),
        width=800,
        height=600,
        is_valid=True,
        is_minimized=False,
    )

    frame = backend.capture_target_frame(own_macro_target)
    assert frame is None, "WindowGraphicsCaptureBackend MUST refuse to capture own macro process ID!"


def test_window_graphics_capture_minimized_window():
    backend = WindowGraphicsCaptureBackend()

    minimized_target = CaptureTarget(
        hwnd=8888,
        pid=5555,
        process_name="Era.exe",
        title="GraalOnline Era",
        client_rect=(0, 0, 0, 0),
        width=0,
        height=0,
        is_valid=False,
        is_minimized=True,
    )

    frame = backend.capture_target_frame(minimized_target)
    assert frame is None, "WindowGraphicsCaptureBackend MUST return None (CAPTURE UNAVAILABLE) for minimized windows!"


def test_capture_invalid_window():
    engine = ScreenCaptureEngine()
    invalid_win = WindowInfo(
        hwnd=0,
        title="",
        outer_rect=(0, 0, 0, 0),
        client_rect=(0, 0, 0, 0),
        client_width=0,
        client_height=0,
        is_visible=False,
        is_minimized=True,
    )
    frame = engine.capture_window_frame(invalid_win)
    assert frame is None
    engine.close()
