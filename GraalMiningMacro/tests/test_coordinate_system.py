"""Unit tests for Coordinate System module."""

import pytest
from app.window.models import WindowInfo
from app.coordinates.coordinate_system import CoordinateSystem


@pytest.fixture
def sample_window_info():
    return WindowInfo(
        hwnd=100,
        title="Test Window",
        outer_rect=(100, 100, 900, 700),
        client_rect=(100, 140, 900, 740),
        client_width=800,
        client_height=600,
        is_visible=True,
        is_minimized=False,
    )


def test_client_to_screen(sample_window_info):
    screen_x, screen_y = CoordinateSystem.client_to_screen(50, 50, sample_window_info)
    assert screen_x == 150
    assert screen_y == 190


def test_screen_to_client(sample_window_info):
    client_x, client_y = CoordinateSystem.screen_to_client(150, 190, sample_window_info)
    assert client_x == 50
    assert client_y == 50


def test_normalized_conversions(sample_window_info):
    cx, cy = CoordinateSystem.normalized_to_client(0.5, 0.5, sample_window_info)
    assert cx == 400
    assert cy == 300

    nx, ny = CoordinateSystem.client_to_normalized(400, 300, sample_window_info)
    assert nx == 0.5
    assert ny == 0.5
