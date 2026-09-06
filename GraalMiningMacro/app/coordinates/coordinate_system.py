"""Coordinate Transformation Utility for Graal Mining Macro."""

from typing import Tuple
from app.window.models import WindowInfo


class CoordinateSystem:
    """Handles coordinate transformations between Screen, Client, and Normalized spaces."""

    @staticmethod
    def client_to_screen(client_x: int, client_y: int, window_info: WindowInfo) -> Tuple[int, int]:
        """Converts client-relative coordinate (x, y) to absolute screen (x, y)."""
        screen_left, screen_top, _, _ = window_info.client_rect
        return (screen_left + client_x, screen_top + client_y)

    @staticmethod
    def screen_to_client(screen_x: int, screen_y: int, window_info: WindowInfo) -> Tuple[int, int]:
        """Converts absolute screen coordinate (x, y) to client-relative (x, y)."""
        screen_left, screen_top, _, _ = window_info.client_rect
        return (screen_x - screen_left, screen_y - screen_top)

    @staticmethod
    def normalized_to_client(norm_x: float, norm_y: float, window_info: WindowInfo) -> Tuple[int, int]:
        """Converts normalized (0.0 to 1.0) coordinate to client pixel (x, y)."""
        client_x = int(norm_x * window_info.client_width)
        client_y = int(norm_y * window_info.client_height)
        return (client_x, client_y)

    @staticmethod
    def client_to_normalized(client_x: int, client_y: int, window_info: WindowInfo) -> Tuple[float, float]:
        """Converts client pixel (x, y) to normalized (0.0 to 1.0) coordinate."""
        if window_info.client_width <= 0 or window_info.client_height <= 0:
            return (0.0, 0.0)
        return (client_x / window_info.client_width, client_y / window_info.client_height)


class ViewportGeometry:
    """Detects and isolates the actual 2D game canvas viewport boundaries within captured frames."""

    @staticmethod
    def extract_viewport_rect(frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h) bounding box of playable game area, stripping title bar and outer padding."""
        if frame_width <= 0 or frame_height <= 0:
            return (0, 0, 100, 100)
        
        # GraalOnline Era game viewport occupies central region below top HUD bar
        top_margin = int(frame_height * 0.05)
        left_margin = int(frame_width * 0.02)
        viewport_w = int(frame_width * 0.96)
        viewport_h = int(frame_height * 0.90)

        return (left_margin, top_margin, viewport_w, viewport_h)

