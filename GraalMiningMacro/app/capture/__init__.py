"""Screen capture package for Graal Mining Macro."""
from app.capture.capture_target import CaptureTarget
from app.capture.base_backend import BaseCaptureBackend
from app.capture.window_graphics_capture import WindowGraphicsCaptureBackend
from app.capture.screen_region_capture import ScreenRegionCaptureBackend
from app.capture.screen_capture import ScreenCaptureEngine
from app.capture.worker import CaptureWorkerThread

__all__ = [
    "CaptureTarget",
    "BaseCaptureBackend",
    "WindowGraphicsCaptureBackend",
    "ScreenRegionCaptureBackend",
    "ScreenCaptureEngine",
    "CaptureWorkerThread",
]
