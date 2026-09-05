"""QThread Background Capture Worker for Graal Mining Macro."""

import time
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from app.window.detector import WindowDetector
from app.window.models import WindowInfo
from app.capture.screen_capture import ScreenCaptureEngine
from app.core.events import events
from app.core.logger import setup_logger

logger = setup_logger("CaptureWorkerThread")


class CaptureWorkerThread(QThread):
    """Background worker thread capturing game window at target FPS."""

    frame_captured_signal = Signal(object, object)  # (np.ndarray frame, WindowInfo win_info)
    status_changed_signal = Signal(str, object)      # (status_str, WindowInfo or None)

    def __init__(self, detector: WindowDetector, target_fps: int = 30):
        super().__init__()
        self.detector = detector
        self.target_fps = max(1, min(60, target_fps))
        self._running = False
        self._mutex = QMutex()
        self._engine = ScreenCaptureEngine()

    def set_target_fps(self, fps: int) -> None:
        with QMutexLocker(self._mutex):
            self.target_fps = max(1, min(60, fps))

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)

    def run(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = True

        logger.info("Capture worker thread started.")
        frame_count = 0
        last_fps_calc = time.time()
        current_fps = 0.0

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                fps = self.target_fps

            loop_start = time.time()
            win_info = self.detector.refresh()

            if win_info and win_info.is_valid:
                frame = self._engine.capture_window_frame(win_info)
                if frame is not None:
                    frame_count += 1
                    self.frame_captured_signal.emit(frame, win_info)
                    events.frame_captured.emit(frame)
                else:
                    self.status_changed_signal.emit("Capture error / empty frame", win_info)
            else:
                self.status_changed_signal.emit("Window not found or minimized", win_info)
                events.window_lost.emit()

            # Calculate FPS every 1 second
            now = time.time()
            elapsed = now - last_fps_calc
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                events.capture_fps_updated.emit(current_fps)
                frame_count = 0
                last_fps_calc = now

            # Sleep to match target FPS
            target_frame_time = 1.0 / fps
            process_time = time.time() - loop_start
            sleep_time = target_frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._engine.close()
        logger.info("Capture worker thread stopped.")
