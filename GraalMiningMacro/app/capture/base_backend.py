"""Abstract Base Capture Backend for Graal Mining Macro."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from app.capture.capture_target import CaptureTarget


class BaseCaptureBackend(ABC):
    """Abstract Base Class for Screen / HWND Capture Engines."""

    @abstractmethod
    def capture_target_frame(self, target: CaptureTarget) -> Optional[np.ndarray]:
        """Captures frame from the specified target as an OpenCV BGR numpy array."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleans up resources."""
        pass
