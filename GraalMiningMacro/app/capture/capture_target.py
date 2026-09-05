"""Capture Target Data Model for Graal Mining Macro."""

from dataclasses import dataclass
from typing import Tuple, Optional
from app.window.models import WindowInfo


@dataclass
class CaptureTarget:
    hwnd: int
    pid: int
    process_name: str
    title: str
    client_rect: Tuple[int, int, int, int]
    width: int
    height: int
    is_valid: bool
    is_minimized: bool

    @classmethod
    def from_window_info(cls, info: Optional[WindowInfo]) -> Optional["CaptureTarget"]:
        if not info:
            return None
        return cls(
            hwnd=info.hwnd,
            pid=info.pid,
            process_name=info.process_name,
            title=info.title,
            client_rect=info.client_rect,
            width=info.client_width,
            height=info.client_height,
            is_valid=info.is_valid,
            is_minimized=info.is_minimized,
        )
