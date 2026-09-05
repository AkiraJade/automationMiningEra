"""Window Detection Data Models for Graal Mining Macro."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    pid: int = 0
    process_name: str = ""
    outer_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (left, top, right, bottom)
    client_rect: Tuple[int, int, int, int] = (0, 0, 0, 0) # (left, top, right, bottom)
    client_width: int = 0
    client_height: int = 0
    is_visible: bool = False
    is_minimized: bool = False

    @property
    def is_valid(self) -> bool:
        return (
            self.hwnd > 0
            and self.is_visible
            and not self.is_minimized
            and self.client_width > 50
            and self.client_height > 50
        )

    def __str__(self) -> str:
        return (
            f"WindowInfo(HWND={self.hwnd}, Title='{self.title}', PID={self.pid}, "
            f"Process='{self.process_name}', Res={self.client_width}x{self.client_height}, Valid={self.is_valid})"
        )


@dataclass
class CandidateWindowInfo:
    hwnd: int
    title: str
    pid: int
    process_name: str
    client_width: int
    client_height: int
    is_visible: bool
    is_minimized: bool
    is_own_process: bool
    score: int
    rejection_reason: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        return self.rejection_reason is None
