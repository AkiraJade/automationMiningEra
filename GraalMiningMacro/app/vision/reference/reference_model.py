"""Reference Data Models for Graal Mining Macro."""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, List

CATEGORIES: List[str] = [
    "player",
    "spider",
    "rock",
    "status",
    "messages",
    "mine"
]

SUBCATEGORIES: Dict[str, List[str]] = {
    "player": ["left", "right", "up", "down", "mining", "custom"],
    "spider": ["default", "threat", "custom"],
    "rock": ["normal", "yellow_complete", "custom"],
    "status": ["drill_equipped", "drill_unequipped", "battery_empty", "custom"],
    "messages": ["nothing_to_mine", "custom"],
    "mine": ["inside", "surface", "entrance", "collapsed", "custom"],
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "player": 0.80,
    "spider": 0.75,
    "rock": 0.75,
    "status": 0.80,
    "messages": 0.80,
    "mine": 0.75,
}


@dataclass
class ReferenceImage:
    """Represents a reference image metadata item registered in the reference library."""

    id: str
    name: str
    category: str
    subcategory: str
    file_path: str
    enabled: bool = True
    threshold: float = 0.80
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceImage":
        cat = data.get("category", "player")
        default_thresh = DEFAULT_THRESHOLDS.get(cat, 0.80)
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "Unnamed Reference")),
            category=str(cat),
            subcategory=str(data.get("subcategory", "custom")),
            file_path=str(data.get("file_path", "")),
            enabled=bool(data.get("enabled", True)),
            threshold=float(data.get("threshold", default_thresh)),
            created_at=float(data.get("created_at", time.time())),
            notes=str(data.get("notes", "")),
        )


@dataclass
class ReferenceMatchResult:
    """Result of template matching evaluation against a reference image."""

    found: bool = False
    reference_id: str = ""
    reference_name: str = ""
    category: str = ""
    subcategory: str = ""
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None  # (x, y)
    error_message: str = ""

    def summary_text(self) -> str:
        if not self.found:
            return f"REF [{self.category.upper()}]: NO MATCH"
        return f"REF {self.reference_name} ({self.confidence * 100:.0f}%) BBOX:{self.bbox}"
