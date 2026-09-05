"""Mining Target Memory Bank for Graal Mining Macro."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, List, Optional


class TargetState(Enum):
    AVAILABLE = "AVAILABLE"
    MINING = "MINING"
    COMPLETED = "COMPLETED"
    COOLDOWN = "COOLDOWN"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass
class MiningTarget:
    target_id: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    state: TargetState = TargetState.AVAILABLE
    iterations: int = 0
    max_iterations: int = 3
    is_yellow_completed: bool = False
    confidence: float = 0.0
    last_seen: float = field(default_factory=time.time)
    last_mined: float = 0.0
    cooldown_until: float = 0.0

    @property
    def is_finished(self) -> bool:
        return self.is_yellow_completed or self.iterations >= self.max_iterations

    @property
    def is_on_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def mark_cooldown(self, duration_seconds: float = 10.0) -> None:
        self.cooldown_until = time.time() + duration_seconds
        self.state = TargetState.COOLDOWN

    def increment_iteration(self) -> int:
        self.iterations = min(self.max_iterations, self.iterations + 1)
        self.last_mined = time.time()
        return self.iterations


class TargetMemoryBank:
    """Stores and tracks recently observed mining targets."""

    def __init__(self, match_tolerance_px: int = 30):
        self.targets: Dict[str, MiningTarget] = {}
        self.match_tolerance_px = match_tolerance_px

    def get_or_create(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], confidence: float) -> MiningTarget:
        cx, cy = center
        now = time.time()

        for t_id, target in self.targets.items():
            tx, ty = target.center
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist <= self.match_tolerance_px:
                target.center = center
                target.bbox = bbox
                target.confidence = confidence
                target.last_seen = now
                if target.is_on_cooldown and now >= target.cooldown_until:
                    target.state = TargetState.AVAILABLE
                return target

        new_id = f"rock_{len(self.targets) + 1}_{int(now)}"
        new_target = MiningTarget(
            target_id=new_id,
            center=center,
            bbox=bbox,
            confidence=confidence,
            last_seen=now,
        )
        self.targets[new_id] = new_target
        return new_target

    def get_available_targets(self) -> List[MiningTarget]:
        return [
            t for t in self.targets.values()
            if t.state == TargetState.AVAILABLE and not t.is_finished and not t.is_on_cooldown
        ]

    def clear(self) -> None:
        self.targets.clear()
