"""Global Configuration Module for Graal Mining Macro."""

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Optional


class AutomationLevel(IntEnum):
    OBSERVE = 1
    RECOMMEND = 2
    MOVEMENT = 3
    MINING = 4
    RECOVERY = 5
    FULL_MINING = 6


@dataclass
class WindowConfig:
    title_pattern: str = "GraalOnline Era"
    target_executable: str = "Era.exe"
    auto_detect: bool = True
    match_exact: bool = False
    custom_hwnd: Optional[int] = None


@dataclass
class CaptureConfig:
    fps: int = 30
    preview_scale: float = 1.0
    aspect_ratio_preserve: bool = True


@dataclass
class KeyConfig:
    drill_equip_key: str = "t"
    spider_combat_key: str = "s"
    attack_key: str = "space"
    emergency_stop_key: str = "f12"


@dataclass
class TimingConfig:
    mining_action_interval: float = 0.4
    verification_timeout: float = 3.0
    nothing_to_mine_cooldown: float = 10.0
    spider_recheck_interval: float = 0.2
    movement_pulse_duration: float = 0.15
    recovery_timeout: float = 15.0


@dataclass
class VisionConfig:
    perception_fps: int = 10
    player_confidence_threshold: float = 0.65
    target_confidence_threshold: float = 0.60
    spider_confidence_threshold: float = 0.70
    yellow_glow_hsv_min: list[int] = field(default_factory=lambda: [15, 120, 150])
    yellow_glow_hsv_max: list[int] = field(default_factory=lambda: [35, 255, 255])
    yolo_model_path: str = "models/graal_mining.pt"
    yolo_device: str = "cpu"
    debug_overlay_enabled: bool = True

    # Deliverable 4 Accuracy & Performance Tuning Parameters
    min_scale: float = 0.90
    max_scale: float = 1.10
    scale_step: float = 0.05
    minimum_match_confidence: float = 0.65
    minimum_match_size: list[int] = field(default_factory=lambda: [12, 12])
    maximum_match_size: list[int] = field(default_factory=lambda: [150, 150])
    spider_confirm_frames: int = 2
    spider_max_distance: float = 800.0
    yellow_confirm_frames: int = 3
    playable_world_roi: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.90, 0.85])


@dataclass
class SafetyConfig:
    dry_run: bool = True
    automation_level: int = AutomationLevel.OBSERVE
    max_retries: int = 3
    action_rate_limit: float = 0.1
    pause_on_window_lost: bool = True


@dataclass
class AppConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    keys: KeyConfig = field(default_factory=KeyConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mining_level: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        window = WindowConfig(**data.get("window", {}))
        capture = CaptureConfig(**data.get("capture", {}))
        keys = KeyConfig(**data.get("keys", {}))
        timing = TimingConfig(**data.get("timing", {}))
        vision = VisionConfig(**data.get("vision", {}))
        safety = SafetyConfig(**data.get("safety", {}))
        mining_level = data.get("mining_level", 0)
        return cls(
            window=window,
            capture=capture,
            keys=keys,
            timing=timing,
            vision=vision,
            safety=safety,
            mining_level=mining_level,
        )

    def save_to_file(self, filepath: str = "config.json") -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str = "config.json") -> "AppConfig":
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception:
                pass
        return cls()
