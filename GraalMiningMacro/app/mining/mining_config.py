"""Mining Configuration helper structures."""

from dataclasses import dataclass, field


@dataclass
class LevelTiming:
    mining_level: int
    attack_interval: float
    expected_iteration_time: float


MINING_LEVEL_TABLE = {
    0: LevelTiming(mining_level=0, attack_interval=0.40, expected_iteration_time=1.20),
    1: LevelTiming(mining_level=1, attack_interval=0.38, expected_iteration_time=1.14),
    2: LevelTiming(mining_level=2, attack_interval=0.36, expected_iteration_time=1.08),
    3: LevelTiming(mining_level=3, attack_interval=0.34, expected_iteration_time=1.02),
}


def get_level_timing(level: int) -> LevelTiming:
    return MINING_LEVEL_TABLE.get(level, MINING_LEVEL_TABLE[0])
