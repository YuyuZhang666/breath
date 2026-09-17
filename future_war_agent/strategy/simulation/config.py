from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RobotSpec:
    attack_power: int
    attack_range: int
    max_health: int
    kill_score: int


ROBOT_SPECS: Mapping[str, RobotSpec] = MappingProxyType(
    {
        "smallRobot": RobotSpec(5, 3, 40, 1),
        "middleRobot": RobotSpec(10, 3, 60, 2),
        "largeRobot": RobotSpec(20, 3, 500, 4),
        "bossRobot": RobotSpec(40, 3, 800, 10),
    }
)


@dataclass(frozen=True, slots=True)
class Phase3Config:
    max_root_actions: int = 64
    scenario_count: int = 4
    max_horizon: int = 6
    watchdog_seconds: float = 0.8
    scenario_weight_floor: Fraction = Fraction(1, 20)
    gatling_damage: int = 10
    rocket_center_damage: int = 20
    rocket_splash_damage: int = 10
    rocket_cooldown_rounds: int = 3
    max_weapon_candidates: int = 3

    def __post_init__(self) -> None:
        if self.max_root_actions != 64:
            raise ValueError("Phase 3 root cap must remain 64")
        if self.scenario_count != 4:
            raise ValueError("Phase 3 requires exactly four scenarios")
        if not 1 <= self.max_horizon <= 6:
            raise ValueError("max_horizon must be between one and six")
        if self.watchdog_seconds <= 0:
            raise ValueError("watchdog_seconds must be positive")


DEFAULT_PHASE3_CONFIG = Phase3Config()
