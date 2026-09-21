from dataclasses import dataclass
from enum import StrEnum
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


class Phase3Level(StrEnum):
    NONE = "none"
    LITE = "lite"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class Phase3Budget:
    root_candidates: int
    scenarios: int
    exact_horizon: int
    seconds: float

    def __post_init__(self) -> None:
        if self.root_candidates <= 0:
            raise ValueError("root_candidates must be positive")
        if self.scenarios <= 0:
            raise ValueError("scenarios must be positive")
        if self.exact_horizon <= 0:
            raise ValueError("exact_horizon must be positive")
        if self.seconds <= 0:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True, slots=True)
class Phase3Config:
    max_root_actions: int = 8
    scenario_count: int = 2
    max_horizon: int = 4
    watchdog_seconds: float = 0.250
    lite_root_actions: int = 4
    lite_scenario_count: int = 1
    lite_horizon: int = 2
    lite_watchdog_seconds: float = 0.080
    scenario_weight_floor: Fraction = Fraction(1, 20)
    gatling_damage: int = 10
    rocket_center_damage: int = 20
    rocket_splash_damage: int = 10
    rocket_cooldown_rounds: int = 3
    max_weapon_candidates: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.max_root_actions <= 8:
            raise ValueError("Phase 3 Full root cap must be between one and eight")
        if not 1 <= self.scenario_count <= 2:
            raise ValueError("Phase 3 Full requires one or two scenarios")
        if not 1 <= self.max_horizon <= 4:
            raise ValueError("max_horizon must be between one and four")
        if self.watchdog_seconds <= 0:
            raise ValueError("watchdog_seconds must be positive")
        if not 1 <= self.lite_root_actions <= 4:
            raise ValueError("Phase 3 Lite root cap must be between one and four")
        if self.lite_scenario_count != 1:
            raise ValueError("Phase 3 Lite requires exactly one scenario")
        if not 1 <= self.lite_horizon <= 2:
            raise ValueError("lite_horizon must be between one and two")
        if self.lite_watchdog_seconds <= 0:
            raise ValueError("lite_watchdog_seconds must be positive")

    def budget_for(self, level: Phase3Level) -> Phase3Budget:
        if level is Phase3Level.LITE:
            return Phase3Budget(
                root_candidates=self.lite_root_actions,
                scenarios=self.lite_scenario_count,
                exact_horizon=self.lite_horizon,
                seconds=self.lite_watchdog_seconds,
            )
        if level is Phase3Level.FULL:
            return Phase3Budget(
                root_candidates=self.max_root_actions,
                scenarios=self.scenario_count,
                exact_horizon=self.max_horizon,
                seconds=self.watchdog_seconds,
            )
        raise ValueError("Phase 3 NONE has no search budget")


DEFAULT_PHASE3_CONFIG = Phase3Config()
