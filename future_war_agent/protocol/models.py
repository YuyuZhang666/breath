from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .time import TurnTime


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def chebyshev_distance(self, other: "Position") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))


@dataclass(frozen=True, slots=True)
class Zone:
    position: Position
    neutral_type: str


@dataclass(frozen=True, slots=True)
class UnitState:
    unit_id: int
    position: Position
    role_type: str
    health: int
    attack_power: int = 0
    attack_range: int = 0
    backpack_capacity: int = 0
    backpack: tuple[str, ...] = ()
    level: int | None = None
    cooldown: int = 0
    provided_fields: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RobotState:
    robot_id: int
    position: Position
    role_type: str
    health: int
    abnormal_state: str = ""
    target_team: str | None = None


@dataclass(frozen=True, slots=True)
class TaskPointState:
    task_type: str
    position: Position
    cooldown_rounds: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int | None = None


@dataclass(frozen=True, slots=True)
class OurTeamState:
    team_type: str = ""
    team_id: str = ""
    team_name: str = ""
    gold: int = 0
    total_score: int = 0
    tasks: tuple[TaskPointState, ...] = ()
    units: tuple[UnitState, ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyTeamState:
    units: tuple[UnitState, ...] = ()


@dataclass(frozen=True, slots=True)
class WorldNews:
    official_news: str = ""
    folk_legends: str = ""


@dataclass(frozen=True, slots=True)
class ShopItem:
    name: str
    price: int


@dataclass(frozen=True, slots=True)
class GameError:
    error_code: int
    description: str


@dataclass(frozen=True, slots=True)
class Observation:
    time: TurnTime
    width: int
    height: int
    zones: tuple[Zone, ...]
    our: OurTeamState
    enemy: EnemyTeamState
    robots: tuple[RobotState, ...]
    phase_task: str
    last_action_results: Mapping[int, bool] = field(default_factory=dict)
    last_summon_treasure_result: int = 0
    llm_response: str = ""
    world_news: WorldNews = field(default_factory=WorldNews)
    last_command_result: str = ""
    vendor_shop: tuple[ShopItem, ...] = ()
    weapon_shop: tuple[ShopItem, ...] = ()
    errors: tuple[GameError, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "last_action_results",
            MappingProxyType(dict(self.last_action_results)),
        )
