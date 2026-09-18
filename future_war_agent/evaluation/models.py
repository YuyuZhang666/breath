from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from future_war_agent.protocol.models import Observation


class MatchOutcome(StrEnum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"
    UNKNOWN = "unknown"

    @property
    def league_points(self) -> int:
        return {
            MatchOutcome.WIN: 3,
            MatchOutcome.DRAW: 1,
            MatchOutcome.LOSS: 0,
            MatchOutcome.UNKNOWN: 0,
        }[self]


@dataclass(frozen=True, slots=True)
class ReplayCase:
    name: str
    outcome: MatchOutcome
    turns: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class ReplayCorpus:
    cases: tuple[ReplayCase, ...]


PlannerFactory = Callable[[], object]
ProfileReader = Callable[[object, Observation], object | None]


@dataclass(frozen=True, slots=True)
class ReplayVariant:
    name: str
    planner_factory: PlannerFactory
    profile_reader: ProfileReader | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variant name must be nonblank")
        if not callable(self.planner_factory):
            raise TypeError("planner_factory must be callable")
        if self.profile_reader is not None and not callable(self.profile_reader):
            raise TypeError("profile_reader must be callable")


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    league_points: int
    initial_score: int
    final_score: int
    score_gain: int
    final_station_survived: bool
    minimum_station_health: int
    command_count: int
    prompt_count: int
    execute_command_count: int
    task_accept_count: int
    answer_submit_count: int
    action_kind_counts: tuple[tuple[str, int], ...]
    profile_switch_count: int
    median_latency_ns: int
    p99_latency_ns: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    variant_name: str
    case_name: str
    outcome: MatchOutcome
    response_digests: tuple[str, ...]
    profile_labels: tuple[str | None, ...]
    elapsed_ns: tuple[int, ...]
    metrics: ReplayMetrics
