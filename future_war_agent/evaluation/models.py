from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


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
