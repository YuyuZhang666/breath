from dataclasses import dataclass
from enum import StrEnum


DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS


class Phase(StrEnum):
    DAY = "day"
    NIGHT = "night"


@dataclass(frozen=True, slots=True)
class TurnTime:
    round_no: int
    day_no: int
    phase: Phase
    round_in_phase: int
    offset_in_day: int

    @classmethod
    def from_round(cls, round_no: int) -> "TurnTime":
        if isinstance(round_no, bool) or not isinstance(round_no, int):
            raise TypeError("round number must be an integer")
        if round_no < 1:
            raise ValueError("round number must be positive")

        zero_based = round_no - 1
        day_no = zero_based // ROUNDS_PER_DAY + 1
        offset_in_day = zero_based % ROUNDS_PER_DAY

        if offset_in_day < DAY_ROUNDS:
            phase = Phase.DAY
            round_in_phase = offset_in_day + 1
        else:
            phase = Phase.NIGHT
            round_in_phase = offset_in_day - DAY_ROUNDS + 1

        return cls(
            round_no=round_no,
            day_no=day_no,
            phase=phase,
            round_in_phase=round_in_phase,
            offset_in_day=offset_in_day,
        )
