from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NightObjective:
    minimum_station_health: int = 1
    protect_all_controllers: bool = True
    protect_all_weapons: bool = True
    require_post_horizon_buffer: bool = True
    enable_score_band: bool = True

    def __post_init__(self) -> None:
        if self.minimum_station_health < 1:
            raise ValueError("minimum_station_health must be positive")


DEFAULT_NIGHT_OBJECTIVE = NightObjective()
