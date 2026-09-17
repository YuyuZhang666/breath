from dataclasses import dataclass

from future_war_agent.protocol.models import Position


@dataclass(frozen=True, slots=True)
class RulesConfig:
    weapon_build_cost: int = 25
    wall_material: str = "stone"
    wall_material_cost: int = 1
    weapon_loadout: tuple[str, str, str] = ("gatling", "railgun", "rocket")
    twilight_safety_margin: int = 2
    station_y_direction: int = 1

    def __post_init__(self) -> None:
        if self.station_y_direction not in {-1, 1}:
            raise ValueError("station_y_direction must be -1 or 1")


DEFAULT_RULES = RulesConfig()


def station_footprint(
    position: Position,
    rules: RulesConfig = DEFAULT_RULES,
) -> frozenset[Position]:
    vertical = rules.station_y_direction
    return frozenset(
        {
            position,
            Position(position.x + 1, position.y),
            Position(position.x, position.y + vertical),
            Position(position.x + 1, position.y + vertical),
        }
    )
