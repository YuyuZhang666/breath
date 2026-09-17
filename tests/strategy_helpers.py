from collections.abc import Iterable

from future_war_agent.protocol.models import (
    EnemyTeamState,
    Observation,
    OurTeamState,
    Position,
    RobotState,
    ShopItem,
    UnitState,
    Zone,
)
from future_war_agent.protocol.time import TurnTime


def unit(
    unit_id: int,
    x: int,
    y: int,
    role_type: str,
    *,
    health: int = 100,
    attack_range: int = 0,
    backpack_capacity: int = 100,
    backpack: Iterable[str] = (),
    level: int | None = None,
    cooldown: int = 0,
) -> UnitState:
    return UnitState(
        unit_id=unit_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        attack_range=attack_range,
        backpack_capacity=backpack_capacity,
        backpack=tuple(backpack),
        level=level,
        cooldown=cooldown,
    )


def robot(
    robot_id: int,
    x: int,
    y: int,
    *,
    role_type: str = "smallRobot",
    health: int = 40,
    target_team: str | None = "challenger",
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        target_team=target_team,
    )


def observation(
    *,
    round_no: int = 1,
    width: int = 15,
    height: int = 15,
    our_units: Iterable[UnitState] = (),
    enemy_units: Iterable[UnitState] = (),
    robots: Iterable[RobotState] = (),
    zones: Iterable[Zone] = (),
    gold: int = 75,
    vendor_shop: Iterable[ShopItem] = (),
) -> Observation:
    return Observation(
        time=TurnTime.from_round(round_no),
        width=width,
        height=height,
        zones=tuple(zones),
        our=OurTeamState(
            team_type="challenger",
            team_id="team",
            gold=gold,
            units=tuple(our_units),
        ),
        enemy=EnemyTeamState(units=tuple(enemy_units)),
        robots=tuple(robots),
        phase_task="",
        vendor_shop=tuple(vendor_shop),
    )
