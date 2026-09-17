from collections.abc import Iterable, Mapping

from future_war_agent.protocol.models import (
    EnemyTeamState,
    Observation,
    OurTeamState,
    Position,
    RobotState,
    ShopItem,
    TaskPointState,
    UnitState,
    WorldNews,
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
    attack_power: int = 0,
    attack_range: int = 0,
    backpack_capacity: int = 100,
    backpack: Iterable[str] = (),
    level: int | None = None,
    cooldown: int = 0,
    provided_fields: Iterable[str] = (),
) -> UnitState:
    return UnitState(
        unit_id=unit_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        attack_power=attack_power,
        attack_range=attack_range,
        backpack_capacity=backpack_capacity,
        backpack=tuple(backpack),
        level=level,
        cooldown=cooldown,
        provided_fields=frozenset(provided_fields),
    )


def robot(
    robot_id: int,
    x: int,
    y: int,
    *,
    role_type: str = "smallRobot",
    health: int = 40,
    target_team: str | None = "challenger",
    abnormal_state: str = "",
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        target_team=target_team,
        abnormal_state=abnormal_state,
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
    weapon_shop: Iterable[ShopItem] = (),
    tasks: Iterable[TaskPointState] = (),
    phase_task: str = '',
    llm_response: str = '',
    last_action_results: Mapping[int, bool] | None = None,
    total_score: int = 0,
    world_news: WorldNews = WorldNews(),
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
            total_score=total_score,
            tasks=tuple(tasks),
            units=tuple(our_units),
        ),
        enemy=EnemyTeamState(units=tuple(enemy_units)),
        robots=tuple(robots),
        phase_task=phase_task,
        llm_response=llm_response,
        last_action_results=(
            {} if last_action_results is None else last_action_results
        ),
        world_news=world_news,
        vendor_shop=tuple(vendor_shop),
        weapon_shop=tuple(weapon_shop),
    )
