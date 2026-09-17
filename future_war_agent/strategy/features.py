from collections import Counter
from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, UnitState
from future_war_agent.protocol.time import Phase

from .policy import DEFAULT_BUILD_PLAN, BuildPlan
from .rules import DEFAULT_RULES, station_footprint
from .simulation.certificate import (
    RobotWaveSafetyCertificate,
    WaveClassification,
)


_PERSONAL_ROLES = frozenset({"worker", "pioneer"})
_WEAPON_ROLES = frozenset({"gatling", "railgun", "rocket"})
_ROBOT_ATTACK_POWER = {
    "smallrobot": 10,
    "mediumrobot": 20,
    "largerobot": 30,
    "bossrobot": 40,
}


@dataclass(frozen=True, slots=True)
class StrategyFeatures:
    round_no: int
    day_no: int
    phase: Phase
    gold: int
    score: int
    our_station_health: int | None
    our_station_level: int | None
    enemy_station_health: int | None
    station_health_loss: int
    living_personal_role_count: int
    living_weapon_count: int
    living_wall_count: int
    targeted_robot_count: int
    targeted_robot_attack_power: int
    nearest_targeted_robot_distance: int | None
    defense_complete: bool
    wave_classification: WaveClassification
    wave_secured: bool | None


def extract_features(
    observation: Observation,
    *,
    previous_observation: Observation | None = None,
    certificate: RobotWaveSafetyCertificate | None = None,
    build_plan: BuildPlan = DEFAULT_BUILD_PLAN,
) -> StrategyFeatures:
    station = _station(observation.our.units)
    previous_station = (
        _station(previous_observation.our.units)
        if previous_observation is not None
        else None
    )
    enemy_station = _station(observation.enemy.units)
    station_health_loss = 0
    if station is not None and previous_station is not None:
        station_health_loss = max(0, previous_station.health - station.health)

    living = tuple(unit for unit in observation.our.units if unit.health > 0)
    weapon_counts = Counter(
        unit.role_type for unit in living if unit.role_type in _WEAPON_ROLES
    )
    required_counts = Counter(build_plan.weapon_loadout)
    defense_complete = (
        not build_plan.build_weapons
        or all(weapon_counts[name] >= count for name, count in required_counts.items())
    )

    targeted = tuple(
        robot
        for robot in observation.robots
        if robot.health > 0 and robot.target_team == observation.our.team_type
    )
    attack_power = sum(_robot_attack_power(robot.role_type) for robot in targeted)
    nearest_distance = None
    if station is not None and targeted:
        station_cells = station_footprint(station.position, DEFAULT_RULES)
        nearest_distance = min(
            robot.position.chebyshev_distance(cell)
            for robot in targeted
            for cell in station_cells
        )

    return StrategyFeatures(
        round_no=observation.time.round_no,
        day_no=observation.time.day_no,
        phase=observation.time.phase,
        gold=observation.our.gold,
        score=observation.our.total_score,
        our_station_health=station.health if station is not None else None,
        our_station_level=station.level if station is not None else None,
        enemy_station_health=(
            enemy_station.health if enemy_station is not None else None
        ),
        station_health_loss=station_health_loss,
        living_personal_role_count=sum(
            unit.role_type in _PERSONAL_ROLES for unit in living
        ),
        living_weapon_count=sum(unit.role_type in _WEAPON_ROLES for unit in living),
        living_wall_count=sum(unit.role_type == "wall" for unit in living),
        targeted_robot_count=len(targeted),
        targeted_robot_attack_power=attack_power,
        nearest_targeted_robot_distance=nearest_distance,
        defense_complete=defense_complete,
        wave_classification=(
            certificate.classification
            if certificate is not None
            else WaveClassification.UNKNOWN
        ),
        wave_secured=certificate.secured if certificate is not None else None,
    )


def _station(units: tuple[UnitState, ...]) -> UnitState | None:
    return next(
        (
            unit
            for unit in units
            if unit.role_type == "station" and unit.health > 0
        ),
        None,
    )


def _robot_attack_power(role_type: str) -> int:
    normalized = role_type.replace("_", "").replace("-", "").lower()
    return _ROBOT_ATTACK_POWER.get(normalized, 0)
