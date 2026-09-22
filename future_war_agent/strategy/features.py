from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, UnitState
from future_war_agent.protocol.time import Phase

from .defense import core_weapon_readiness
from .policy import DEFAULT_BUILD_PLAN, BuildPlan
from .forecast import NightForecast, RiskLevel
from .safety import CheapestSafePlan
from .rules import DEFAULT_RULES, station_footprint
from .simulation.certificate import (
    RobotWaveSafetyCertificate,
    WaveClassification,
)
from .simulation.config import ROBOT_SPECS, RobotSpec


_PERSONAL_ROLES = frozenset({"worker", "pioneer"})
_WEAPON_ROLES = frozenset({"gatling", "railgun", "rocket"})
_ROBOT_SPECS_BY_NORMALIZED_NAME = {
    name.replace('_', '').replace('-', '').lower(): spec
    for name, spec in ROBOT_SPECS.items()
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
    targeted_robot_one_turn_attack_power: int
    nearest_targeted_robot_distance: int | None
    defense_complete: bool
    wave_classification: WaveClassification
    wave_secured: bool | None
    night_risk_level: RiskLevel | None
    risk_ratio: float | None
    survival_margin: int | None
    safety_plan_cost: int | None


def extract_features(
    observation: Observation,
    *,
    previous_observation: Observation | None = None,
    certificate: RobotWaveSafetyCertificate | None = None,
    forecast: NightForecast | None = None,
    safety_plan: CheapestSafePlan | None = None,
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
    weapon_readiness = core_weapon_readiness(
        observation.our.units,
        build_plan.weapon_loadout,
    )
    defense_complete = (
        not build_plan.build_weapons
        or weapon_readiness.complete
    )

    targeted = tuple(
        robot
        for robot in observation.robots
        if robot.health > 0 and robot.target_team == observation.our.team_type
    )
    nearest_distance = None
    one_turn_attack_power = 0
    if station is not None and targeted:
        station_cells = station_footprint(station.position, DEFAULT_RULES)
        distances = tuple(
            (
                robot,
                min(
                    robot.position.chebyshev_distance(cell)
                    for cell in station_cells
                ),
            )
            for robot in targeted
        )
        nearest_distance = min(distance for _, distance in distances)
        one_turn_attack_power = sum(
            _robot_spec(robot.role_type).attack_power
            for robot, distance in distances
            if distance <= _robot_spec(robot.role_type).attack_range
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
        targeted_robot_attack_power=sum(
            _robot_spec(robot.role_type).attack_power for robot in targeted
        ),
        targeted_robot_one_turn_attack_power=one_turn_attack_power,
        nearest_targeted_robot_distance=nearest_distance,
        defense_complete=defense_complete,
        wave_classification=(
            certificate.classification
            if certificate is not None
            else WaveClassification.UNKNOWN
        ),
        wave_secured=certificate.secured if certificate is not None else None,
        night_risk_level=(
            forecast.risk_level if forecast is not None else None
        ),
        risk_ratio=(
            float(forecast.risk_ratio) if forecast is not None else None
        ),
        survival_margin=(
            forecast.survival_margin if forecast is not None else None
        ),
        safety_plan_cost=(
            safety_plan.gold_cost if safety_plan is not None else None
        ),
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


def _robot_spec(role_type: str) -> RobotSpec:
    normalized = role_type.replace('_', '').replace('-', '').lower()
    return _ROBOT_SPECS_BY_NORMALIZED_NAME.get(
        normalized,
        RobotSpec(0, 0, 0, 0),
    )
