from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, UnitState
from future_war_agent.protocol.time import Phase
from future_war_agent.strategy.policy import StrategicIntent, StrategyProfile
from future_war_agent.strategy.rules import DEFAULT_RULES, station_footprint

from .certificate import RobotWaveSafetyCertificate
from .config import Phase3Level, ROBOT_SPECS, RobotSpec


_PERSONAL_ROLES = frozenset({'worker', 'pioneer'})
_WEAPON_ROLES = frozenset({'gatling', 'railgun', 'rocket'})
_ROBOT_SPECS_BY_NORMALIZED_NAME = {
    name.replace('_', '').replace('-', '').lower(): spec
    for name, spec in ROBOT_SPECS.items()
}


@dataclass(frozen=True, slots=True)
class Phase3Trigger:
    level: Phase3Level
    reason: str


@dataclass(frozen=True, slots=True)
class Phase3TriggerConfig:
    large_wave_count_delta: int = 4
    large_wave_attack_power_delta: int = 40
    wall_pressure_margin: int = 1

    def __post_init__(self) -> None:
        if self.large_wave_count_delta <= 0:
            raise ValueError('large_wave_count_delta must be positive')
        if self.large_wave_attack_power_delta <= 0:
            raise ValueError('large_wave_attack_power_delta must be positive')
        if self.wall_pressure_margin < 0:
            raise ValueError('wall_pressure_margin cannot be negative')


DEFAULT_PHASE3_TRIGGER_CONFIG = Phase3TriggerConfig()


def select_phase3_level(
    observation: Observation,
    previous: Observation | None,
    intent: StrategicIntent,
    *,
    certificate: RobotWaveSafetyCertificate | None = None,
    config: Phase3TriggerConfig = DEFAULT_PHASE3_TRIGGER_CONFIG,
) -> Phase3Trigger:
    del certificate
    if observation.time.phase is not Phase.NIGHT:
        return Phase3Trigger(Phase3Level.NONE, 'day phase')

    if intent.profile is StrategyProfile.DESPERATION:
        return Phase3Trigger(Phase3Level.FULL, 'desperation mode')
    station = _living_unit(observation, 'station')
    targeted = tuple(
        robot
        for robot in observation.robots
        if robot.health > 0 and robot.target_team == observation.our.team_type
    )
    immediate_damage = 0
    previous_targeted: tuple[UnitState, ...] = ()
    if station is not None:
        immediate_damage = sum(
            _robot_spec(robot.role_type).attack_power
            for robot in targeted
            if _distance_to_station(robot, station) <= _robot_spec(
                robot.role_type
            ).attack_range
        )
        if immediate_damage >= station.health:
            return Phase3Trigger(Phase3Level.FULL, 'visible lethal station threat')

    if previous is not None:
        if _station_health_loss(previous, observation) > 0:
            return Phase3Trigger(Phase3Level.FULL, 'station took damage')
        lost_kind = _lost_critical_kind(previous, observation)
        if lost_kind is not None:
            return Phase3Trigger(Phase3Level.FULL, f'{lost_kind} destroyed')
        previous_targeted = tuple(
            robot
            for robot in previous.robots
            if robot.health > 0 and robot.target_team == previous.our.team_type
        )
        count_delta = len(targeted) - len(previous_targeted)
        power_delta = _attack_power(targeted) - _attack_power(previous_targeted)
        if (
            count_delta >= config.large_wave_count_delta
            or power_delta >= config.large_wave_attack_power_delta
        ):
            return Phase3Trigger(Phase3Level.FULL, 'large wave increase')
        if _rocket_became_ready(previous, observation):
            return Phase3Trigger(Phase3Level.LITE, 'rocket became ready')

    current_wall_pressure = bool(targeted) and _wall_under_pressure(
        observation,
        targeted,
        margin=config.wall_pressure_margin,
    )
    previous_wall_pressure = (
        previous is not None
        and bool(previous_targeted)
        and _wall_under_pressure(
            previous,
            previous_targeted,
            margin=config.wall_pressure_margin,
        )
    )
    if current_wall_pressure and not previous_wall_pressure:
        return Phase3Trigger(Phase3Level.LITE, 'wall under pressure')
    if targeted and intent.profile is StrategyProfile.SURVIVE:
        threat_increased = (
            not previous_targeted
            or len(targeted) > len(previous_targeted)
            or _attack_power(targeted) > _attack_power(previous_targeted)
            or _nearest_station_distance(observation, targeted)
            < _nearest_station_distance(previous, previous_targeted)
        )
        if threat_increased:
            return Phase3Trigger(Phase3Level.LITE, 'visible survival pressure')
    return Phase3Trigger(Phase3Level.NONE, 'no material night event')


def _living_unit(observation: Observation, role_type: str) -> UnitState | None:
    return next(
        (
            unit
            for unit in observation.our.units
            if unit.health > 0 and unit.role_type == role_type
        ),
        None,
    )


def _station_health_loss(previous: Observation, current: Observation) -> int:
    before = _living_unit(previous, 'station')
    after = _living_unit(current, 'station')
    if before is None or after is None or before.unit_id != after.unit_id:
        return 0
    return max(0, before.health - after.health)


def _lost_critical_kind(
    previous: Observation,
    current: Observation,
) -> str | None:
    current_ids = {
        unit.unit_id for unit in current.our.units if unit.health > 0
    }
    for unit in previous.our.units:
        if unit.health <= 0 or unit.unit_id in current_ids:
            continue
        if unit.role_type in _PERSONAL_ROLES:
            return 'role'
        if unit.role_type in _WEAPON_ROLES:
            return 'weapon'
        if unit.role_type == 'wall':
            return 'wall'
    return None


def _rocket_became_ready(
    previous: Observation,
    current: Observation,
) -> bool:
    prior = {
        unit.unit_id: unit
        for unit in previous.our.units
        if unit.health > 0 and unit.role_type == 'rocket'
    }
    return any(
        unit.health > 0
        and unit.role_type == 'rocket'
        and unit.cooldown == 0
        and unit.unit_id in prior
        and prior[unit.unit_id].cooldown > 0
        for unit in current.our.units
    )


def _wall_under_pressure(
    observation: Observation,
    robots: tuple[UnitState, ...],
    *,
    margin: int,
) -> bool:
    walls = tuple(
        unit
        for unit in observation.our.units
        if unit.health > 0 and unit.role_type == 'wall'
    )
    return any(
        robot.position.chebyshev_distance(wall.position)
        <= _robot_spec(robot.role_type).attack_range + margin
        for robot in robots
        for wall in walls
    )


def _distance_to_station(robot: UnitState, station: UnitState) -> int:
    return min(
        robot.position.chebyshev_distance(cell)
        for cell in station_footprint(station.position, DEFAULT_RULES)
    )


def _attack_power(robots: tuple[UnitState, ...]) -> int:
    return sum(_robot_spec(robot.role_type).attack_power for robot in robots)


def _nearest_station_distance(
    observation: Observation,
    robots: tuple[UnitState, ...],
) -> int:
    station = _living_unit(observation, 'station')
    if station is None or not robots:
        return observation.width + observation.height
    return min(_distance_to_station(robot, station) for robot in robots)


def _robot_spec(role_type: str) -> RobotSpec:
    normalized = role_type.replace('_', '').replace('-', '').lower()
    return _ROBOT_SPECS_BY_NORMALIZED_NAME.get(
        normalized,
        RobotSpec(0, 0, 0, 0),
    )
