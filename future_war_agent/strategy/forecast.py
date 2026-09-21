from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, UnitState
from future_war_agent.protocol.time import NIGHT_ROUNDS, Phase

from .night import ControllerAssignment, assign_controllers
from .policy import StrategyProfile
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config, ROBOT_SPECS
from .simulation.future import choose_future_action
from .simulation.kernel import step_simulation
from .simulation.robots import RobotPolicy
from .simulation.state import AssignedStand, build_sim_state
from .simulation.tail import estimate_tail
from .world import WorldGrid


MODEL_VERSION = 'night-forecast-v1'
_PERSONAL_ROLES = frozenset({'worker', 'pioneer'})
_WEAPON_ROLES = frozenset({'gatling', 'railgun', 'rocket'})


class RiskLevel(StrEnum):
    UNKNOWN = 'unknown'
    SAFE = 'safe'
    WATCH = 'watch'
    CRITICAL = 'critical'
    LETHAL = 'lethal'


class ForecastUpdateKind(StrEnum):
    FULL = 'full'
    INCREMENTAL = 'incremental'
    REBASED_DAY = 'rebased_day'


@dataclass(frozen=True, slots=True)
class NightForecast:
    expected_station_hp_at_dawn: int
    predicted_damage_before_dawn: int
    effective_defense_hp: int
    future_firepower: int
    survival_margin: int
    risk_ratio: Fraction
    risk_level: RiskLevel
    expected_wall_losses: int
    expected_weapon_losses: int
    expected_role_losses: int
    lethal_round: int | None
    critical_robot_ids: tuple[int, ...]
    critical_wall_ids: tuple[int, ...]
    generated_round: int
    updated_round: int
    day_no: int
    model_version: str
    update_kind: ForecastUpdateKind
    complete: bool
    uncertainty_reasons: tuple[str, ...]
    observed_station_hp: int
    expected_next_station_hp: int
    ineffective_attack_streak: int = 0


@dataclass(frozen=True, slots=True)
class ForecastRefresh:
    forecast: NightForecast
    recomputed: bool
    reason: str


def refresh_night_forecast(
    observation: Observation,
    *,
    previous_observation: Observation | None = None,
    previous_forecast: NightForecast | None = None,
    previous_decision: Decision | None = None,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> ForecastRefresh:
    if observation.time.phase is not Phase.NIGHT:
        raise ValueError('NightForecast requires a night observation')
    reason = forecast_recompute_reason(
        observation,
        previous_observation=previous_observation,
        previous_forecast=previous_forecast,
        previous_decision=previous_decision,
    )
    if reason is not None:
        return ForecastRefresh(
            forecast=build_night_forecast(
                observation,
                controller_assignments=controller_assignments,
                config=config,
            ),
            recomputed=True,
            reason=reason,
        )
    if previous_forecast is None:
        raise AssertionError('missing forecast must require recomputation')
    return ForecastRefresh(
        forecast=update_forecast_incrementally(
            observation,
            previous_forecast,
            previous_decision=previous_decision,
        ),
        recomputed=False,
        reason='ordinary consecutive night round',
    )


def build_night_forecast(
    observation: Observation,
    *,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> NightForecast:
    world = WorldGrid.from_observation(observation)
    assignments = tuple(
        AssignedStand(item.role_id, item.weapon_id, item.stand)
        for item in (
            assign_controllers(observation, world)
            if controller_assignments is None
            else controller_assignments
        )
    )
    state = build_sim_state(observation, assignments, config)
    initial_station_hp = state.station.health
    initial_wall_ids = frozenset(wall.unit_id for wall in state.walls)
    initial_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    initial_role_ids = frozenset(role.unit_id for role in state.roles)
    expected_next_hp = initial_station_hp
    lethal_round = None
    horizon = min(
        config.forecast_exact_horizon,
        state.remaining_night_turns,
    )
    for step_index in range(horizon):
        action = choose_future_action(
            state,
            config,
            profile=StrategyProfile.SURVIVE,
        )
        state = step_simulation(
            state,
            action,
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
            config,
        )
        if step_index == 0:
            expected_next_hp = state.station.health
        if state.station.health <= 0:
            lethal_round = state.round_no
            break

    tail = estimate_tail(state, config)
    if lethal_round is None:
        lethal_round = tail.lethal_round
    exact_station_damage = max(0, initial_station_hp - state.station.health)
    predicted_damage = exact_station_damage + tail.incoming_damage
    effective_defense = (
        initial_station_hp
        + max(0, tail.effective_hp - state.station.health)
    )
    survival_margin = (
        effective_defense + tail.future_firepower - predicted_damage
    )
    risk_ratio = Fraction(predicted_damage, max(1, effective_defense))
    risk_level = classify_risk(
        risk_ratio,
        survival_margin,
        lethal_round,
        observation.time.round_no,
        complete=tail.complete,
    )
    surviving_wall_ids = frozenset(wall.unit_id for wall in state.walls)
    surviving_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    surviving_role_ids = frozenset(role.unit_id for role in state.roles)
    return NightForecast(
        expected_station_hp_at_dawn=tail.expected_station_hp_at_dawn,
        predicted_damage_before_dawn=predicted_damage,
        effective_defense_hp=effective_defense,
        future_firepower=tail.future_firepower,
        survival_margin=survival_margin,
        risk_ratio=risk_ratio,
        risk_level=risk_level,
        expected_wall_losses=(
            len(initial_wall_ids - surviving_wall_ids)
            + tail.expected_wall_losses
        ),
        expected_weapon_losses=(
            len(initial_weapon_ids - surviving_weapon_ids)
            + tail.expected_weapon_losses
        ),
        expected_role_losses=(
            len(initial_role_ids - surviving_role_ids)
            + tail.expected_role_losses
        ),
        lethal_round=lethal_round,
        critical_robot_ids=tail.critical_robot_ids,
        critical_wall_ids=tail.critical_wall_ids,
        generated_round=observation.time.round_no,
        updated_round=observation.time.round_no,
        day_no=observation.time.day_no,
        model_version=MODEL_VERSION,
        update_kind=ForecastUpdateKind.FULL,
        complete=tail.complete,
        uncertainty_reasons=tail.uncertainty_reasons,
        observed_station_hp=initial_station_hp,
        expected_next_station_hp=expected_next_hp,
    )


def update_forecast_incrementally(
    observation: Observation,
    cached: NightForecast,
    *,
    previous_decision: Decision | None = None,
) -> NightForecast:
    station = _living_unit(observation, 'station')
    current_hp = station.health if station is not None else 0
    actual_damage = max(0, cached.observed_station_hp - current_hp)
    expected_damage = max(
        0,
        cached.observed_station_hp - cached.expected_next_station_hp,
    )
    remaining_damage = max(
        0,
        cached.predicted_damage_before_dawn - actual_damage,
    )
    effective_defense = max(0, cached.effective_defense_hp - actual_damage)
    margin = (
        effective_defense + cached.future_firepower - remaining_damage
    )
    ratio = Fraction(remaining_damage, max(1, effective_defense))
    expected_dawn_hp = max(
        0,
        min(
            current_hp,
            cached.expected_station_hp_at_dawn
            + expected_damage
            - actual_damage,
        ),
    )
    remaining_rounds = max(
        1,
        NIGHT_ROUNDS - observation.time.round_in_phase + 1,
    )
    next_damage = _ceil_fraction(
        Fraction(remaining_damage, remaining_rounds)
    )
    streak = _ineffective_attack_streak(
        observation,
        previous_decision,
        cached.ineffective_attack_streak,
    )
    risk_level = classify_risk(
        ratio,
        margin,
        cached.lethal_round,
        observation.time.round_no,
        complete=cached.complete,
    )
    living_robot_ids = {
        robot.robot_id for robot in observation.robots if robot.health > 0
    }
    living_wall_ids = {
        unit.unit_id
        for unit in observation.our.units
        if unit.health > 0 and unit.role_type == 'wall'
    }
    return replace(
        cached,
        expected_station_hp_at_dawn=expected_dawn_hp,
        predicted_damage_before_dawn=remaining_damage,
        effective_defense_hp=effective_defense,
        survival_margin=margin,
        risk_ratio=ratio,
        risk_level=risk_level,
        critical_robot_ids=tuple(
            robot_id
            for robot_id in cached.critical_robot_ids
            if robot_id in living_robot_ids
        ),
        critical_wall_ids=tuple(
            wall_id
            for wall_id in cached.critical_wall_ids
            if wall_id in living_wall_ids
        ),
        updated_round=observation.time.round_no,
        update_kind=ForecastUpdateKind.INCREMENTAL,
        observed_station_hp=current_hp,
        expected_next_station_hp=max(0, current_hp - next_damage),
        ineffective_attack_streak=streak,
    )


def rebase_day_forecast(
    observation: Observation,
    cached: NightForecast,
) -> NightForecast:
    if observation.time.phase is not Phase.DAY:
        raise ValueError('day rebase requires a day observation')
    station = _living_unit(observation, 'station')
    current_hp = station.health if station is not None else 0
    hp_delta = current_hp - cached.observed_station_hp
    effective_defense = max(
        0,
        cached.effective_defense_hp + hp_delta,
    )
    margin = cached.survival_margin + hp_delta
    ratio = Fraction(
        cached.predicted_damage_before_dawn,
        max(1, effective_defense),
    )
    risk_level = classify_risk(
        ratio,
        margin,
        None,
        observation.time.round_no,
        complete=cached.complete,
    )
    return replace(
        cached,
        expected_station_hp_at_dawn=max(
            0,
            min(current_hp, cached.expected_station_hp_at_dawn + hp_delta),
        ),
        effective_defense_hp=effective_defense,
        risk_ratio=ratio,
        survival_margin=margin,
        risk_level=risk_level,
        lethal_round=None,
        updated_round=observation.time.round_no,
        day_no=observation.time.day_no,
        update_kind=ForecastUpdateKind.REBASED_DAY,
        observed_station_hp=current_hp,
        expected_next_station_hp=current_hp,
        ineffective_attack_streak=0,
    )


def forecast_recompute_reason(
    observation: Observation,
    *,
    previous_observation: Observation | None,
    previous_forecast: NightForecast | None,
    previous_decision: Decision | None,
) -> str | None:
    if previous_forecast is None:
        return 'night forecast missing'
    if previous_forecast.model_version != MODEL_VERSION:
        return 'forecast model changed'
    if previous_forecast.day_no != observation.time.day_no:
        return 'first forecast of night'
    if previous_observation is None or previous_observation.time.phase is not Phase.NIGHT:
        return 'first forecast of night'
    if observation.time.round_in_phase == 1:
        return 'first forecast of night'
    if bool(previous_observation.phase_task.strip()) != bool(
        observation.phase_task.strip()
    ):
        return 'active task changed controller availability'
    lost_kind = _lost_asset_kind(previous_observation, observation)
    if lost_kind is not None:
        return f'{lost_kind} destroyed'
    if _station_damage_deviated(observation, previous_forecast):
        return 'station damage deviated from forecast'
    if _wave_changed_materially(previous_observation, observation):
        return 'robot wave changed materially'
    if _experimental_item_was_used(previous_decision):
        return 'experimental emergency item used'
    if (
        previous_forecast.risk_level is not RiskLevel.SAFE
        and _rocket_became_ready(previous_observation, observation)
    ):
        return 'rocket readiness changed safety'
    streak = _ineffective_attack_streak(
        observation,
        previous_decision,
        previous_forecast.ineffective_attack_streak,
    )
    if streak >= 2:
        return 'joint fire ineffective twice'
    return None


def classify_risk(
    risk_ratio: Fraction,
    survival_margin: int,
    lethal_round: int | None,
    current_round: int,
    *,
    complete: bool = True,
) -> RiskLevel:
    if (
        lethal_round is not None
        and lethal_round - current_round <= 2
    ):
        return RiskLevel.LETHAL
    if risk_ratio >= Fraction(17, 20) or survival_margin < 0:
        return RiskLevel.CRITICAL
    if risk_ratio >= Fraction(11, 20):
        return RiskLevel.WATCH
    if not complete:
        return RiskLevel.UNKNOWN
    return RiskLevel.SAFE


def _station_damage_deviated(
    observation: Observation,
    forecast: NightForecast,
) -> bool:
    station = _living_unit(observation, 'station')
    current_hp = station.health if station is not None else 0
    expected = max(
        0,
        forecast.observed_station_hp - forecast.expected_next_station_hp,
    )
    actual = max(0, forecast.observed_station_hp - current_hp)
    if expected == 0:
        return actual > 0
    return Fraction(abs(actual - expected), expected) >= Fraction(1, 2)


def _wave_changed_materially(
    previous: Observation,
    current: Observation,
) -> bool:
    before = {
        robot.robot_id: robot
        for robot in previous.robots
        if robot.health > 0 and robot.target_team == previous.our.team_type
    }
    after = {
        robot.robot_id: robot
        for robot in current.robots
        if robot.health > 0 and robot.target_team == current.our.team_type
    }
    if set(after) - set(before):
        return True
    if abs(len(after) - len(before)) >= 2:
        return True
    before_power = sum(_robot_attack_power(robot) for robot in before.values())
    after_power = sum(_robot_attack_power(robot) for robot in after.values())
    if abs(after_power - before_power) >= max(40, before_power // 2):
        return True
    return any(
        robot_id in before
        and after[robot_id].position.chebyshev_distance(
            before[robot_id].position
        )
        > 1
        for robot_id in after
    )


def _lost_asset_kind(
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
        if unit.role_type == 'station':
            return 'station'
    return None


def _experimental_item_was_used(decision: Decision | None) -> bool:
    if decision is None:
        return False
    return any(
        action.kind is ActionKind.USE
        and action.name is not None
        and action.name.casefold() in {'bomb', 'dizzy', 'stun'}
        for action in decision.commands.values()
    )


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


def _ineffective_attack_streak(
    observation: Observation,
    previous_decision: Decision | None,
    prior: int,
) -> int:
    if previous_decision is None:
        return prior
    attack_ids = tuple(
        actor_id
        for actor_id, action in previous_decision.commands.items()
        if action.kind is ActionKind.ATTACK
    )
    if not attack_ids or any(
        actor_id not in observation.last_action_results
        for actor_id in attack_ids
    ):
        return prior
    if any(observation.last_action_results[actor_id] for actor_id in attack_ids):
        return 0
    return prior + 1


def _living_unit(
    observation: Observation,
    role_type: str,
) -> UnitState | None:
    return next(
        (
            unit
            for unit in observation.our.units
            if unit.health > 0 and unit.role_type == role_type
        ),
        None,
    )


def _robot_attack_power(robot: UnitState) -> int:
    spec = ROBOT_SPECS.get(robot.role_type)
    if spec is None:
        return max(item.attack_power for item in ROBOT_SPECS.values())
    return spec.attack_power


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)
