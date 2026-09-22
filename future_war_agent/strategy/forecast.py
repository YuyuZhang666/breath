from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from hashlib import sha256
from time import monotonic

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, RobotState, UnitState
from future_war_agent.protocol.time import NIGHT_ROUNDS, Phase

from .night import ControllerAssignment, assign_controllers
from .policy import StrategyProfile
from .rules import station_footprint
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config, ROBOT_SPECS
from .simulation.errors import DeadlineExceeded, UnsupportedSimulation
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
    LIGHTWEIGHT = 'lightweight'
    REBASED_DAY = 'rebased_day'


@dataclass(frozen=True, slots=True)
class ForecastInputSummary:
    signature: str
    station_hp: int
    hostile_robot_count: int
    hostile_robot_health: int
    hostile_attack_power: int
    ready_weapon_count: int
    minimum_station_distance: int


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
    input_signature: str = ''


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
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
    allow_full: bool = True,
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
        if not allow_full:
            incremental = None
            if (
                previous_forecast is not None
                and previous_forecast.model_version == MODEL_VERSION
                and previous_forecast.day_no == observation.time.day_no
            ):
                incremental = update_forecast_incrementally(
                    observation,
                    previous_forecast,
                    previous_decision=previous_decision,
                )
            try:
                lightweight = build_lightweight_forecast(
                    observation,
                    config=config,
                    clock=clock,
                    deadline=deadline,
                )
            except DeadlineExceeded:
                if incremental is None:
                    raise
                return ForecastRefresh(
                    forecast=incremental,
                    recomputed=False,
                    reason=f'{reason}; lightweight forecast deadline expired',
                )
            if incremental is not None:
                lightweight = _more_conservative_forecast(
                    incremental,
                    lightweight,
                )
            return ForecastRefresh(
                forecast=lightweight,
                recomputed=True,
                reason=f'{reason}; full forecast disabled',
            )
        try:
            forecast = build_night_forecast(
                observation,
                controller_assignments=controller_assignments,
                config=config,
                clock=clock,
                deadline=deadline,
            )
        except UnsupportedSimulation as exc:
            forecast = build_lightweight_forecast(
                observation,
                config=config,
                clock=clock,
                deadline=deadline,
            )
            return ForecastRefresh(
                forecast=forecast,
                recomputed=True,
                reason=f'{reason}; full forecast unsupported: {exc}',
            )
        return ForecastRefresh(
            forecast=forecast,
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
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
) -> NightForecast:
    def check_deadline() -> None:
        if deadline is not None and clock() >= deadline:
            raise DeadlineExceeded('NightForecast deadline expired')

    check_deadline()
    world = WorldGrid.from_observation(observation)
    check_deadline()
    assignments = tuple(
        AssignedStand(item.role_id, item.weapon_id, item.stand)
        for item in (
            assign_controllers(observation, world)
            if controller_assignments is None
            else controller_assignments
        )
    )
    check_deadline()
    state = build_sim_state(observation, assignments, config)
    check_deadline()
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
            deadline_check=check_deadline,
        )
        state = step_simulation(
            state,
            action,
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
            config,
            deadline_check=check_deadline,
        )
        if step_index == 0:
            expected_next_hp = state.station.health
        if state.station.health <= 0:
            lethal_round = state.round_no
            break

    check_deadline()
    tail = estimate_tail(
        state,
        config,
        clock=clock,
        deadline=deadline,
    )
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
    input_summary = summarize_forecast_inputs(observation)
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
        input_signature=input_summary.signature,
    )


def build_lightweight_forecast(
    observation: Observation,
    *,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
) -> NightForecast:
    '''Build a conservative O(units + robots) night-risk estimate.'''

    def check_deadline() -> None:
        if deadline is not None and clock() >= deadline:
            raise DeadlineExceeded('Lightweight NightForecast deadline expired')

    if observation.time.phase is not Phase.NIGHT:
        raise ValueError('NightForecast requires a night observation')
    living_stations = tuple(
        unit
        for unit in observation.our.units
        if unit.role_type == 'station' and unit.health > 0
    )
    if len(living_stations) != 1:
        raise UnsupportedSimulation(
            'lightweight forecast requires exactly one living station'
        )
    check_deadline()
    station = _living_unit(observation, 'station')
    station_hp = station.health if station is not None else 0
    station_cells = station_footprint(station.position) if station else ()
    remaining_turns = max(
        0,
        NIGHT_ROUNDS - observation.time.round_in_phase + 1,
    )
    incoming_damage = 0
    next_damage = 0
    critical_robots: list[tuple[int, int]] = []
    for robot in observation.robots:
        check_deadline()
        if (
            robot.health <= 0
            or not _is_possible_threat(robot, observation.our.team_type)
        ):
            continue
        spec = ROBOT_SPECS.get(robot.role_type)
        attack_power = (
            spec.attack_power
            if spec is not None
            else max(item.attack_power for item in ROBOT_SPECS.values())
        )
        attack_range = (
            spec.attack_range
            if spec is not None
            else max(item.attack_range for item in ROBOT_SPECS.values())
        )
        distance = (
            min(
                robot.position.chebyshev_distance(cell)
                for cell in station_cells
            )
            if station_cells
            else 0
        )
        turns_to_attack = max(0, distance - attack_range)
        if robot.abnormal_state.casefold() == 'dizzy':
            turns_to_attack += 1
        attack_turns = max(0, remaining_turns - turns_to_attack)
        contribution = attack_power * attack_turns
        incoming_damage += contribution
        if turns_to_attack == 0:
            next_damage += attack_power
        if contribution > 0:
            critical_robots.append((contribution, robot.robot_id))

    effective_defense = station_hp
    survival_margin = effective_defense - incoming_damage
    risk_ratio = Fraction(incoming_damage, max(1, effective_defense))
    lethal_round = (
        observation.time.round_no + 1
        if station_hp <= 0 or next_damage >= station_hp
        else None
    )
    uncertainty = ['lightweight_conservative_estimate']
    if not config.tail_visible_roster_complete:
        uncertainty.append('future_robot_roster_unconfirmed')
    if any(
        robot.health > 0
        and (robot.target_team is None or not robot.target_team.strip())
        for robot in observation.robots
    ):
        uncertainty.append('robot_target_team_unconfirmed')
    if (
        observation.time.day_no >= 3
        and not config.tail_late_wave_calibration_source.strip()
    ):
        uncertainty.append('d3_calibration_unavailable')
    critical_robots.sort(key=lambda item: (-item[0], item[1]))
    risk_level = classify_risk(
        risk_ratio,
        survival_margin,
        lethal_round,
        observation.time.round_no,
        complete=False,
    )
    input_summary = summarize_forecast_inputs(observation)
    return NightForecast(
        expected_station_hp_at_dawn=max(0, station_hp - incoming_damage),
        predicted_damage_before_dawn=incoming_damage,
        effective_defense_hp=effective_defense,
        future_firepower=0,
        survival_margin=survival_margin,
        risk_ratio=risk_ratio,
        risk_level=risk_level,
        expected_wall_losses=0,
        expected_weapon_losses=0,
        expected_role_losses=0,
        lethal_round=lethal_round,
        critical_robot_ids=tuple(item[1] for item in critical_robots[:8]),
        critical_wall_ids=(),
        generated_round=observation.time.round_no,
        updated_round=observation.time.round_no,
        day_no=observation.time.day_no,
        model_version=MODEL_VERSION,
        update_kind=ForecastUpdateKind.LIGHTWEIGHT,
        complete=False,
        uncertainty_reasons=tuple(uncertainty),
        observed_station_hp=station_hp,
        expected_next_station_hp=max(0, station_hp - next_damage),
        input_signature=input_summary.signature,
    )


def _more_conservative_forecast(
    incremental: NightForecast,
    lightweight: NightForecast,
) -> NightForecast:
    selected = (
        lightweight
        if lightweight.survival_margin <= incremental.survival_margin
        else incremental
    )
    risk_order = {
        RiskLevel.SAFE: 0,
        RiskLevel.UNKNOWN: 1,
        RiskLevel.WATCH: 2,
        RiskLevel.CRITICAL: 3,
        RiskLevel.LETHAL: 4,
    }
    risk_level = max(
        (incremental.risk_level, lightweight.risk_level),
        key=risk_order.__getitem__,
    )
    lethal_rounds = tuple(
        value
        for value in (incremental.lethal_round, lightweight.lethal_round)
        if value is not None
    )
    uncertainty = tuple(
        dict.fromkeys(
            (
                *selected.uncertainty_reasons,
                *lightweight.uncertainty_reasons,
                'incremental_cache_lightweight_correction',
            )
        )
    )
    return replace(
        selected,
        update_kind=ForecastUpdateKind.LIGHTWEIGHT,
        complete=False,
        risk_level=risk_level,
        lethal_round=min(lethal_rounds) if lethal_rounds else None,
        uncertainty_reasons=uncertainty,
        updated_round=lightweight.updated_round,
        input_signature=lightweight.input_signature,
    )


def update_forecast_incrementally(
    observation: Observation,
    cached: NightForecast,
    *,
    previous_decision: Decision | None = None,
) -> NightForecast:
    living_stations = tuple(
        unit
        for unit in observation.our.units
        if unit.role_type == 'station' and unit.health > 0
    )
    if len(living_stations) != 1:
        raise UnsupportedSimulation(
            'incremental forecast requires exactly one living station'
        )
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
    input_summary = summarize_forecast_inputs(observation)
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
        input_signature=input_summary.signature,
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
    input_summary = summarize_forecast_inputs(observation)
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
        input_signature=input_summary.signature,
    )


def summarize_forecast_inputs(
    observation: Observation,
) -> ForecastInputSummary:
    living_station = _living_unit(observation, 'station')
    station_hp = living_station.health if living_station is not None else 0
    hostile_robots = tuple(
        sorted(
            (
                robot
                for robot in observation.robots
                if robot.health > 0
                and _is_possible_threat(robot, observation.our.team_type)
            ),
            key=lambda robot: robot.robot_id,
        )
    )
    ready_weapon_count = sum(
        unit.health > 0
        and unit.role_type in _WEAPON_ROLES
        and unit.cooldown == 0
        for unit in observation.our.units
    )
    station_cells = (
        station_footprint(living_station.position)
        if living_station is not None
        else ()
    )
    minimum_station_distance = (
        min(
            min(
                robot.position.chebyshev_distance(cell)
                for cell in station_cells
            )
            for robot in hostile_robots
        )
        if station_cells and hostile_robots
        else -1
    )
    friendly_state = tuple(
        (
            unit.unit_id,
            unit.role_type,
            unit.health,
            unit.position.x,
            unit.position.y,
            unit.cooldown,
            unit.level,
        )
        for unit in sorted(
            observation.our.units,
            key=lambda unit: unit.unit_id,
        )
    )
    robot_state = tuple(
        (
            robot.robot_id,
            robot.role_type,
            robot.health,
            robot.position.x,
            robot.position.y,
            robot.abnormal_state,
            robot.target_team,
        )
        for robot in sorted(
            observation.robots,
            key=lambda robot: robot.robot_id,
        )
    )
    signature_payload = repr(
        (
            MODEL_VERSION,
            observation.time.day_no,
            observation.time.round_in_phase,
            bool(observation.phase_task.strip()),
            friendly_state,
            robot_state,
        )
    ).encode('utf-8')
    return ForecastInputSummary(
        signature=sha256(signature_payload).hexdigest()[:16],
        station_hp=station_hp,
        hostile_robot_count=len(hostile_robots),
        hostile_robot_health=sum(
            robot.health for robot in hostile_robots
        ),
        hostile_attack_power=sum(
            _robot_attack_power(robot) for robot in hostile_robots
        ),
        ready_weapon_count=ready_weapon_count,
        minimum_station_distance=minimum_station_distance,
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
        if robot.health > 0
        and _is_possible_threat(robot, previous.our.team_type)
    }
    after = {
        robot.robot_id: robot
        for robot in current.robots
        if robot.health > 0
        and _is_possible_threat(robot, current.our.team_type)
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


def _is_possible_threat(robot: RobotState, team_type: str) -> bool:
    target_team = robot.target_team
    return (
        target_team is None
        or not target_team.strip()
        or target_team == team_type
    )


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
