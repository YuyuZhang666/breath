import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from time import monotonic, perf_counter_ns

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.strategy.night import (
    ControllerAssignment,
    assign_controllers,
)
from future_war_agent.strategy.policy import (
    DEFAULT_STRATEGIC_INTENT,
    StrategicIntent,
)
from future_war_agent.strategy.world import WorldGrid

from .candidates import RootAction, SimJointAction, generate_root_actions
from .certificate import (
    RobotWaveSafetyCertificate,
    ScenarioOutcome,
    WaveClassification,
    build_certificate,
    score_rank_key,
    survival_rank_key,
)
from .config import (
    DEFAULT_PHASE3_CONFIG,
    Phase3Budget,
    Phase3Config,
    Phase3Level,
)
from .errors import DeadlineExceeded, UnsupportedSimulation
from .future import choose_future_action
from .kernel import step_simulation
from .objective import DEFAULT_NIGHT_OBJECTIVE, NightObjective
from .robots import ALL_ROBOT_POLICIES, RobotPolicy
from .state import AssignedStand, SimState, build_sim_state
from .tail import TailEstimate, estimate_tail


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


ScenarioWeights = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class SearchStats:
    roots_generated: int
    roots_evaluated: int
    scenarios_per_root: int
    maximum_steps: int
    candidate_generation_ms: float = 0.0
    simulation_ms: float = 0.0
    tail_estimator_ms: float = 0.0
    level: Phase3Level = Phase3Level.FULL
    deadline_hit: bool = False


@dataclass(frozen=True, slots=True)
class SearchResult:
    decision: Decision
    simulation_action: SimJointAction
    certificate: RobotWaveSafetyCertificate
    stats: SearchStats


def search_night(
    observation: Observation,
    scenario_weights: ScenarioWeights,
    *,
    objective: NightObjective = DEFAULT_NIGHT_OBJECTIVE,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    level: Phase3Level = Phase3Level.FULL,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    baseline_decision: Decision | None = None,
    budget: Phase3Budget | None = None,
) -> SearchResult:
    _validate_weights(scenario_weights)
    resolved_budget = config.budget_for(level) if budget is None else budget
    started_at = clock()
    budget_deadline = started_at + resolved_budget.seconds
    effective_deadline = (
        budget_deadline
        if deadline is None
        else min(deadline, budget_deadline)
    )
    _check_deadline(clock, effective_deadline)

    candidate_started_ns = perf_counter_ns()
    world = WorldGrid.from_observation(observation)
    resolved_assignments = (
        assign_controllers(observation, world)
        if controller_assignments is None
        else controller_assignments
    )
    assignments = tuple(
        AssignedStand(item.role_id, item.weapon_id, item.stand)
        for item in resolved_assignments
    )
    state = build_sim_state(observation, assignments, config)
    roots = generate_root_actions(
        observation,
        world,
        state,
        config,
        intent=intent,
        controller_assignments=resolved_assignments,
        baseline_decision=baseline_decision,
        max_roots=resolved_budget.root_candidates,
        deadline_check=lambda: _check_deadline(clock, effective_deadline),
        level=level,
    )
    candidate_generation_ms = (
        perf_counter_ns() - candidate_started_ns
    ) / 1_000_000
    _check_deadline(clock, effective_deadline)
    if not roots:
        raise UnsupportedSimulation("no legal Phase 3 root actions")

    horizon = min(resolved_budget.exact_horizon, state.remaining_night_turns)
    scenarios = _select_scenarios(scenario_weights, resolved_budget.scenarios)
    initial_controlled_role_ids = frozenset(
        role.unit_id for role in state.roles
    )
    initial_controller_ids = frozenset(
        role.unit_id
        for role in state.roles
        if role.assigned_weapon_id is not None
    )
    initial_key_weapon_ids = frozenset(
        role.assigned_weapon_id
        for role in state.roles
        if role.assigned_weapon_id is not None
    )
    initial_wall_ids = frozenset(wall.unit_id for wall in state.walls)
    initial_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    non_key_weapon_ids = initial_weapon_ids - initial_key_weapon_ids
    evaluated: list[tuple[RootAction, RobotWaveSafetyCertificate]] = []
    simulation_started_ns = perf_counter_ns()
    tail_estimator_ms = 0.0

    deadline_hit = False
    for root in roots:
        if clock() >= effective_deadline:
            deadline_hit = True
            break
        outcomes: list[ScenarioOutcome] = []
        complete_root = True
        for policy, weight in scenarios:
            if clock() >= effective_deadline:
                deadline_hit = True
                complete_root = False
                break
            simulated = state
            for step_index in range(horizon):
                if clock() >= effective_deadline:
                    deadline_hit = True
                    complete_root = False
                    break
                try:
                    simulated_action = (
                        root.simulation_action
                        if step_index == 0
                        else choose_future_action(
                            simulated,
                            config,
                            profile=intent.profile,
                            deadline_check=lambda: _check_deadline(
                                clock,
                                effective_deadline,
                            ),
                        )
                    )
                    simulated = step_simulation(
                        simulated,
                        simulated_action,
                        policy,
                        config,
                        deadline_check=lambda: _check_deadline(
                            clock,
                            effective_deadline,
                        ),
                    )
                except DeadlineExceeded:
                    deadline_hit = True
                    complete_root = False
                    break
                if clock() >= effective_deadline:
                    deadline_hit = True
                    complete_root = False
                    break
                if simulated.station.health <= 0:
                    break
            if not complete_root:
                break
            tail_estimate = None
            if config.use_tail_estimator and simulated.remaining_night_turns > 0:
                tail_started_ns = perf_counter_ns()
                try:
                    tail_estimate = estimate_tail(
                        simulated,
                        config,
                        clock=clock,
                        deadline=effective_deadline,
                    )
                except DeadlineExceeded:
                    deadline_hit = True
                    LOGGER.warning(
                        'TailEstimator reached the Phase 3 deadline; '
                        'retaining bounded certificate'
                    )
                except Exception:
                    LOGGER.exception(
                        'TailEstimator failed; retaining bounded certificate'
                    )
                finally:
                    tail_estimator_ms += (
                        perf_counter_ns() - tail_started_ns
                    ) / 1_000_000
            outcomes.append(
                _scenario_outcome(
                    simulated,
                    weight,
                    initial_controlled_role_ids=initial_controlled_role_ids,
                    initial_controller_ids=initial_controller_ids,
                    initial_key_weapon_ids=initial_key_weapon_ids,
                    initial_wall_ids=initial_wall_ids,
                    initial_weapon_ids=initial_weapon_ids,
                    non_key_weapon_ids=non_key_weapon_ids,
                    tail_estimate=tail_estimate,
                )
            )
        if not complete_root:
            break
        evaluated.append(
            (
                root,
                _projected_certificate(tuple(outcomes), objective),
            )
        )
    if not evaluated:
        raise DeadlineExceeded("Phase 3 search expired before one complete root")

    simulation_ms = (perf_counter_ns() - simulation_started_ns) / 1_000_000

    secured = tuple(item for item in evaluated if item[1].secured)
    if secured and objective.enable_score_band:
        pool = secured
        winner = min(
            pool,
            key=lambda item: score_rank_key(item[1], item[0].stable_key),
        )
    else:
        pool = tuple(evaluated)
        winner = min(
            pool,
            key=lambda item: survival_rank_key(item[1], item[0].stable_key),
        )
    root, certificate = winner
    return SearchResult(
        decision=root.decision,
        simulation_action=root.simulation_action,
        certificate=certificate,
        stats=SearchStats(
            roots_generated=len(roots),
            roots_evaluated=len(evaluated),
            scenarios_per_root=len(scenarios),
            maximum_steps=horizon,
            candidate_generation_ms=candidate_generation_ms,
            simulation_ms=simulation_ms,
            tail_estimator_ms=tail_estimator_ms,
            level=level,
            deadline_hit=deadline_hit,
        ),
    )


def _projected_certificate(
    outcomes: tuple[ScenarioOutcome, ...],
    objective: NightObjective,
) -> RobotWaveSafetyCertificate:
    certificate = build_certificate(outcomes, objective)
    if (
        certificate.classification is not WaveClassification.WAVE_SAFE
        or all(
            outcome.ended_with_night or outcome.projection_complete
            for outcome in outcomes
        )
    ):
        return certificate
    return replace(
        certificate,
        classification=WaveClassification.UNKNOWN,
        secured=False,
    )


def _select_scenarios(
    weights: ScenarioWeights,
    count: int,
) -> tuple[tuple[RobotPolicy, Fraction], ...]:
    worst = RobotPolicy.MAXIMUM_STATION_PROGRESS
    worst_index = ALL_ROBOT_POLICIES.index(worst)
    selected_indices = [worst_index]
    if count > 1:
        alternatives = sorted(
            (
                (-weights[index], policy.value, index)
                for index, policy in enumerate(ALL_ROBOT_POLICIES)
                if index != worst_index
            )
        )
        selected_indices.extend(item[2] for item in alternatives[: count - 1])
    selected_total = sum((weights[index] for index in selected_indices), Fraction())
    return tuple(
        (ALL_ROBOT_POLICIES[index], weights[index] / selected_total)
        for index in selected_indices
    )


def _scenario_outcome(
    state: SimState,
    weight: Fraction,
    *,
    initial_controlled_role_ids: frozenset[int],
    initial_controller_ids: frozenset[int],
    initial_key_weapon_ids: frozenset[int],
    initial_wall_ids: frozenset[int],
    initial_weapon_ids: frozenset[int],
    non_key_weapon_ids: frozenset[int],
    tail_estimate: TailEstimate | None = None,
) -> ScenarioOutcome:
    surviving_role_ids = frozenset(role.unit_id for role in state.roles)
    surviving_wall_ids = frozenset(wall.unit_id for wall in state.walls)
    surviving_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    exact_wall_losses = len(initial_wall_ids - surviving_wall_ids)
    exact_weapon_losses = len(initial_weapon_ids - surviving_weapon_ids)
    projected_station_health = (
        tail_estimate.expected_station_hp_at_dawn
        if tail_estimate is not None
        else state.station.health
    )
    return ScenarioOutcome(
        weight=weight,
        station_health=projected_station_health,
        surviving_controlled_role_count=len(
            initial_controlled_role_ids & surviving_role_ids
        ),
        surviving_controller_count=len(
            initial_controller_ids & surviving_role_ids
        ),
        controller_losses=len(initial_controller_ids - surviving_role_ids),
        surviving_key_weapon_count=len(
            initial_key_weapon_ids & surviving_weapon_ids
        ),
        key_weapon_losses=len(initial_key_weapon_ids - surviving_weapon_ids),
        wall_losses=(
            exact_wall_losses
            + (
                min(
                    len(surviving_wall_ids),
                    tail_estimate.expected_wall_losses,
                )
                if tail_estimate is not None
                else 0
            )
        ),
        weapon_losses=(
            exact_weapon_losses
            + (
                min(
                    len(surviving_weapon_ids),
                    tail_estimate.expected_weapon_losses,
                )
                if tail_estimate is not None
                else 0
            )
        ),
        minimum_controlled_role_health=min(
            (
                role.health
                for role in state.roles
                if role.unit_id in initial_controlled_role_ids
            ),
            default=0,
        ),
        surviving_wall_non_key_weapon_value=(
            sum(wall.health for wall in state.walls)
            + sum(
                weapon.health
                for weapon in state.weapons
                if weapon.unit_id in non_key_weapon_ids
            )
        ),
        owned_kill_score=state.owned_kill_score,
        remaining_threat=(
            tail_estimate.incoming_damage
            if tail_estimate is not None
            else sum(
                robot.attack_power * robot.health for robot in state.robots
            )
        ),
        remaining_one_turn_damage=(
            0
            if tail_estimate is not None
            else sum(robot.attack_power for robot in state.robots)
        ),
        ended_with_night=state.remaining_night_turns == 0,
        tail_estimated=tail_estimate is not None,
        projection_complete=(
            tail_estimate.complete if tail_estimate is not None else False
        ),
        tail_survival_margin=(
            tail_estimate.survival_margin
            if tail_estimate is not None
            else None
        ),
        tail_risk_ratio=(
            Fraction(
                tail_estimate.incoming_damage,
                max(1, tail_estimate.effective_hp),
            )
            if tail_estimate is not None
            else None
        ),
        tail_lethal_round=(
            tail_estimate.lethal_round
            if tail_estimate is not None
            else None
        ),
        uncertainty_reasons=(
            tail_estimate.uncertainty_reasons
            if tail_estimate is not None
            else ()
        ),
    )


def _validate_weights(weights: tuple[Fraction, ...]) -> None:
    if len(weights) != len(ALL_ROBOT_POLICIES):
        raise ValueError("Phase 3 requires exactly four scenario weights")
    if any(weight <= 0 for weight in weights):
        raise ValueError("scenario weights must be positive")
    if sum(weights, Fraction()) != 1:
        raise ValueError("scenario weights must sum to one")


def _check_deadline(
    clock: Callable[[], float],
    deadline: float,
) -> None:
    if clock() >= deadline:
        raise DeadlineExceeded("Phase 3 search deadline expired")
