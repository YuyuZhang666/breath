from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.strategy.night import assign_controllers
from future_war_agent.strategy.world import WorldGrid

from .candidates import RootAction, SimJointAction, generate_root_actions
from .certificate import (
    RobotWaveSafetyCertificate,
    ScenarioOutcome,
    build_certificate,
    score_rank_key,
    survival_rank_key,
)
from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .errors import DeadlineExceeded, UnsupportedSimulation
from .future import choose_future_action
from .kernel import step_simulation
from .objective import DEFAULT_NIGHT_OBJECTIVE, NightObjective
from .robots import ALL_ROBOT_POLICIES
from .state import AssignedStand, SimState, build_sim_state


ScenarioWeights = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class SearchStats:
    roots_generated: int
    roots_evaluated: int
    scenarios_per_root: int
    maximum_steps: int


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
) -> SearchResult:
    _validate_weights(scenario_weights)
    effective_deadline = (
        clock() + config.watchdog_seconds if deadline is None else deadline
    )
    _check_deadline(clock, effective_deadline)

    world = WorldGrid.from_observation(observation)
    controller_assignments = assign_controllers(observation, world)
    assignments = tuple(
        AssignedStand(item.role_id, item.weapon_id, item.stand)
        for item in controller_assignments
    )
    state = build_sim_state(observation, assignments, config)
    roots = generate_root_actions(observation, world, state, config)
    _check_deadline(clock, effective_deadline)
    if not roots:
        raise UnsupportedSimulation("no legal Phase 3 root actions")

    horizon = min(config.max_horizon, state.remaining_night_turns)
    initial_controller_ids = frozenset(
        role.unit_id
        for role in state.roles
        if role.assigned_weapon_id is not None
    )
    initial_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    evaluated: list[tuple[RootAction, RobotWaveSafetyCertificate]] = []

    for root in roots:
        _check_deadline(clock, effective_deadline)
        outcomes: list[ScenarioOutcome] = []
        for policy, weight in zip(ALL_ROBOT_POLICIES, scenario_weights):
            _check_deadline(clock, effective_deadline)
            simulated = state
            for step_index in range(horizon):
                _check_deadline(clock, effective_deadline)
                simulated_action = (
                    root.simulation_action
                    if step_index == 0
                    else choose_future_action(simulated, config)
                )
                simulated = step_simulation(
                    simulated,
                    simulated_action,
                    policy,
                    config,
                )
            outcomes.append(
                _scenario_outcome(
                    simulated,
                    weight,
                    initial_controller_ids,
                    initial_weapon_ids,
                )
            )
        evaluated.append(
            (
                root,
                build_certificate(tuple(outcomes), objective),
            )
        )
        _check_deadline(clock, effective_deadline)

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
    _check_deadline(clock, effective_deadline)
    root, certificate = winner
    return SearchResult(
        decision=root.decision,
        simulation_action=root.simulation_action,
        certificate=certificate,
        stats=SearchStats(
            roots_generated=len(roots),
            roots_evaluated=len(evaluated),
            scenarios_per_root=len(ALL_ROBOT_POLICIES),
            maximum_steps=horizon,
        ),
    )


def _scenario_outcome(
    state: SimState,
    weight: Fraction,
    initial_controller_ids: frozenset[int],
    initial_weapon_ids: frozenset[int],
) -> ScenarioOutcome:
    surviving_role_ids = frozenset(role.unit_id for role in state.roles)
    surviving_weapon_ids = frozenset(weapon.unit_id for weapon in state.weapons)
    return ScenarioOutcome(
        weight=weight,
        station_health=state.station.health,
        controller_losses=len(initial_controller_ids - surviving_role_ids),
        key_weapon_losses=len(initial_weapon_ids - surviving_weapon_ids),
        minimum_role_health=min(
            (role.health for role in state.roles),
            default=0,
        ),
        surviving_asset_health=(
            sum(wall.health for wall in state.walls)
            + sum(weapon.health for weapon in state.weapons)
        ),
        owned_kill_score=state.owned_kill_score,
        remaining_threat=sum(
            robot.attack_power * robot.health for robot in state.robots
        ),
        remaining_one_turn_damage=sum(
            robot.attack_power for robot in state.robots
        ),
        ended_with_night=state.remaining_night_turns == 0,
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
