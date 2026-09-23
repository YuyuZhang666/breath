from fractions import Fraction
from threading import RLock

from future_war_agent.protocol.models import Observation

from .night import assign_controllers
from .session import StrategySession, observation_fingerprint
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .simulation.errors import UnsupportedSimulation
from .simulation.kernel import step_simulation
from .simulation.robots import ALL_ROBOT_POLICIES
from .simulation.search import ScenarioWeights
from .simulation.state import AssignedStand, SimState, build_sim_state
from .world import WorldGrid


class ScenarioStateCache:
    def __init__(self, capacity: int = 4) -> None:
        if capacity < 2:
            raise ValueError('scenario state cache capacity must be at least two')
        self._capacity = capacity
        self._entries: dict[tuple[str, Phase3Config], SimState] = {}
        self._lock = RLock()

    def resolve(
        self,
        observation: Observation,
        config: Phase3Config,
        *,
        world: WorldGrid | None = None,
    ) -> tuple[SimState, bool]:
        key = (observation_fingerprint(observation), config)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached, True
        state = _state_for(observation, config, world=world)
        with self._lock:
            self._entries[key] = state
            while len(self._entries) > self._capacity:
                del self._entries[next(iter(self._entries))]
        return state, False


def uniform_scenario_weights() -> ScenarioWeights:
    return (
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
    )


def reconcile_scenario_weights(
    previous: StrategySession,
    current: Observation,
    *,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    state_cache: ScenarioStateCache | None = None,
    current_world: WorldGrid | None = None,
) -> ScenarioWeights:
    if previous.simulation_action is None:
        return previous.scenario_weights
    try:
        if state_cache is None:
            previous_state = _state_for(previous.observation, config)
            current_state = _state_for(current, config, world=current_world)
        else:
            previous_state, _ = state_cache.resolve(previous.observation, config)
            current_state, _ = state_cache.resolve(
                current, config, world=current_world,
            )
        predictions = tuple(
            step_simulation(
                previous_state,
                previous.simulation_action,
                policy,
                config,
            )
            for policy in ALL_ROBOT_POLICIES
        )
    except UnsupportedSimulation:
        return previous.scenario_weights

    losses: list[int] = []
    for predicted in predictions:
        loss, comparable = _reconciliation_loss(predicted, current_state)
        if not comparable:
            return previous.scenario_weights
        losses.append(loss)
    return _weights_from_losses(
        previous.scenario_weights,
        tuple(losses),
        config.scenario_weight_floor,
    )


def _state_for(
    observation: Observation,
    config: Phase3Config,
    *,
    world: WorldGrid | None = None,
) -> SimState:
    world = WorldGrid.from_observation(observation) if world is None else world
    assignments = tuple(
        AssignedStand(item.role_id, item.weapon_id, item.stand)
        for item in assign_controllers(observation, world)
    )
    return build_sim_state(observation, assignments, config)


def _reconciliation_loss(
    predicted: SimState,
    current: SimState,
) -> tuple[int, bool]:
    loss = 0
    comparable = False

    predicted_robots = {robot.robot_id: robot for robot in predicted.robots}
    current_robots = {robot.robot_id: robot for robot in current.robots}
    for robot_id in sorted(predicted_robots.keys() | current_robots.keys()):
        predicted_robot = predicted_robots.get(robot_id)
        current_robot = current_robots.get(robot_id)
        comparable = True
        if predicted_robot is None or current_robot is None:
            loss += 20
        else:
            loss += predicted_robot.position.chebyshev_distance(
                current_robot.position
            )

    for predicted_items, current_items in (
        (predicted.roles, current.roles),
        (predicted.walls, current.walls),
        (predicted.weapons, current.weapons),
    ):
        predicted_by_id = {item.unit_id: item for item in predicted_items}
        current_by_id = {item.unit_id: item for item in current_items}
        for unit_id in sorted(predicted_by_id.keys() & current_by_id.keys()):
            comparable = True
            loss += abs(
                predicted_by_id[unit_id].health - current_by_id[unit_id].health
            )

    if predicted.station.unit_id == current.station.unit_id:
        comparable = True
        loss += abs(predicted.station.health - current.station.health)
    return loss, comparable


def _weights_from_losses(
    old_weights: ScenarioWeights,
    losses: tuple[int, int, int, int],
    floor: Fraction,
) -> ScenarioWeights:
    if len(old_weights) != 4 or len(losses) != 4:
        raise ValueError("four scenario values are required")
    if any(weight <= 0 for weight in old_weights):
        raise ValueError("old scenario weights must be positive")
    if sum(old_weights, Fraction()) != 1:
        raise ValueError("old scenario weights must sum to one")
    if any(loss < 0 for loss in losses):
        raise ValueError("scenario losses cannot be negative")
    if floor <= 0 or floor * 4 >= 1:
        raise ValueError("scenario floor must reserve less than total mass")

    raw = tuple(
        weight / (1 + loss) for weight, loss in zip(old_weights, losses)
    )
    raw_total = sum(raw, Fraction())
    distributable = 1 - floor * 4
    normalized = tuple(
        floor + distributable * value / raw_total for value in raw
    )
    if sum(normalized, Fraction()) != 1:
        raise AssertionError("scenario weights must normalize exactly")
    return normalized
