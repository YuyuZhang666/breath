from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache

from future_war_agent.protocol.models import Position

from .state import SimRobot, SimState


class RobotPolicy(StrEnum):
    STATION_SHORTEST_PATH = "station_shortest_path"
    MAIN_PATH_BLOCKER = "main_path_blocker"
    LOW_HEALTH_BLOCKER = "low_health_blocker"
    MAXIMUM_STATION_PROGRESS = "maximum_station_progress"


ALL_ROBOT_POLICIES = tuple(RobotPolicy)


@dataclass(frozen=True, slots=True)
class RobotIntent:
    robot_id: int
    move_target: Position | None = None
    attack_target_kind: str | None = None
    attack_target_id: int | None = None


@dataclass(frozen=True, slots=True)
class _AttackTarget:
    kind: str
    target_id: int
    cells: frozenset[Position]
    health: int

    @property
    def stable_position(self) -> Position:
        return min(self.cells, key=lambda cell: (cell.x, cell.y))


_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_DEADLINE_CHECK: ContextVar[Callable[[], None] | None] = ContextVar(
    'robot_deadline_check',
    default=None,
)


def _deadline_checkpoint() -> None:
    check = _DEADLINE_CHECK.get()
    if check is not None:
        check()


def choose_robot_intents(
    state: SimState,
    policy: RobotPolicy,
    *,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[RobotIntent, ...]:
    token = _DEADLINE_CHECK.set(deadline_check)
    try:
        _deadline_checkpoint()
        if state.station.health <= 0:
            return ()
        navigation_state = _navigation_state(state)
        targets = _attack_targets(navigation_state)
        intents: list[RobotIntent] = []
        for robot in sorted(
            navigation_state.robots,
            key=lambda item: item.robot_id,
        ):
            _deadline_checkpoint()
            if robot.health > 0:
                intents.append(
                    _choose_robot_intent(
                        navigation_state,
                        robot,
                        targets,
                        policy,
                    )
                )
        return tuple(intents)
    finally:
        _DEADLINE_CHECK.reset(token)


def _navigation_state(state: SimState) -> SimState:
    if (
        state.round_no == 0
        and state.remaining_night_turns == 0
        and state.owned_kill_score == 0
    ):
        return state
    return SimState(
        round_no=0,
        width=state.width,
        height=state.height,
        remaining_night_turns=0,
        team_type=state.team_type,
        static_blocked=state.static_blocked,
        roles=state.roles,
        station=state.station,
        walls=state.walls,
        weapons=state.weapons,
        robots=state.robots,
        owned_kill_score=0,
    )


def _choose_robot_intent(
    state: SimState,
    robot: SimRobot,
    targets: tuple[_AttackTarget, ...],
    policy: RobotPolicy,
) -> RobotIntent:
    _deadline_checkpoint()
    if robot.waits_this_turn:
        return RobotIntent(robot.robot_id)

    candidate_route_cells = _candidate_route_cells(state, robot, targets)
    legal_targets = tuple(
        target
        for target in targets
        if min(
            robot.position.chebyshev_distance(cell) for cell in target.cells
        )
        <= robot.attack_range
        and bool(target.cells & candidate_route_cells)
    )
    if legal_targets and policy is RobotPolicy.MAXIMUM_STATION_PROGRESS:
        return _maximum_progress_intent(state, robot, legal_targets)
    if legal_targets:
        route = _ideal_station_route(state, robot)
        route_index = {
            cell: index for index, cell in enumerate(route)
        }

        def key(target: _AttackTarget) -> tuple[object, ...]:
            distance = min(
                robot.position.chebyshev_distance(cell)
                for cell in target.cells
            )
            index = min(
                (route_index[cell] for cell in target.cells if cell in route_index),
                default=state.width * state.height,
            )
            impact = len(target.cells & candidate_route_cells)
            stable = (
                target.kind,
                target.target_id,
                target.stable_position.x,
                target.stable_position.y,
            )
            if policy is RobotPolicy.STATION_SHORTEST_PATH:
                return (index, distance, stable)
            if policy is RobotPolicy.LOW_HEALTH_BLOCKER:
                return (target.health, -impact, distance, stable)
            return (-impact, index, distance, stable)

        selected = min(legal_targets, key=key)
        return RobotIntent(
            robot_id=robot.robot_id,
            attack_target_kind=selected.kind,
            attack_target_id=selected.target_id,
        )

    move_target = _best_move(state, robot)
    return RobotIntent(robot.robot_id, move_target=move_target)


def _maximum_progress_intent(
    state: SimState,
    robot: SimRobot,
    targets: tuple[_AttackTarget, ...],
) -> RobotIntent:
    _deadline_checkpoint()
    baseline = _station_distance(state, robot.position, robot, None)
    candidates: list[tuple[tuple[object, ...], RobotIntent]] = []

    for target in targets:
        _deadline_checkpoint()
        damage_fraction = Fraction(
            min(robot.attack_power, target.health),
            target.health,
        )
        expected_progress = (
            state.width * state.height
            if target.kind == 'station'
            else damage_fraction
        )
        stable = (
            target.kind,
            target.target_id,
            target.stable_position.x,
            target.stable_position.y,
        )
        candidates.append(
            (
                (-expected_progress, "attack", stable),
                RobotIntent(
                    robot_id=robot.robot_id,
                    attack_target_kind=target.kind,
                    attack_target_id=target.target_id,
                ),
            )
        )

    unreachable = state.width * state.height
    for target in _legal_moves(state, robot):
        _deadline_checkpoint()
        after = _station_distance(state, target, robot, None)
        if after is None:
            progress = -unreachable
        elif baseline is None:
            progress = unreachable - after
        else:
            progress = baseline - after
        candidates.append(
            (
                (-progress, "move", target.x, target.y),
                RobotIntent(robot.robot_id, move_target=target),
            )
        )

    if not candidates:
        return RobotIntent(robot.robot_id)
    return min(candidates, key=lambda item: item[0])[1]


def _legal_moves(state: SimState, robot: SimRobot) -> tuple[Position, ...]:
    _deadline_checkpoint()
    occupied = _dynamic_occupied(state) - {robot.position}
    moves: list[Position] = []
    for delta_x, delta_y in _DIRECTIONS:
        _deadline_checkpoint()
        target = Position(
            robot.position.x + delta_x,
            robot.position.y + delta_y,
        )
        if (
            _in_bounds(state, target)
            and target not in state.static_blocked
            and target not in occupied
        ):
            moves.append(target)
    return tuple(sorted(moves, key=lambda item: (item.x, item.y)))


def _attack_targets(state: SimState) -> tuple[_AttackTarget, ...]:
    _deadline_checkpoint()
    targets: list[_AttackTarget] = [
        _AttackTarget(
            "role",
            role.unit_id,
            frozenset({role.position}),
            role.health,
        )
        for role in state.roles
        if role.health > 0
    ]
    _deadline_checkpoint()
    targets.append(
        _AttackTarget(
            "station",
            state.station.unit_id,
            state.station.occupied_cells,
            state.station.health,
        )
    )
    _deadline_checkpoint()
    targets.extend(
        _AttackTarget("wall", wall.unit_id, wall.occupied_cells, wall.health)
        for wall in state.walls
        if wall.health > 0
    )
    _deadline_checkpoint()
    targets.extend(
        _AttackTarget(
            "weapon",
            weapon.unit_id,
            frozenset({weapon.position}),
            weapon.health,
        )
        for weapon in state.weapons
        if weapon.health > 0
    )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                target.kind,
                target.target_id,
                target.stable_position.x,
                target.stable_position.y,
            ),
        )
    )


def _best_move(state: SimState, robot: SimRobot) -> Position | None:
    _deadline_checkpoint()
    options: list[tuple[int, int, int, int, Position]] = []
    for target in _legal_moves(state, robot):
        _deadline_checkpoint()
        distance = _station_distance(state, target, robot, None)
        fallback = min(
            target.chebyshev_distance(cell)
            for cell in state.station.occupied_cells
        )
        options.append(
            (
                distance if distance is not None else state.width * state.height,
                fallback,
                target.x,
                target.y,
                target,
            )
        )
    if not options:
        return None
    return min(options, key=lambda item: item[:-1])[-1]


@lru_cache(maxsize=4096)
def _station_distance(
    state: SimState,
    start: Position,
    robot: SimRobot,
    removed: _AttackTarget | None,
) -> int | None:
    return _station_distance_map(state, robot, removed).get(start)


@lru_cache(maxsize=4096)
def _station_distance_map(
    state: SimState,
    robot: SimRobot,
    removed: _AttackTarget | None,
) -> dict[Position, int]:
    _deadline_checkpoint()
    blocked = set(state.static_blocked)
    blocked.update(_dynamic_occupied(state))
    blocked.discard(robot.position)
    if removed is not None:
        blocked.difference_update(removed.cells)

    goals_set: set[Position] = set()
    for x in range(state.width):
        _deadline_checkpoint()
        for y in range(state.height):
            position = Position(x, y)
            if (
                position not in blocked
                and min(
                    position.chebyshev_distance(cell)
                    for cell in state.station.occupied_cells
                )
                <= robot.attack_range
            ):
                goals_set.add(position)
    goals = frozenset(goals_set)
    return _distance_map(
        state.width,
        state.height,
        goals,
        frozenset(blocked),
    )


@lru_cache(maxsize=4096)
def _candidate_route_cells(
    state: SimState,
    robot: SimRobot,
    targets: tuple[_AttackTarget, ...],
) -> frozenset[Position]:
    _deadline_checkpoint()
    attackable_cells = set().union(*(target.cells for target in targets))
    blocked = set(state.static_blocked) - attackable_cells
    blocked.update(
        other.position
        for other in state.robots
        if other.health > 0 and other.robot_id != robot.robot_id
    )
    from_robot = _distance_map(
        state.width,
        state.height,
        frozenset({robot.position}),
        frozenset(blocked),
    )
    _deadline_checkpoint()
    from_station = _distance_map(
        state.width,
        state.height,
        state.station.occupied_cells,
        frozenset(blocked),
    )
    shortest = min(
        (
            from_robot[cell]
            for cell in state.station.occupied_cells
            if cell in from_robot
        ),
        default=None,
    )
    if shortest is None:
        return frozenset()
    return frozenset(
        cell
        for cell, start_distance in from_robot.items()
        if cell in from_station
        and start_distance + from_station[cell] == shortest
    )


@lru_cache(maxsize=4096)
def _distance_map(
    width: int,
    height: int,
    starts: frozenset[Position],
    blocked: frozenset[Position],
) -> dict[Position, int]:
    _deadline_checkpoint()
    valid_starts = tuple(
        sorted(
            (
                cell
                for cell in starts
                if 0 <= cell.x < width and 0 <= cell.y < height
            ),
            key=lambda cell: (cell.x, cell.y),
        )
    )
    distances = {cell: 0 for cell in valid_starts}
    queue = deque(valid_starts)
    while queue:
        _deadline_checkpoint()
        current = queue.popleft()
        for delta_x, delta_y in _DIRECTIONS:
            _deadline_checkpoint()
            neighbor = Position(current.x + delta_x, current.y + delta_y)
            if (
                neighbor in distances
                or not (0 <= neighbor.x < width and 0 <= neighbor.y < height)
                or neighbor in blocked
            ):
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


@lru_cache(maxsize=4096)
def _ideal_station_route(
    state: SimState,
    robot: SimRobot,
) -> tuple[Position, ...]:
    _deadline_checkpoint()
    attackable_cells = set().union(
        *(target.cells for target in _attack_targets(state))
    )
    terrain = set(state.static_blocked) - attackable_cells
    terrain.update(
        other.position
        for other in state.robots
        if other.health > 0 and other.robot_id != robot.robot_id
    )
    return _bfs_path(
        state,
        robot.position,
        state.station.occupied_cells,
        frozenset(terrain),
    )


@lru_cache(maxsize=8192)
def _bfs_path(
    state: SimState,
    start: Position,
    goals: frozenset[Position],
    blocked: frozenset[Position],
) -> tuple[Position, ...]:
    _deadline_checkpoint()
    if start in goals:
        return (start,)
    queue = deque([start])
    parent: dict[Position, Position | None] = {start: None}
    found: Position | None = None
    while queue and found is None:
        _deadline_checkpoint()
        current = queue.popleft()
        for delta_x, delta_y in _DIRECTIONS:
            _deadline_checkpoint()
            neighbor = Position(current.x + delta_x, current.y + delta_y)
            if (
                neighbor in parent
                or not _in_bounds(state, neighbor)
                or neighbor in blocked
            ):
                continue
            parent[neighbor] = current
            if neighbor in goals:
                found = neighbor
                break
            queue.append(neighbor)
    if found is None:
        return ()
    reversed_path = [found]
    while parent[reversed_path[-1]] is not None:
        _deadline_checkpoint()
        reversed_path.append(parent[reversed_path[-1]])
    return tuple(reversed(reversed_path))


def _dynamic_occupied(state: SimState) -> set[Position]:
    occupied = {
        role.position for role in state.roles if role.health > 0
    }
    occupied.update(
        robot.position for robot in state.robots if robot.health > 0
    )
    return occupied


def _in_bounds(state: SimState, position: Position) -> bool:
    return 0 <= position.x < state.width and 0 <= position.y < state.height
