from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

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


def choose_robot_intents(
    state: SimState,
    policy: RobotPolicy,
) -> tuple[RobotIntent, ...]:
    if state.station.health <= 0:
        return ()
    targets = _attack_targets(state)
    return tuple(
        _choose_robot_intent(state, robot, targets, policy)
        for robot in sorted(state.robots, key=lambda item: item.robot_id)
        if robot.health > 0
    )


def _choose_robot_intent(
    state: SimState,
    robot: SimRobot,
    targets: tuple[_AttackTarget, ...],
    policy: RobotPolicy,
) -> RobotIntent:
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
        baseline = _station_distance(state, robot.position, robot, None)

        def key(target: _AttackTarget) -> tuple[object, ...]:
            distance = min(
                robot.position.chebyshev_distance(cell)
                for cell in target.cells
            )
            index = min(
                (route_index[cell] for cell in target.cells if cell in route_index),
                default=state.width * state.height,
            )
            impact = _path_impact(state, robot, target, baseline)
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
    baseline = _station_distance(state, robot.position, robot, None)
    candidates: list[tuple[tuple[object, ...], RobotIntent]] = []

    for target in targets:
        path_impact = _path_impact(state, robot, target, baseline)
        damage_fraction = Fraction(
            min(robot.attack_power, target.health),
            target.health,
        )
        expected_progress = path_impact * damage_fraction
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
    occupied = _dynamic_occupied(state) - {robot.position}
    moves: list[Position] = []
    for delta_x, delta_y in _DIRECTIONS:
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
    targets.append(
        _AttackTarget(
            "station",
            state.station.unit_id,
            state.station.occupied_cells,
            state.station.health,
        )
    )
    targets.extend(
        _AttackTarget("wall", wall.unit_id, wall.occupied_cells, wall.health)
        for wall in state.walls
        if wall.health > 0
    )
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
    options: list[tuple[int, int, int, int, Position]] = []
    for target in _legal_moves(state, robot):
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


def _path_impact(
    state: SimState,
    robot: SimRobot,
    target: _AttackTarget,
    baseline: int | None,
) -> int:
    if target.kind == "station":
        return state.width * state.height
    after = _station_distance(state, robot.position, robot, target)
    if after is None:
        return 0
    if baseline is None:
        return state.width * state.height - after
    return max(0, baseline - after)


def _station_distance(
    state: SimState,
    start: Position,
    robot: SimRobot,
    removed: _AttackTarget | None,
) -> int | None:
    blocked = set(state.static_blocked)
    blocked.update(_dynamic_occupied(state))
    blocked.discard(robot.position)
    blocked.discard(start)
    if removed is not None:
        blocked.difference_update(removed.cells)

    goals = {
        Position(x, y)
        for x in range(state.width)
        for y in range(state.height)
        if Position(x, y) not in blocked
        and min(
            Position(x, y).chebyshev_distance(cell)
            for cell in state.station.occupied_cells
        )
        <= robot.attack_range
    }
    return _bfs_distance(state, start, goals, frozenset(blocked))


def _candidate_route_cells(
    state: SimState,
    robot: SimRobot,
    targets: tuple[_AttackTarget, ...],
) -> frozenset[Position]:
    attackable_cells = set().union(*(target.cells for target in targets))
    blocked = set(state.static_blocked) - attackable_cells
    blocked.update(
        other.position
        for other in state.robots
        if other.health > 0 and other.robot_id != robot.robot_id
    )
    from_robot = _distance_map(
        state,
        frozenset({robot.position}),
        frozenset(blocked),
    )
    from_station = _distance_map(
        state,
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


def _distance_map(
    state: SimState,
    starts: frozenset[Position],
    blocked: frozenset[Position],
) -> dict[Position, int]:
    valid_starts = tuple(
        sorted(
            (cell for cell in starts if _in_bounds(state, cell)),
            key=lambda cell: (cell.x, cell.y),
        )
    )
    distances = {cell: 0 for cell in valid_starts}
    queue = deque(valid_starts)
    while queue:
        current = queue.popleft()
        for delta_x, delta_y in _DIRECTIONS:
            neighbor = Position(current.x + delta_x, current.y + delta_y)
            if (
                neighbor in distances
                or not _in_bounds(state, neighbor)
                or neighbor in blocked
            ):
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _ideal_station_route(
    state: SimState,
    robot: SimRobot,
) -> tuple[Position, ...]:
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


def _bfs_distance(
    state: SimState,
    start: Position,
    goals: set[Position],
    blocked: frozenset[Position],
) -> int | None:
    path = _bfs_path(state, start, frozenset(goals), blocked)
    return len(path) - 1 if path else None


def _bfs_path(
    state: SimState,
    start: Position,
    goals: frozenset[Position],
    blocked: frozenset[Position],
) -> tuple[Position, ...]:
    if start in goals:
        return (start,)
    queue = deque([start])
    parent: dict[Position, Position | None] = {start: None}
    found: Position | None = None
    while queue and found is None:
        current = queue.popleft()
        for delta_x, delta_y in _DIRECTIONS:
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
