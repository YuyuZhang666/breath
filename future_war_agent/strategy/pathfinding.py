from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from future_war_agent.protocol.models import Position

from .world import WorldGrid


_STEPS = tuple(
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)

# WorldGrid is rebuilt every round, so the per-instance distance cache dies
# with it; day maps stay geometrically identical for many consecutive rounds
# (walls appear a few times per day, robots only at night), so a small
# module-level cache keyed on the blocked set removes nearly every BFS.
_REVERSE_CACHE_CAPACITY = 64
_REVERSE_CACHE: dict[
    tuple[frozenset[Position], frozenset[Position], int, int],
    dict[Position, int],
] = {}


@dataclass(frozen=True, slots=True)
class PathResult:
    path: tuple[Position, ...]
    cost: int


def shortest_path(
    world: WorldGrid,
    start: Position,
    goals: tuple[Position, ...],
    *,
    additional_blocked: frozenset[Position] = frozenset(),
    deadline_check: Callable[[], None] | None = None,
) -> PathResult | None:
    _check_deadline(deadline_check)
    valid_goals = frozenset(
        goal
        for goal in goals
        if world.in_bounds(goal)
        and (world.can_traverse(goal) or goal == start)
        and goal not in additional_blocked
    )
    if not valid_goals:
        return None
    cache_key = (
        'shortest_path',
        start,
        valid_goals,
        additional_blocked,
    )
    if cache_key in world._path_cache:
        return world._path_cache[cache_key]  # type: ignore[return-value]

    if not additional_blocked and start not in world.hard_blocked:
        shared_key = (
            valid_goals,
            world.hard_blocked,
            world.observation.width,
            world.observation.height,
        )
        if shared_key in _REVERSE_CACHE:
            result = _path_from_reverse_distances(
                world, start, valid_goals, deadline_check,
            )
            world._path_cache[cache_key] = result
            return result

    heuristic_cache = world._heuristic_cache.setdefault(valid_goals, {})

    order = count()
    frontier: list[tuple[int, int, int, int, int, Position]] = []
    heappush(
        frontier,
        (
            _heuristic(start, valid_goals, heuristic_cache),
            0,
            start.x,
            start.y,
            next(order),
            start,
        ),
    )
    came_from: dict[Position, Position] = {}
    best_cost = {start: 0}
    while frontier:
        _check_deadline(deadline_check)
        _, cost, _, _, _, current = heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current in valid_goals:
            path = _reconstruct(came_from, start, current)
            result = PathResult(path=path, cost=len(path) - 1)
            world._path_cache[cache_key] = result
            return result

        for dx, dy in _STEPS:
            neighbor = Position(current.x + dx, current.y + dy)
            if neighbor != start and (
                not world.can_traverse(neighbor)
                or neighbor in additional_blocked
            ):
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(neighbor, next_cost + 1):
                continue
            best_cost[neighbor] = next_cost
            came_from[neighbor] = current
            heappush(
                frontier,
                (
                    next_cost
                    + _heuristic(neighbor, valid_goals, heuristic_cache),
                    next_cost,
                    neighbor.x,
                    neighbor.y,
                    next(order),
                    neighbor,
                ),
            )
    world._path_cache[cache_key] = None
    return None


def path_to_interaction(
    world: WorldGrid,
    start: Position,
    target: Position,
    *,
    additional_blocked: frozenset[Position] = frozenset(),
    deadline_check: Callable[[], None] | None = None,
) -> PathResult | None:
    return shortest_path(
        world,
        start,
        world.interaction_cells(target),
        additional_blocked=additional_blocked,
        deadline_check=deadline_check,
    )


def first_step_options(
    world: WorldGrid,
    start: Position,
    goals: tuple[Position, ...],
    *,
    limit: int = 2,
    additional_blocked: frozenset[Position] = frozenset(),
    deadline_check: Callable[[], None] | None = None,
) -> tuple[Position, ...]:
    _check_deadline(deadline_check)
    if limit <= 0:
        return ()

    valid_goals = frozenset(
        goal
        for goal in goals
        if world.in_bounds(goal)
        and (world.can_traverse(goal) or goal == start)
        and goal not in additional_blocked
    )
    if not valid_goals:
        return ()
    cache_key = (
        'first_step_options',
        start,
        valid_goals,
        limit,
        additional_blocked,
    )
    if cache_key in world._path_cache:
        return world._path_cache[cache_key]  # type: ignore[return-value]

    distances = _reverse_distances(
        world,
        valid_goals,
        additional_blocked,
        deadline_check,
    )

    ranked: list[tuple[int, int, int, Position]] = []
    for dx, dy in _STEPS:
        _check_deadline(deadline_check)
        neighbor = Position(start.x + dx, start.y + dy)
        if (
            not world.can_traverse(neighbor)
            or neighbor in additional_blocked
        ):
            continue
        remaining_cost = distances.get(neighbor)
        if remaining_cost is None:
            continue
        ranked.append((1 + remaining_cost, neighbor.x, neighbor.y, neighbor))

    ranked.sort()
    result: list[Position] = []
    for _, _, _, position in ranked:
        if position not in result:
            result.append(position)
        if len(result) == limit:
            break
    resolved = tuple(result)
    world._path_cache[cache_key] = resolved
    return resolved


def _heuristic(
    position: Position,
    goals: frozenset[Position],
    cache: dict[Position, int],
) -> int:
    cached = cache.get(position)
    if cached is not None:
        return cached
    distance = min(position.chebyshev_distance(goal) for goal in goals)
    cache[position] = distance
    return distance


def _path_from_reverse_distances(
    world: WorldGrid,
    start: Position,
    valid_goals: frozenset[Position],
    deadline_check: Callable[[], None] | None,
) -> PathResult | None:
    """Answer a goal query from the shared reverse-distance field.

    On the unweighted grid the BFS distance to the goal set equals the A*
    cost, and descending strictly-decreasing distances yields a shortest
    path, so the cached field replaces a per-query A* search. Neighbour
    ties resolve by (x, y) to match first_step_options ordering.
    """
    distances = _reverse_distances(
        world, valid_goals, frozenset(), deadline_check,
    )
    remaining = distances.get(start)
    if remaining is None:
        return None
    path = [start]
    current = start
    while remaining > 0:
        _check_deadline(deadline_check)
        best: Position | None = None
        for dx, dy in _STEPS:
            neighbor = Position(current.x + dx, current.y + dy)
            if distances.get(neighbor) == remaining - 1 and (
                best is None or (neighbor.x, neighbor.y)
                < (best.x, best.y)
            ):
                best = neighbor
        if best is None:
            return None
        path.append(best)
        current = best
        remaining -= 1
    return PathResult(path=tuple(path), cost=len(path) - 1)


def _reverse_distances(
    world: WorldGrid,
    goals: frozenset[Position],
    additional_blocked: frozenset[Position],
    deadline_check: Callable[[], None] | None,
) -> dict[Position, int]:
    cache_key = (goals, additional_blocked)
    cached = world._distance_cache.get(cache_key)
    if cached is not None:
        return cached

    blocked = world.hard_blocked | additional_blocked
    shared_key = (
        goals,
        blocked,
        world.observation.width,
        world.observation.height,
    )
    shared = _REVERSE_CACHE.get(shared_key)
    if shared is None:
        shared = _bfs_reverse_distances(world, goals, blocked, deadline_check)
        _REVERSE_CACHE[shared_key] = shared
        while len(_REVERSE_CACHE) > _REVERSE_CACHE_CAPACITY:
            del _REVERSE_CACHE[next(iter(_REVERSE_CACHE))]
    world._distance_cache[cache_key] = shared
    return shared


def _bfs_reverse_distances(
    world: WorldGrid,
    goals: frozenset[Position],
    blocked: frozenset[Position],
    deadline_check: Callable[[], None] | None,
) -> dict[Position, int]:
    # Integer keys (x * height + y) avoid millions of Position hashes in
    # the hot loop; out-of-bounds blocked positions are dropped because the
    # per-neighbour bounds check already rejects them.
    width = world.observation.width
    height = world.observation.height

    def cell_key(position: Position) -> int:
        return position.x * height + position.y

    blocked_keys = frozenset(
        cell_key(position)
        for position in blocked
        if 0 <= position.x < width and 0 <= position.y < height
    )
    distances: dict[int, int] = {}
    queue: deque[int] = deque()
    for goal in goals:
        key = cell_key(goal)
        if key not in distances:
            distances[key] = 0
            queue.append(key)
    while queue:
        _check_deadline(deadline_check)
        current = queue.popleft()
        next_cost = distances[current] + 1
        x, y = divmod(current, height)
        for dx, dy in _STEPS:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = nx * height + ny
                if neighbor in distances or neighbor in blocked_keys:
                    continue
                distances[neighbor] = next_cost
                queue.append(neighbor)
    return {
        Position(key // height, key % height): distance
        for key, distance in distances.items()
    }


def _check_deadline(
    deadline_check: Callable[[], None] | None,
) -> None:
    if deadline_check is not None:
        deadline_check()


def _reconstruct(
    came_from: dict[Position, Position],
    start: Position,
    goal: Position,
) -> tuple[Position, ...]:
    reversed_path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        reversed_path.append(current)
    reversed_path.reverse()
    return tuple(reversed_path)
