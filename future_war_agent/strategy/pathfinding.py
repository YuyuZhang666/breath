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
) -> PathResult | None:
    valid_goals = frozenset(
        goal
        for goal in goals
        if world.in_bounds(goal)
        and (world.can_traverse(goal) or goal == start)
        and goal not in additional_blocked
    )
    if not valid_goals:
        return None

    order = count()
    frontier: list[tuple[int, int, int, int, int, Position]] = []
    heappush(
        frontier,
        (
            _heuristic(start, valid_goals),
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
        _, cost, _, _, _, current = heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current in valid_goals:
            path = _reconstruct(came_from, start, current)
            return PathResult(path=path, cost=len(path) - 1)

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
                    next_cost + _heuristic(neighbor, valid_goals),
                    next_cost,
                    neighbor.x,
                    neighbor.y,
                    next(order),
                    neighbor,
                ),
            )
    return None


def path_to_interaction(
    world: WorldGrid,
    start: Position,
    target: Position,
    *,
    additional_blocked: frozenset[Position] = frozenset(),
) -> PathResult | None:
    return shortest_path(
        world,
        start,
        world.interaction_cells(target),
        additional_blocked=additional_blocked,
    )


def first_step_options(
    world: WorldGrid,
    start: Position,
    goals: tuple[Position, ...],
    *,
    limit: int = 2,
    additional_blocked: frozenset[Position] = frozenset(),
) -> tuple[Position, ...]:
    if limit <= 0:
        return ()

    ranked: list[tuple[int, int, int, Position]] = []
    for dx, dy in _STEPS:
        neighbor = Position(start.x + dx, start.y + dy)
        if (
            not world.can_traverse(neighbor)
            or neighbor in additional_blocked
        ):
            continue
        remaining = shortest_path(
            world,
            neighbor,
            goals,
            additional_blocked=additional_blocked,
        )
        if remaining is None:
            continue
        ranked.append((1 + remaining.cost, neighbor.x, neighbor.y, neighbor))

    ranked.sort()
    result: list[Position] = []
    for _, _, _, position in ranked:
        if position not in result:
            result.append(position)
        if len(result) == limit:
            break
    return tuple(result)


def _heuristic(position: Position, goals: frozenset[Position]) -> int:
    return min(position.chebyshev_distance(goal) for goal in goals)


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
