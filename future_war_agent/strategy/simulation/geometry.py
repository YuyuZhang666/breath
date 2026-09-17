from fractions import Fraction
from itertools import combinations

from future_war_agent.protocol.models import Position


Ray = tuple[Position, ...]


def bresenham_cells(start: Position, end: Position) -> Ray:
    if start == end:
        return ()

    x = start.x
    y = start.y
    delta_x = abs(end.x - start.x)
    step_x = 1 if start.x < end.x else -1
    delta_y = -abs(end.y - start.y)
    step_y = 1 if start.y < end.y else -1
    error = delta_x + delta_y
    cells: list[Position] = []

    while x != end.x or y != end.y:
        doubled_error = 2 * error
        if doubled_error >= delta_y:
            error += delta_y
            x += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y += step_y
        cells.append(Position(x, y))
    return tuple(cells)


def supercover_cells(start: Position, end: Position) -> Ray:
    return _intersected_cells(start, end, include_touches=True)


def center_intersection_cells(start: Position, end: Position) -> Ray:
    return _intersected_cells(start, end, include_touches=False)


def is_legal_cone(origin: Position, targets: tuple[Position, ...]) -> bool:
    vectors = tuple(
        (target.x - origin.x, target.y - origin.y) for target in targets
    )
    if any(delta_x == 0 and delta_y == 0 for delta_x, delta_y in vectors):
        return False
    return all(
        left_x * right_x + left_y * right_y >= 0
        for (left_x, left_y), (right_x, right_y) in combinations(vectors, 2)
    )


def _intersected_cells(
    start: Position,
    end: Position,
    *,
    include_touches: bool,
) -> Ray:
    if start == end:
        return ()

    hits: list[tuple[Fraction, Position]] = []
    for x in range(min(start.x, end.x), max(start.x, end.x) + 1):
        for y in range(min(start.y, end.y), max(start.y, end.y) + 1):
            position = Position(x, y)
            if position == start:
                continue
            interval = _cell_interval(start, end, position)
            if interval is None:
                continue
            entry, exit_ = interval
            if include_touches and entry <= exit_:
                hits.append((entry, position))
            elif not include_touches and entry < exit_:
                hits.append((entry, position))

    hits.sort(key=lambda item: (item[0], item[1].x, item[1].y))
    return tuple(position for _, position in hits)


def _cell_interval(
    start: Position,
    end: Position,
    cell: Position,
) -> tuple[Fraction, Fraction] | None:
    entry = Fraction(0)
    exit_ = Fraction(1)
    for start_value, end_value, cell_value in (
        (start.x, end.x, cell.x),
        (start.y, end.y, cell.y),
    ):
        delta = end_value - start_value
        lower = Fraction(2 * cell_value - 1, 2)
        upper = Fraction(2 * cell_value + 1, 2)
        if delta == 0:
            if not lower <= start_value <= upper:
                return None
            continue
        first = (lower - start_value) / delta
        second = (upper - start_value) / delta
        axis_entry = min(first, second)
        axis_exit = max(first, second)
        entry = max(entry, axis_entry)
        exit_ = min(exit_, axis_exit)
        if entry > exit_:
            return None
    return entry, exit_
