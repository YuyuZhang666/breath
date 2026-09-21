from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from time import monotonic

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.policy import StrategyProfile

from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .errors import DeadlineExceeded
from .state import SimRobot, SimState, SimStructure
from .weapons import weapon_damage


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


@dataclass(frozen=True, slots=True)
class TailEstimate:
    incoming_damage: int
    effective_hp: int
    future_firepower: int
    survival_margin: int
    expected_station_hp_at_dawn: int
    expected_wall_losses: int
    expected_weapon_losses: int
    expected_role_losses: int
    lethal_round: int | None
    critical_robot_ids: tuple[int, ...]
    critical_wall_ids: tuple[int, ...]
    complete: bool
    uncertainty_reasons: tuple[str, ...]


def estimate_tail(
    state: SimState,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    *,
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
) -> TailEstimate:
    def check_deadline() -> None:
        if deadline is not None and clock() >= deadline:
            raise DeadlineExceeded('TailEstimator deadline expired')

    check_deadline()
    critical_walls = _minimum_barrier_walls(state, check_deadline)
    contribution_items: list[tuple[int, int, SimRobot]] = []
    for robot in state.robots:
        check_deadline()
        if robot.health > 0:
            contribution_items.append(
                (_robot_damage_before_dawn(state, robot), robot.robot_id, robot)
            )
    contributions = tuple(
        sorted(contribution_items, key=lambda item: (-item[0], item[1]))
    )
    multiplier = (
        config.tail_late_wave_damage_multiplier
        if state.day_no >= 3
        else Fraction(1)
    )
    incoming_damage = _ceil_fraction(
        sum(item[0] for item in contributions) * multiplier
    )
    effective_hp = state.station.health + sum(
        wall.health for wall in critical_walls
    )
    future_firepower, killed_robot_ids = _guaranteed_prevented_damage(
        state,
        config,
        check_deadline,
    )
    future_firepower = min(incoming_damage, future_firepower)
    pressure_after_fire = max(0, incoming_damage - future_firepower)
    wall_absorption = sum(wall.health for wall in critical_walls)
    station_damage = max(0, pressure_after_fire - wall_absorption)
    expected_station_hp = max(0, state.station.health - station_damage)
    survival_margin = effective_hp + future_firepower - incoming_damage
    lethal_round = _lethal_round(
        state,
        contributions,
        effective_hp,
        multiplier,
        killed_robot_ids,
        check_deadline,
    )
    if state.station.health <= 0:
        lethal_round = state.round_no

    uncertainty: list[str] = []
    if state.remaining_night_turns > 0 and not config.tail_visible_roster_complete:
        uncertainty.append('future_robot_roster_unconfirmed')
    if (
        state.remaining_night_turns > 0
        and state.day_no >= 3
        and not config.tail_late_wave_calibration_source.strip()
    ):
        uncertainty.append('d3_calibration_unavailable')

    return TailEstimate(
        incoming_damage=incoming_damage,
        effective_hp=effective_hp,
        future_firepower=future_firepower,
        survival_margin=survival_margin,
        expected_station_hp_at_dawn=expected_station_hp,
        expected_wall_losses=_expected_wall_losses(
            critical_walls,
            pressure_after_fire,
            check_deadline,
        ),
        # There is no verified rule for allocating analytic tail damage to
        # these assets. Exact-prefix losses remain authoritative.
        expected_weapon_losses=0,
        expected_role_losses=0,
        lethal_round=lethal_round,
        critical_robot_ids=tuple(
            robot_id
            for damage, robot_id, _ in contributions
            if damage > 0
        )[:8],
        critical_wall_ids=tuple(wall.unit_id for wall in critical_walls),
        complete=not uncertainty,
        uncertainty_reasons=tuple(uncertainty),
    )


def _robot_attack_turns(state: SimState, robot: SimRobot) -> int:
    distance = min(
        robot.position.chebyshev_distance(cell)
        for cell in state.station.occupied_cells
    )
    turns_to_attack = max(0, distance - robot.attack_range)
    if robot.waits_this_turn:
        turns_to_attack += 1
    return max(0, state.remaining_night_turns - turns_to_attack)


def _robot_damage_before_dawn(
    state: SimState,
    robot: SimRobot,
) -> int:
    return robot.attack_power * _robot_attack_turns(state, robot)


def _minimum_barrier_walls(
    state: SimState,
    deadline_check: Callable[[], None],
) -> tuple[SimStructure, ...]:
    if not state.robots or not state.walls:
        return ()
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for attack_range in sorted(
        {robot.attack_range for robot in state.robots if robot.health > 0}
    ):
        deadline_check()
        starts = frozenset(
            robot.position
            for robot in state.robots
            if robot.health > 0 and robot.attack_range == attack_range
        )
        barrier = _barrier_for_range(
            state,
            starts,
            attack_range,
            deadline_check,
        )
        if barrier is not None:
            candidates.append(barrier)
    if not candidates:
        return ()
    _, wall_ids = min(candidates, key=lambda item: (item[0], item[1]))
    walls_by_id = {wall.unit_id: wall for wall in state.walls}
    return tuple(walls_by_id[wall_id] for wall_id in wall_ids)


def _barrier_for_range(
    state: SimState,
    starts: frozenset[Position],
    attack_range: int,
    deadline_check: Callable[[], None],
) -> tuple[int, tuple[int, ...]] | None:
    wall_by_position = {wall.position: wall for wall in state.walls}
    removable = set(wall_by_position)
    removable.update(weapon.position for weapon in state.weapons)
    removable.update(state.station.occupied_cells)
    immutable_blocked = set(state.static_blocked) - removable
    goals: set[Position] = set()
    for x in range(state.width):
        deadline_check()
        for y in range(state.height):
            position = Position(x, y)
            if (
                position not in immutable_blocked
                and position not in state.station.occupied_cells
                and min(
                    position.chebyshev_distance(cell)
                    for cell in state.station.occupied_cells
                )
                <= attack_range
            ):
                goals.add(position)
    best: dict[Position, tuple[int, tuple[int, ...]]] = {}
    heap: list[tuple[int, tuple[int, ...], int, int, Position]] = []
    for start in sorted(starts, key=lambda cell: (cell.x, cell.y)):
        deadline_check()
        best[start] = (0, ())
        heappush(heap, (0, (), start.x, start.y, start))
    while heap:
        deadline_check()
        cost, wall_ids, _, _, current = heappop(heap)
        if best.get(current) != (cost, wall_ids):
            continue
        if current in goals:
            return cost, wall_ids
        for delta_x, delta_y in _DIRECTIONS:
            neighbor = Position(current.x + delta_x, current.y + delta_y)
            if (
                not (0 <= neighbor.x < state.width)
                or not (0 <= neighbor.y < state.height)
                or neighbor in immutable_blocked
                or neighbor in state.station.occupied_cells
            ):
                continue
            wall = wall_by_position.get(neighbor)
            next_cost = cost
            next_ids = wall_ids
            if wall is not None and wall.unit_id not in wall_ids:
                next_cost += wall.health
                next_ids = tuple(sorted((*wall_ids, wall.unit_id)))
            candidate = (next_cost, next_ids)
            if neighbor in best and best[neighbor] <= candidate:
                continue
            best[neighbor] = candidate
            heappush(
                heap,
                (next_cost, next_ids, neighbor.x, neighbor.y, neighbor),
            )
    return None


def _guaranteed_prevented_damage(
    state: SimState,
    config: Phase3Config,
    deadline_check: Callable[[], None],
) -> tuple[int, frozenset[int]]:
    if not state.robots or state.remaining_night_turns <= 0:
        return 0, frozenset()
    try:
        deadline_check()
        from future_war_agent.strategy.joint_fire import (
            choose_joint_fire_attacks,
        )

        selection = choose_joint_fire_attacks(
            state,
            StrategyProfile.SURVIVE,
            simulation_config=config,
            deadline_check=deadline_check,
        )
    except DeadlineExceeded:
        raise
    except Exception:
        return 0, frozenset()
    damage_by_robot: dict[int, int] = {}
    for attack in selection.attacks:
        deadline_check()
        for robot_id, damage in weapon_damage(state, attack, config).items():
            damage_by_robot[robot_id] = (
                damage_by_robot.get(robot_id, 0) + damage
            )
    killed = frozenset(
        robot.robot_id
        for robot in state.robots
        if damage_by_robot.get(robot.robot_id, 0) >= robot.health
    )
    prevented = sum(
        robot.attack_power * max(0, _robot_attack_turns(state, robot) - 1)
        for robot in state.robots
        if robot.robot_id in killed
    )
    return prevented, killed


def _expected_wall_losses(
    walls: tuple[SimStructure, ...],
    incoming_after_fire: int,
    deadline_check: Callable[[], None],
) -> int:
    remaining = incoming_after_fire
    losses = 0
    for wall in sorted(walls, key=lambda item: (item.health, item.unit_id)):
        deadline_check()
        if remaining < wall.health:
            break
        remaining -= wall.health
        losses += 1
    return losses


def _lethal_round(
    state: SimState,
    contributions: tuple[tuple[int, int, SimRobot], ...],
    capacity: int,
    multiplier: Fraction,
    killed_robot_ids: frozenset[int],
    deadline_check: Callable[[], None],
) -> int | None:
    if capacity <= 0:
        return state.round_no
    cumulative = 0
    for offset in range(1, state.remaining_night_turns + 1):
        deadline_check()
        raw_this_turn = 0
        for _, _, robot in contributions:
            if robot.robot_id in killed_robot_ids and offset > 1:
                continue
            distance = min(
                robot.position.chebyshev_distance(cell)
                for cell in state.station.occupied_cells
            )
            turns_to_attack = max(0, distance - robot.attack_range)
            if robot.waits_this_turn:
                turns_to_attack += 1
            if offset > turns_to_attack:
                raw_this_turn += robot.attack_power
        cumulative += _ceil_fraction(raw_this_turn * multiplier)
        if cumulative >= capacity:
            return state.round_no + offset
    return None


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)
