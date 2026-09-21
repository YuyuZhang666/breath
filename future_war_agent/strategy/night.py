from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, permutations, product
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import (
    Observation,
    Position,
    RobotState,
    UnitState,
)
from future_war_agent.protocol.time import Phase

from .jobs import Job, JobKind
from .items import has_item
from .joint import TacticalCandidate, candidates_for_jobs
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .rules import station_footprint
from .simulation.geometry import is_legal_cone
from .world import WorldGrid


_PERSONAL_ROLES = frozenset({'worker', 'pioneer'})
_EMERGENCY_WEAPON_LIMIT = 3
_EMERGENCY_ROBOT_LIMIT = 64


class _EmergencyDeadlineExpired(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ControllerAssignment:
    role_id: int
    weapon_id: int
    stand: Position
    distance: int


@dataclass(frozen=True, slots=True)
class EmergencyFirePlan:
    decision: Decision
    deadline_hit: bool
    weapons_considered: int


def plan_emergency_night_fire(
    observation: Observation,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_weapons: int = _EMERGENCY_WEAPON_LIMIT,
    max_robots: int = _EMERGENCY_ROBOT_LIMIT,
) -> EmergencyFirePlan:
    if max_weapons <= 0 or max_robots <= 0:
        raise ValueError('emergency fire limits must be positive')
    if observation.time.phase is not Phase.NIGHT:
        return EmergencyFirePlan(Decision(), False, 0)

    commands: dict[int, Action] = {}
    weapons_considered = 0

    def check_deadline() -> None:
        if clock() >= deadline:
            raise _EmergencyDeadlineExpired

    try:
        check_deadline()
        world = WorldGrid.from_observation(observation)
        check_deadline()
        station = world.our_station()
        station_cells = (
            station_footprint(station.position, world.rules)
            if station is not None
            else frozenset()
        )
        hostile: list[RobotState] = []
        for robot in observation.robots:
            check_deadline()
            if (
                robot.health > 0
                and robot.target_team == observation.our.team_type
            ):
                hostile.append(robot)
        hostile.sort(
            key=lambda robot: (
                min(
                    (
                        robot.position.chebyshev_distance(cell)
                        for cell in station_cells
                    ),
                    default=0,
                ),
                robot.health,
                robot.robot_id,
            )
        )
        hostile_robots = tuple(hostile[:max_robots])
        if not hostile_robots:
            return EmergencyFirePlan(Decision(), False, 0)

        controllers = tuple(
            sorted(
                (
                    role
                    for role in world.friendly_roles
                    if role.health > 0 and role.role_type in _PERSONAL_ROLES
                ),
                key=lambda role: role.unit_id,
            )
        )
        weapons = tuple(
            sorted(
                (
                    weapon
                    for weapon in world.weapons
                    if weapon.health > 0
                    and weapon.cooldown == 0
                    and weapon.attack_range > 0
                    and (weapon.level or 0) > 0
                    and weapon.role_type in {'gatling', 'railgun', 'rocket'}
                ),
                key=lambda weapon: (
                    -_assignment_weapon_value(weapon, 'survive'),
                    weapon.unit_id,
                ),
            )[:max_weapons]
        )
        used_controllers: set[int] = set()
        for weapon in weapons:
            check_deadline()
            weapons_considered += 1
            controller = None
            for role in controllers:
                check_deadline()
                if (
                    role.unit_id not in used_controllers
                    and role.position.chebyshev_distance(weapon.position) <= 1
                ):
                    controller = role
                    break
            if controller is None:
                continue
            targets = _phase2_attack_targets(
                observation,
                world,
                weapon,
                robot_candidates=hostile_robots,
                deadline_check=check_deadline,
            )
            if not targets:
                continue
            commands[weapon.unit_id] = Action.attack(
                controller.unit_id,
                targets,
            )
            used_controllers.add(controller.unit_id)
            check_deadline()
    except _EmergencyDeadlineExpired:
        return EmergencyFirePlan(
            Decision(commands=commands),
            True,
            weapons_considered,
        )
    return EmergencyFirePlan(
        Decision(commands=commands),
        False,
        weapons_considered,
    )


@dataclass(frozen=True, slots=True)
class _ControllerCacheEntry:
    signature: tuple[object, ...]
    assignments: tuple[ControllerAssignment, ...]


class ControllerAssignmentCache:
    def __init__(self) -> None:
        self._entries: dict[str, _ControllerCacheEntry] = {}
        self._lock = RLock()

    def resolve(
        self,
        observation: Observation,
        world: WorldGrid,
        *,
        mode_key: str,
        excluded_role_ids: frozenset[int] = frozenset(),
    ) -> tuple[tuple[ControllerAssignment, ...], bool]:
        team_id = observation.our.team_id.strip()
        signature = _assignment_signature(
            observation,
            world,
            mode_key,
            excluded_role_ids,
        )
        if team_id:
            with self._lock:
                cached = self._entries.get(team_id)
                if cached is not None and cached.signature == signature:
                    return cached.assignments, True

        assignments = assign_controllers(
            observation,
            world,
            mode_key=mode_key,
            excluded_role_ids=excluded_role_ids,
        )
        if team_id:
            with self._lock:
                self._entries[team_id] = _ControllerCacheEntry(
                    signature,
                    assignments,
                )
        return assignments, False


def _assignment_signature(
    observation: Observation,
    world: WorldGrid,
    mode_key: str,
    excluded_role_ids: frozenset[int],
) -> tuple[object, ...]:
    return (
        observation.width,
        observation.height,
        observation.our.team_type,
        mode_key,
        tuple(sorted(excluded_role_ids)),
        tuple(
            (role.unit_id, role.position.x, role.position.y)
            for role in world.friendly_roles
        ),
        tuple(
            (
                weapon.unit_id,
                weapon.position.x,
                weapon.position.y,
                weapon.health,
                weapon.attack_power,
                weapon.attack_range,
                weapon.level,
                weapon.cooldown,
            )
            for weapon in world.weapons
        ),
        tuple(
            sorted((position.x, position.y) for position in world.structure_cells)
        ),
    )


def assign_controllers(
    observation: Observation,
    world: WorldGrid,
    *,
    mode_key: str = 'economy',
    excluded_role_ids: frozenset[int] = frozenset(),
) -> tuple[ControllerAssignment, ...]:
    roles = tuple(
        sorted(
            (
                role
                for role in world.friendly_roles
                if role.unit_id not in excluded_role_ids
            ),
            key=lambda value: value.unit_id,
        )
    )
    weapons = tuple(sorted(world.weapons, key=lambda value: value.unit_id))
    distances_by_role = {
        role.unit_id: _distance_map(world, role.position) for role in roles
    }
    options_by_pair = {
        (role.unit_id, weapon.unit_id): _assignment_options(
            world,
            role,
            weapon,
            distances_by_role[role.unit_id],
        )
        for role in roles
        for weapon in weapons
    }
    for size in range(min(len(roles), len(weapons)), 0, -1):
        best: tuple[ControllerAssignment, ...] | None = None
        best_score: tuple[object, ...] | None = None
        for selected_weapons in combinations(weapons, size):
            for selected_roles in permutations(roles, size):
                pair_options: list[tuple[ControllerAssignment, ...]] = []
                for role, weapon in zip(selected_roles, selected_weapons):
                    options = options_by_pair[(role.unit_id, weapon.unit_id)]
                    if not options:
                        break
                    pair_options.append(options)
                else:
                    for possible in product(*pair_options):
                        if len({item.stand for item in possible}) != size:
                            continue
                        ordered = tuple(
                            sorted(possible, key=lambda value: value.role_id)
                        )
                        weapon_by_id = {
                            weapon.unit_id: weapon for weapon in selected_weapons
                        }
                        role_position_by_id = {
                            role.unit_id: role.position for role in selected_roles
                        }
                        ready_now = sum(
                            item.stand == role_position_by_id[item.role_id]
                            and weapon_by_id[item.weapon_id].cooldown == 0
                            and weapon_by_id[item.weapon_id].attack_range > 0
                            for item in ordered
                        )
                        attack_value = sum(
                            _assignment_weapon_value(
                                weapon_by_id[item.weapon_id],
                                mode_key,
                            )
                            for item in ordered
                        )
                        operator_risk = sum(
                            _operator_risk(observation, item.stand)
                            for item in ordered
                        )
                        score: tuple[object, ...] = (
                            -ready_now,
                            -attack_value,
                            sum(item.distance for item in ordered),
                            operator_risk,
                            tuple(
                                (
                                    item.role_id,
                                    item.weapon_id,
                                    item.stand.x,
                                    item.stand.y,
                                )
                                for item in ordered
                            ),
                        )
                        if best_score is None or score < best_score:
                            best = ordered
                            best_score = score
        if best is not None:
            return best
    return ()


def generate_night_candidates(
    observation: Observation,
    world: WorldGrid,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    *,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
) -> Mapping[int, tuple[TacticalCandidate, ...]]:
    assignments = {
        item.role_id: item
        for item in (
            assign_controllers(
                observation,
                world,
                mode_key=intent.profile.value,
            )
            if controller_assignments is None
            else controller_assignments
        )
    }
    choices: dict[int, tuple[TacticalCandidate, ...]] = {}
    for role in world.friendly_roles:
        assignment = assignments.get(role.unit_id)
        if assignment is not None:
            weapon = world.unit_by_id(assignment.weapon_id)
            role_choices = _assigned_candidates(
                observation,
                world,
                role,
                weapon,
                assignment,
                intent,
            )
        else:
            role_choices = _safe_candidates(observation, world, role)
        medicine = _medicine_candidate(role, intent)
        choices[role.unit_id] = (
            (medicine,) + role_choices if medicine is not None else role_choices
        )
    return MappingProxyType(choices)


def _medicine_candidate(
    role: UnitState,
    intent: StrategicIntent,
) -> TacticalCandidate | None:
    policy = intent.item_policy
    if (
        policy.medicine_health_threshold <= 0
        or role.health > policy.medicine_health_threshold
        or not has_item(role.backpack, policy.medicine_name)
    ):
        return None
    return TacticalCandidate.personal_action(
        role.unit_id,
        role.position,
        Action.use(policy.medicine_name),
        JobKind.USE_ITEM,
        intent.day_priorities.emergency_item,
    )


def _assignment_options(
    world: WorldGrid,
    role: UnitState,
    weapon: UnitState,
    distances: Mapping[Position, int] | None = None,
) -> tuple[ControllerAssignment, ...]:
    resolved_distances = (
        _distance_map(world, role.position) if distances is None else distances
    )
    options: list[ControllerAssignment] = []
    for stand in world.interaction_cells(weapon.position):
        distance = resolved_distances.get(stand)
        if distance is not None:
            options.append(
                ControllerAssignment(
                    role_id=role.unit_id,
                    weapon_id=weapon.unit_id,
                    stand=stand,
                    distance=distance,
                )
            )
    return tuple(
        sorted(
            options,
            key=lambda value: (
                value.distance,
                value.stand.x,
                value.stand.y,
            ),
        )
    )


def _assigned_candidates(
    observation: Observation,
    world: WorldGrid,
    role: UnitState,
    weapon: UnitState | None,
    assignment: ControllerAssignment,
    intent: StrategicIntent,
) -> tuple[TacticalCandidate, ...]:
    if role.position != assignment.stand:
        job = Job(
            role_id=role.unit_id,
            kind=JobKind.PREPOSITION,
            target=assignment.stand,
            priority=450,
            value=-float(assignment.distance),
            weapon_id=assignment.weapon_id,
        )
        return candidates_for_jobs(observation, world, role, (job,))

    candidates: list[TacticalCandidate] = []
    if weapon is not None:
        targets = _phase2_attack_targets(
            observation,
            world,
            weapon,
            allow_cross_team=(
                intent.feature_flags.enable_cross_team_robot_attack
            ),
        )
        if targets:
            candidates.append(
                TacticalCandidate.attack(
                    role_id=role.unit_id,
                    start=role.position,
                    weapon_id=weapon.unit_id,
                    targets=targets,
                    priority=500,
                )
            )
    candidates.append(TacticalCandidate.wait(role.unit_id, role.position))
    return tuple(candidates)


def _phase2_attack_targets(
    observation: Observation,
    world: WorldGrid,
    weapon: UnitState,
    *,
    allow_cross_team: bool = False,
    robot_candidates: tuple[RobotState, ...] | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[Position, ...]:
    if deadline_check is not None:
        deadline_check()
    if (
        weapon.cooldown != 0
        or weapon.attack_range <= 0
        or weapon.level is None
        or weapon.level <= 0
    ):
        return ()
    living_list: list[RobotState] = []
    for robot in (
        observation.robots if robot_candidates is None else robot_candidates
    ):
        if deadline_check is not None:
            deadline_check()
        if robot.health > 0 and (
            allow_cross_team
            or robot.target_team == observation.our.team_type
        ):
            living_list.append(robot)
    living = tuple(living_list)
    if not living:
        return ()
    if weapon.role_type == 'railgun':
        in_range_list: list[RobotState] = []
        for robot in living:
            if deadline_check is not None:
                deadline_check()
            if (
                weapon.position.chebyshev_distance(robot.position)
                <= weapon.attack_range
            ):
                in_range_list.append(robot)
        in_range = tuple(in_range_list)
        if not in_range:
            return ()
        target = min(
            in_range,
            key=lambda robot: _target_key(observation, world, weapon, robot),
        )
        return (target.position,)
    if weapon.role_type == 'gatling':
        return _gatling_targets(
            observation,
            world,
            weapon,
            living,
            deadline_check=deadline_check,
        )
    if weapon.role_type == 'rocket':
        return _rocket_targets(
            observation,
            world,
            weapon,
            living,
            deadline_check=deadline_check,
        )
    return ()


def _assignment_weapon_value(weapon: UnitState, mode_key: str) -> int:
    base = {
        'gatling': 300,
        'railgun': 280,
        'rocket': 260,
    }.get(weapon.role_type, 0)
    if mode_key in {'survive', 'desperation'}:
        base += {'rocket': 90, 'railgun': 60, 'gatling': 30}.get(
            weapon.role_type,
            0,
        )
    elif mode_key == 'score':
        base += {'gatling': 90, 'railgun': 50, 'rocket': 30}.get(
            weapon.role_type,
            0,
        )
    ready_bonus = 100 if weapon.cooldown == 0 and weapon.attack_range > 0 else 0
    level_bonus = max(0, weapon.level or 0) * 10
    return base + ready_bonus + level_bonus


def _operator_risk(observation: Observation, stand: Position) -> int:
    targeted = tuple(
        robot
        for robot in observation.robots
        if robot.health > 0 and robot.target_team == observation.our.team_type
    )
    if not targeted:
        return 0
    nearest = min(
        stand.chebyshev_distance(robot.position) for robot in targeted
    )
    return max(0, 6 - nearest)


def _distance_map(
    world: WorldGrid,
    start: Position,
) -> Mapping[Position, int]:
    distances = {start: 0}
    frontier = deque((start,))
    while frontier:
        current = frontier.popleft()
        next_distance = distances[current] + 1
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                if delta_x == 0 and delta_y == 0:
                    continue
                neighbor = Position(
                    current.x + delta_x,
                    current.y + delta_y,
                )
                if neighbor in distances or (
                    neighbor != start and not world.can_traverse(neighbor)
                ):
                    continue
                distances[neighbor] = next_distance
                frontier.append(neighbor)
    return MappingProxyType(distances)


def _gatling_targets(
    observation: Observation,
    world: WorldGrid,
    weapon: UnitState,
    robots: tuple[RobotState, ...],
    *,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[Position, ...]:
    robots_by_cell: dict[Position, list[RobotState]] = {}
    for robot in robots:
        if deadline_check is not None:
            deadline_check()
        if weapon.position.chebyshev_distance(robot.position) <= weapon.attack_range:
            robots_by_cell.setdefault(robot.position, []).append(robot)
    ranked_cells = tuple(
        sorted(
            robots_by_cell,
            key=lambda position: min(
                _target_key(observation, world, weapon, robot)
                for robot in robots_by_cell[position]
            ),
        )[:16]
    )
    if len(ranked_cells) < weapon.level:
        return ()
    for group in combinations(ranked_cells, weapon.level):
        if deadline_check is not None:
            deadline_check()
        if is_legal_cone(weapon.position, group):
            return group
    return ()


def _rocket_targets(
    observation: Observation,
    world: WorldGrid,
    weapon: UnitState,
    robots: tuple[RobotState, ...],
    *,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[Position, ...]:
    candidate_cells: set[Position] = set()
    for robot in robots:
        if deadline_check is not None:
            deadline_check()
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                candidate_cells.add(
                    Position(
                        robot.position.x + delta_x,
                        robot.position.y + delta_y,
                    )
                )
    ranked_cells = tuple(
        sorted(
            (
                cell
                for cell in candidate_cells
                if world.in_bounds(cell)
                and weapon.position.chebyshev_distance(cell)
                <= weapon.attack_range
            ),
            key=lambda cell: _rocket_target_key(
                observation,
                world,
                weapon,
                robots,
                cell,
                deadline_check=deadline_check,
            ),
        )
    )
    if len(ranked_cells) < weapon.level:
        return ()
    return ranked_cells[: weapon.level]


def _rocket_target_key(
    observation: Observation,
    world: WorldGrid,
    weapon: UnitState,
    robots: tuple[RobotState, ...],
    cell: Position,
    *,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[object, ...]:
    affected_list: list[RobotState] = []
    for robot in robots:
        if deadline_check is not None:
            deadline_check()
        if cell.chebyshev_distance(robot.position) <= 1:
            affected_list.append(robot)
    affected = tuple(affected_list)
    targeted_count = sum(
        robot.target_team == observation.our.team_type for robot in affected
    )
    best = min(
        (
            _target_key(observation, world, weapon, robot)
            for robot in affected
        ),
        default=(True, 10**9, 10**9, 10**9),
    )
    return (
        targeted_count == 0,
        best,
        -targeted_count,
        -len(affected),
        cell.x,
        cell.y,
    )


def _target_key(observation, world, weapon, robot) -> tuple[object, ...]:
    station = world.our_station()
    if station is None:
        defense_distance = robot.position.chebyshev_distance(weapon.position)
    else:
        station_cells = station_footprint(station.position, world.rules)
        defense_distance = min(
            robot.position.chebyshev_distance(cell) for cell in station_cells
        )
    return (
        robot.target_team != observation.our.team_type,
        defense_distance,
        robot.health,
        robot.robot_id,
    )


def _safe_candidates(
    observation: Observation,
    world: WorldGrid,
    role: UnitState,
) -> tuple[TacticalCandidate, ...]:
    station = world.our_station()
    if station is None:
        targets = (role.position,)
    else:
        footprint = station_footprint(station.position, world.rules)
        targets = tuple(
            sorted(
                (
                    Position(x, y)
                    for x in range(observation.width)
                    for y in range(observation.height)
                    if world.can_traverse(Position(x, y))
                    and min(
                        Position(x, y).chebyshev_distance(cell)
                        for cell in footprint
                    )
                    == 2
                ),
                key=lambda position: _safe_key(position, footprint, observation),
            )
        )

    for target in targets:
        job = Job(
            role_id=role.unit_id,
            kind=JobKind.PREPOSITION,
            target=target,
            priority=100,
            value=0,
        )
        candidates = candidates_for_jobs(observation, world, role, (job,))
        if any(item.action is not None for item in candidates) or target == role.position:
            return candidates
    return (TacticalCandidate.wait(role.unit_id, role.position),)


def _safe_key(
    position: Position,
    station_cells: frozenset[Position],
    observation: Observation,
) -> tuple[object, ...]:
    living_robots = tuple(robot for robot in observation.robots if robot.health > 0)
    robot_distance = (
        min(position.chebyshev_distance(robot.position) for robot in living_robots)
        if living_robots
        else 0
    )
    station_distance = min(
        position.chebyshev_distance(cell) for cell in station_cells
    )
    return (-robot_distance, station_distance, position.x, position.y)
