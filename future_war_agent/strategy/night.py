from dataclasses import dataclass
from itertools import combinations, permutations, product
from types import MappingProxyType
from typing import Mapping

from future_war_agent.decision.actions import Action
from future_war_agent.protocol.models import Observation, Position, UnitState

from .jobs import Job, JobKind
from .items import has_item
from .joint import TacticalCandidate, candidates_for_jobs
from .pathfinding import shortest_path
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .rules import station_footprint
from .world import WorldGrid


@dataclass(frozen=True, slots=True)
class ControllerAssignment:
    role_id: int
    weapon_id: int
    stand: Position
    distance: int


def assign_controllers(
    observation: Observation,
    world: WorldGrid,
) -> tuple[ControllerAssignment, ...]:
    del observation
    roles = tuple(sorted(world.friendly_roles, key=lambda value: value.unit_id))
    weapons = tuple(sorted(world.weapons, key=lambda value: value.unit_id))
    for size in range(min(len(roles), len(weapons)), 0, -1):
        best: tuple[ControllerAssignment, ...] | None = None
        best_score: tuple[object, ...] | None = None
        for selected_weapons in combinations(weapons, size):
            for selected_roles in permutations(roles, size):
                pair_options: list[tuple[ControllerAssignment, ...]] = []
                for role, weapon in zip(selected_roles, selected_weapons):
                    options = _assignment_options(world, role, weapon)
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
                        score: tuple[object, ...] = (
                            sum(item.distance for item in ordered),
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
) -> Mapping[int, tuple[TacticalCandidate, ...]]:
    assignments = {
        item.role_id: item for item in assign_controllers(observation, world)
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
) -> tuple[ControllerAssignment, ...]:
    options: list[ControllerAssignment] = []
    for stand in world.interaction_cells(weapon.position):
        path = shortest_path(world, role.position, (stand,))
        if path is not None:
            options.append(
                ControllerAssignment(
                    role_id=role.unit_id,
                    weapon_id=weapon.unit_id,
                    stand=stand,
                    distance=path.cost,
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
    if (
        weapon is not None
        and weapon.level == 1
        and weapon.cooldown == 0
        and weapon.attack_range > 0
    ):
        targets = tuple(
            robot
            for robot in observation.robots
            if robot.health > 0
            and weapon.position.chebyshev_distance(robot.position)
            <= weapon.attack_range
        )
        if targets:
            target = min(
                targets,
                key=lambda robot: _target_key(observation, world, weapon, robot),
            )
            candidates.append(
                TacticalCandidate.attack(
                    role_id=role.unit_id,
                    start=role.position,
                    weapon_id=weapon.unit_id,
                    targets=(target.position,),
                    priority=500,
                )
            )
    candidates.append(TacticalCandidate.wait(role.unit_id, role.position))
    return tuple(candidates)


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
