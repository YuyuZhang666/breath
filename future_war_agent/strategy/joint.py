from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Mapping

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import Phase

from .jobs import Job, JobKind
from .pathfinding import first_step_options
from .world import WorldGrid


@dataclass(frozen=True, slots=True)
class TacticalCandidate:
    role_id: int
    command_actor_id: int
    action: Action | None
    job_kind: JobKind
    start: Position
    move_target: Position | None = None
    build_target: Position | None = None
    priority: int = 0
    completes_job: bool = False
    progress: int = 0
    gold_cost: int = 0
    stone_cost: int = 0

    @classmethod
    def wait(cls, role_id: int, start: Position) -> "TacticalCandidate":
        return cls(
            role_id=role_id,
            command_actor_id=role_id,
            action=None,
            job_kind=JobKind.PREPOSITION,
            start=start,
        )

    @classmethod
    def build(
        cls,
        role_id: int,
        start: Position,
        target: Position,
        name: str,
        priority: int,
        gold_cost: int = 0,
        stone_cost: int = 0,
    ) -> "TacticalCandidate":
        kind = JobKind.BUILD_WALL if name == "wall" else JobKind.BUILD_WEAPON
        return cls(
            role_id=role_id,
            command_actor_id=role_id,
            action=Action.build(name, target),
            job_kind=kind,
            start=start,
            build_target=target,
            priority=priority,
            completes_job=True,
            progress=1,
            gold_cost=gold_cost,
            stone_cost=stone_cost,
        )

    @classmethod
    def attack(
        cls,
        role_id: int,
        start: Position,
        weapon_id: int,
        target: Position,
        priority: int,
    ) -> "TacticalCandidate":
        return cls(
            role_id=role_id,
            command_actor_id=weapon_id,
            action=Action.attack(role_id, (target,)),
            job_kind=JobKind.ATTACK,
            start=start,
            priority=priority,
            completes_job=True,
            progress=1,
        )


def candidates_for_jobs(
    observation: Observation,
    world: WorldGrid,
    role: UnitState,
    jobs: tuple[Job, ...],
    *,
    limit: int = 4,
) -> tuple[TacticalCandidate, ...]:
    generated: list[TacticalCandidate] = []
    for job in jobs:
        if job.kind is JobKind.COLLECT:
            if role.position.chebyshev_distance(job.target) == 1:
                generated.append(
                    _direct(role, job, Action.collect(job.target))
                )
            else:
                generated.extend(_move_candidates(world, role, job, interaction=True))
        elif job.kind is JobKind.SELL:
            if role.position.chebyshev_distance(job.target) == 1:
                if job.name is not None and job.quantity is not None:
                    generated.append(
                        _direct(role, job, Action.sell(job.name, job.quantity))
                    )
            else:
                generated.extend(_move_candidates(world, role, job, interaction=True))
        elif job.kind in {JobKind.BUILD_WEAPON, JobKind.BUILD_WALL}:
            if role.position.chebyshev_distance(job.target) == 1:
                if job.name is not None:
                    generated.append(
                        TacticalCandidate.build(
                            role.unit_id,
                            role.position,
                            job.target,
                            job.name,
                            job.priority,
                            gold_cost=(
                                world.rules.weapon_build_cost
                                if job.kind is JobKind.BUILD_WEAPON
                                else 0
                            ),
                            stone_cost=(
                                world.rules.wall_material_cost
                                if job.kind is JobKind.BUILD_WALL
                                else 0
                            ),
                        )
                    )
            else:
                generated.extend(_move_candidates(world, role, job, interaction=True))
        elif job.kind in {JobKind.RECALL, JobKind.PREPOSITION}:
            interaction = job.target in world.hard_blocked
            generated.extend(
                _move_candidates(world, role, job, interaction=interaction)
            )

    unique: dict[tuple[int, Action | None], TacticalCandidate] = {}
    for item in generated:
        unique.setdefault((item.command_actor_id, item.action), item)
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -item.priority,
            -item.completes_job,
            -item.progress,
            item.command_actor_id,
            repr(item.action),
        ),
    )
    wait = TacticalCandidate.wait(role.unit_id, role.position)
    if limit <= 0:
        return ()
    if len(ranked) >= limit:
        return tuple(ranked[: limit - 1] + [wait])
    return tuple(ranked + [wait])


def is_valid_joint(
    observation: Observation,
    world: WorldGrid,
    joint: tuple[TacticalCandidate, ...],
) -> bool:
    role_ids = [item.role_id for item in joint]
    actor_ids = [item.command_actor_id for item in joint]
    if len(set(role_ids)) != len(role_ids) or len(set(actor_ids)) != len(actor_ids):
        return False

    move_by_role = {
        item.role_id: item.move_target
        for item in joint
        if item.move_target is not None
    }
    move_targets = tuple(move_by_role.values())
    if len(set(move_targets)) != len(move_targets):
        return False
    for item in joint:
        target = item.move_target
        if target is None:
            continue
        if (
            not world.in_bounds(target)
            or target in world.hard_blocked
            or item.start.chebyshev_distance(target) != 1
        ):
            return False
        for other in joint:
            if (
                other.role_id != item.role_id
                and other.move_target == item.start
                and target == other.start
            ):
                return False

    occupant_by_position = {
        role.position: role.unit_id for role in world.friendly_roles
    }
    for role_id, target in move_by_role.items():
        occupant = occupant_by_position.get(target)
        if occupant is None or occupant == role_id:
            continue
        occupant_target = move_by_role.get(occupant)
        if occupant_target is None or occupant_target == target:
            return False

    build_targets = tuple(
        item.build_target for item in joint if item.build_target is not None
    )
    if len(set(build_targets)) != len(build_targets):
        return False
    if set(build_targets).intersection(move_targets):
        return False
    if sum(item.gold_cost for item in joint) > observation.our.gold:
        return False
    for item in joint:
        if item.stone_cost:
            role = world.unit_by_id(item.role_id)
            if role is None:
                return False
            if Counter(role.backpack)[world.rules.wall_material] < item.stone_cost:
                return False
        if not _direct_action_is_legal(observation, world, item):
            return False
    return True


def solve_joint(
    observation: Observation,
    world: WorldGrid,
    choices: Mapping[int, tuple[TacticalCandidate, ...]],
) -> Decision:
    role_ids = tuple(sorted(choices))
    if not role_ids or any(not choices[role_id] for role_id in role_ids):
        return Decision()

    winner: tuple[TacticalCandidate, ...] | None = None
    winner_score: tuple[object, ...] | None = None
    for combination in product(*(choices[role_id] for role_id in role_ids)):
        joint = tuple(combination)
        if not is_valid_joint(observation, world, joint):
            continue
        score: tuple[object, ...] = (
            sum(item.priority for item in joint),
            sum(item.completes_job for item in joint),
            sum(item.progress for item in joint),
            sum(item.action is not None for item in joint),
            tuple(
                (item.command_actor_id, repr(item.action))
                for item in sorted(
                    joint,
                    key=lambda value: value.command_actor_id,
                )
            ),
        )
        if winner_score is None or score > winner_score:
            winner = joint
            winner_score = score

    if winner is None:
        return Decision()
    return Decision(
        commands={
            item.command_actor_id: item.action
            for item in winner
            if item.action is not None
        }
    )


def _move_candidates(
    world: WorldGrid,
    role: UnitState,
    job: Job,
    *,
    interaction: bool,
) -> list[TacticalCandidate]:
    goals = world.interaction_cells(job.target) if interaction else (job.target,)
    if role.position in goals:
        return []
    return [
        TacticalCandidate(
            role_id=role.unit_id,
            command_actor_id=role.unit_id,
            action=Action.move(target),
            job_kind=job.kind,
            start=role.position,
            move_target=target,
            priority=job.priority,
            progress=1,
        )
        for target in first_step_options(world, role.position, goals, limit=2)
    ]


def _direct(role: UnitState, job: Job, action: Action) -> TacticalCandidate:
    return TacticalCandidate(
        role_id=role.unit_id,
        command_actor_id=role.unit_id,
        action=action,
        job_kind=job.kind,
        start=role.position,
        priority=job.priority,
        completes_job=True,
        progress=1,
    )


def _direct_action_is_legal(
    observation: Observation,
    world: WorldGrid,
    item: TacticalCandidate,
) -> bool:
    action = item.action
    if action is None or action.kind is ActionKind.MOVE:
        return True
    role = world.unit_by_id(item.role_id)
    if role is None or role.health <= 0:
        return False
    if action.kind is ActionKind.BUILD:
        return (
            observation.time.phase is Phase.DAY
            and role.role_type == "worker"
            and item.build_target is not None
            and role.position.chebyshev_distance(item.build_target) == 1
            and world.in_bounds(item.build_target)
            and item.build_target not in world.hard_blocked
        )
    if action.kind is ActionKind.COLLECT:
        return (
            role.role_type == "worker"
            and len(action.target_positions) == 1
            and role.position.chebyshev_distance(action.target_positions[0]) == 1
            and action.target_positions[0] in world.neutral_cells
        )
    if action.kind is ActionKind.SELL:
        return (
            len(action.target_positions) == 0
            and any(
                role.position.chebyshev_distance(vendor) == 1
                for vendor in world.positions_for_zone("vendor")
            )
            and action.name is not None
            and action.quantity is not None
            and action.quantity > 0
            and Counter(role.backpack)[action.name] >= action.quantity
        )
    if action.kind is ActionKind.ATTACK:
        weapon = world.unit_by_id(item.command_actor_id)
        return (
            observation.time.phase is Phase.NIGHT
            and weapon is not None
            and role.position.chebyshev_distance(weapon.position) == 1
            and action.controller_id == role.unit_id
            and len(action.target_positions) == 1
            and weapon.position.chebyshev_distance(action.target_positions[0])
            <= weapon.attack_range
        )
    return False
