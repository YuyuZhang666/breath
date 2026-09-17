from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Position

from .state import SimState


ActorKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class MoveIntent:
    actor_kind: str
    actor_id: int
    start: Position
    target: Position


def resolve_simultaneous_moves(
    state: SimState,
    intents: tuple[MoveIntent, ...],
) -> Mapping[ActorKey, Position]:
    actor_positions: dict[ActorKey, Position] = {
        ("role", role.unit_id): role.position
        for role in state.roles
        if role.health > 0
    }
    actor_positions.update(
        {
            ("robot", robot.robot_id): robot.position
            for robot in state.robots
            if robot.health > 0
        }
    )
    position_occupants: dict[Position, set[ActorKey]] = {}
    for actor, position in actor_positions.items():
        position_occupants.setdefault(position, set()).add(actor)

    actor_counts = Counter(
        (intent.actor_kind, intent.actor_id) for intent in intents
    )
    intent_by_actor = {
        (intent.actor_kind, intent.actor_id): intent for intent in intents
    }
    blocked: set[ActorKey] = {
        actor for actor, count in actor_counts.items() if count != 1
    }

    target_counts = Counter(intent.target for intent in intents)
    for actor, intent in intent_by_actor.items():
        if (
            actor not in actor_positions
            or actor_positions[actor] != intent.start
            or intent.start.chebyshev_distance(intent.target) != 1
            or not _in_bounds(state, intent.target)
            or intent.target in state.static_blocked
            or target_counts[intent.target] != 1
        ):
            blocked.add(actor)

    actors = tuple(sorted(intent_by_actor))
    for left_index, left_actor in enumerate(actors):
        left = intent_by_actor[left_actor]
        for right_actor in actors[left_index + 1 :]:
            right = intent_by_actor[right_actor]
            if left.target == right.start and right.target == left.start:
                blocked.update((left_actor, right_actor))

    changed = True
    while changed:
        changed = False
        for actor, intent in intent_by_actor.items():
            if actor in blocked:
                continue
            occupants = position_occupants.get(intent.target, set())
            if any(
                occupant not in intent_by_actor or occupant in blocked
                for occupant in occupants
                if occupant != actor
            ):
                blocked.add(actor)
                changed = True

    resolved = {
        actor: intent.start if actor in blocked else intent.target
        for actor, intent in sorted(intent_by_actor.items())
    }
    return MappingProxyType(resolved)


def _in_bounds(state: SimState, position: Position) -> bool:
    return 0 <= position.x < state.width and 0 <= position.y < state.height
