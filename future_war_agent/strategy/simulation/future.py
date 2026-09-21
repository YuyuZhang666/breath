from collections import deque

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.joint_fire import choose_joint_fire_attacks
from future_war_agent.strategy.policy import StrategyProfile

from .candidates import RoleMove, SimJointAction
from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .state import SimRole, SimState
from .weapons import WeaponAttack


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


def choose_future_action(
    state: SimState,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    *,
    profile: StrategyProfile = StrategyProfile.SURVIVE,
) -> SimJointAction:
    weapons = {
        weapon.unit_id: weapon for weapon in state.weapons if weapon.health > 0
    }
    role_moves: list[RoleMove] = []
    weapon_attacks: list[WeaponAttack] = list(
        choose_joint_fire_attacks(
            state,
            profile,
            simulation_config=config,
        ).attacks
    )
    reserved_targets: set[Position] = set()

    for role in sorted(state.roles, key=lambda item: item.unit_id):
        if (
            role.health <= 0
            or role.assigned_weapon_id is None
            or role.assigned_stand is None
        ):
            continue
        weapon = weapons.get(role.assigned_weapon_id)
        if weapon is None:
            continue
        if role.position != role.assigned_stand:
            step = _shortest_step(state, role, role.assigned_stand, reserved_targets)
            if step is not None:
                role_moves.append(RoleMove(role.unit_id, step))
                reserved_targets.add(step)
            continue

    return SimJointAction(
        role_moves=tuple(role_moves),
        weapon_attacks=tuple(weapon_attacks),
    )


def _shortest_step(
    state: SimState,
    role: SimRole,
    goal: Position,
    reserved_targets: set[Position],
) -> Position | None:
    blocked = set(state.static_blocked)
    blocked.update(
        other.position
        for other in state.roles
        if other.health > 0 and other.unit_id != role.unit_id
    )
    blocked.update(robot.position for robot in state.robots if robot.health > 0)
    blocked.update(reserved_targets)
    if goal in blocked or not _in_bounds(state, goal):
        return None

    queue = deque([role.position])
    parent: dict[Position, Position | None] = {role.position: None}
    found = False
    while queue and not found:
        current = queue.popleft()
        for delta_x, delta_y in _DIRECTIONS:
            neighbor = Position(current.x + delta_x, current.y + delta_y)
            if (
                neighbor in parent
                or neighbor in blocked
                or not _in_bounds(state, neighbor)
            ):
                continue
            parent[neighbor] = current
            if neighbor == goal:
                found = True
                break
            queue.append(neighbor)
    if not found:
        return None

    current = goal
    while parent[current] != role.position:
        previous = parent[current]
        if previous is None:
            return None
        current = previous
    return current


def _in_bounds(state: SimState, position: Position) -> bool:
    return 0 <= position.x < state.width and 0 <= position.y < state.height
