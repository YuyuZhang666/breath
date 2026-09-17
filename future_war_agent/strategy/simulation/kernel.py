from dataclasses import dataclass, replace

from future_war_agent.protocol.models import Position

from .candidates import SimJointAction
from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .geometry import is_legal_cone
from .movement import MoveIntent, resolve_simultaneous_moves
from .robots import RobotPolicy, choose_robot_intents
from .state import SimRobot, SimRole, SimState, SimStructure, SimWeapon
from .weapons import WeaponAttack, weapon_damage


@dataclass(slots=True)
class DamageLedger:
    role_damage: dict[int, int]
    structure_damage: dict[int, int]
    weapon_damage: dict[int, int]
    robot_damage_by_us: dict[int, int]

    @classmethod
    def empty(cls) -> "DamageLedger":
        return cls({}, {}, {}, {})


def step_simulation(
    state: SimState,
    action: SimJointAction,
    robot_policy: RobotPolicy,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> SimState:
    ledger = DamageLedger.empty()
    living_roles = {role.unit_id: role for role in state.roles if role.health > 0}
    living_weapons = {
        weapon.unit_id: weapon for weapon in state.weapons if weapon.health > 0
    }
    moving_role_ids = {move.role_id for move in action.role_moves}
    fired_rockets: set[int] = set()
    used_weapons: set[int] = set()

    for attack in sorted(
        action.weapon_attacks,
        key=lambda item: item.stable_key,
    ):
        weapon = living_weapons.get(attack.weapon_id)
        controller = living_roles.get(attack.controller_id)
        if (
            weapon is None
            or controller is None
            or attack.weapon_id in used_weapons
            or attack.controller_id in moving_role_ids
            or not _can_fire(state, weapon, controller, attack)
        ):
            continue
        used_weapons.add(attack.weapon_id)
        damage = weapon_damage(state, attack, config)
        for robot_id, amount in damage.items():
            ledger.robot_damage_by_us[robot_id] = (
                ledger.robot_damage_by_us.get(robot_id, 0) + amount
            )
        if weapon.role_type == "rocket":
            fired_rockets.add(weapon.unit_id)

    robot_intents = choose_robot_intents(state, robot_policy)
    role_by_id = {role.unit_id: role for role in state.roles if role.health > 0}
    robot_by_id = {
        robot.robot_id: robot for robot in state.robots if robot.health > 0
    }
    move_intents: list[MoveIntent] = []
    for move in action.role_moves:
        role = role_by_id.get(move.role_id)
        if role is not None:
            move_intents.append(
                MoveIntent("role", role.unit_id, role.position, move.target)
            )
    for intent in robot_intents:
        robot = robot_by_id.get(intent.robot_id)
        if robot is not None and intent.move_target is not None:
            move_intents.append(
                MoveIntent(
                    "robot",
                    robot.robot_id,
                    robot.position,
                    intent.move_target,
                )
            )
    resolved_positions = resolve_simultaneous_moves(state, tuple(move_intents))

    for intent in robot_intents:
        robot = robot_by_id.get(intent.robot_id)
        if (
            robot is None
            or intent.attack_target_kind is None
            or intent.attack_target_id is None
        ):
            continue
        target_id = intent.attack_target_id
        if intent.attack_target_kind == "role" and target_id in role_by_id:
            _add_damage(ledger.role_damage, target_id, robot.attack_power)
        elif intent.attack_target_kind == "weapon" and target_id in living_weapons:
            _add_damage(ledger.weapon_damage, target_id, robot.attack_power)
        elif intent.attack_target_kind == "wall" and any(
            wall.unit_id == target_id and wall.health > 0 for wall in state.walls
        ):
            _add_damage(ledger.structure_damage, target_id, robot.attack_power)
        elif (
            intent.attack_target_kind == "station"
            and state.station.unit_id == target_id
            and state.station.health > 0
        ):
            _add_damage(ledger.structure_damage, target_id, robot.attack_power)

    roles = tuple(
        updated
        for role in state.roles
        if (
            updated := replace(
                role,
                position=resolved_positions.get(
                    ("role", role.unit_id),
                    role.position,
                ),
                health=max(0, role.health - ledger.role_damage.get(role.unit_id, 0)),
            )
        ).health
        > 0
    )
    station = replace(
        state.station,
        health=max(
            0,
            state.station.health
            - ledger.structure_damage.get(state.station.unit_id, 0),
        ),
    )
    walls = tuple(
        updated
        for wall in state.walls
        if (
            updated := replace(
                wall,
                health=max(
                    0,
                    wall.health - ledger.structure_damage.get(wall.unit_id, 0),
                ),
            )
        ).health
        > 0
    )
    weapons = tuple(
        updated
        for weapon in state.weapons
        if (
            updated := replace(
                weapon,
                health=max(
                    0,
                    weapon.health - ledger.weapon_damage.get(weapon.unit_id, 0),
                ),
                cooldown=(
                    config.rocket_cooldown_rounds
                    if weapon.unit_id in fired_rockets
                    else max(0, weapon.cooldown - 1)
                ),
            )
        ).health
        > 0
    )

    score_gain = 0
    robots_list: list[SimRobot] = []
    for robot in state.robots:
        health = max(
            0,
            robot.health - ledger.robot_damage_by_us.get(robot.robot_id, 0),
        )
        if (
            robot.health > 0
            and health == 0
            and ledger.robot_damage_by_us.get(robot.robot_id, 0) > 0
        ):
            score_gain += robot.kill_score
        if health > 0:
            robots_list.append(
                replace(
                    robot,
                    position=resolved_positions.get(
                        ("robot", robot.robot_id),
                        robot.position,
                    ),
                    health=health,
                    waits_this_turn=False,
                )
            )

    remaining_night_turns = max(0, state.remaining_night_turns - 1)
    robots = tuple(robots_list) if remaining_night_turns > 0 else ()
    static_blocked = _updated_static_blocked(
        state,
        station,
        walls,
        weapons,
    )
    return SimState(
        round_no=state.round_no + 1,
        width=state.width,
        height=state.height,
        remaining_night_turns=remaining_night_turns,
        team_type=state.team_type,
        static_blocked=static_blocked,
        roles=roles,
        station=station,
        walls=walls,
        weapons=weapons,
        robots=robots,
        owned_kill_score=state.owned_kill_score + score_gain,
    )


def _can_fire(
    state: SimState,
    weapon: SimWeapon,
    controller: SimRole,
    attack: WeaponAttack,
) -> bool:
    if (
        weapon.cooldown != 0
        or controller.assigned_weapon_id != weapon.unit_id
        or controller.position.chebyshev_distance(weapon.position) != 1
        or len(set(attack.targets)) != len(attack.targets)
    ):
        return False
    if any(
        not (0 <= target.x < state.width and 0 <= target.y < state.height)
        or weapon.position.chebyshev_distance(target) > weapon.attack_range
        for target in attack.targets
    ):
        return False
    if weapon.role_type == "railgun":
        return len(attack.targets) == 1
    if weapon.role_type == "gatling":
        return len(attack.targets) == weapon.level and is_legal_cone(
            weapon.position,
            attack.targets,
        )
    if weapon.role_type == "rocket":
        return len(attack.targets) == weapon.level
    return False


def _add_damage(ledger: dict[int, int], target_id: int, amount: int) -> None:
    ledger[target_id] = ledger.get(target_id, 0) + amount


def _updated_static_blocked(
    old_state: SimState,
    station: SimStructure,
    walls: tuple[SimStructure, ...],
    weapons: tuple[SimWeapon, ...],
) -> frozenset[Position]:
    old_building_cells = set(old_state.station.occupied_cells)
    old_building_cells.update(
        cell for wall in old_state.walls for cell in wall.occupied_cells
    )
    old_building_cells.update(weapon.position for weapon in old_state.weapons)
    blocked = set(old_state.static_blocked) - old_building_cells
    if station.health > 0:
        blocked.update(station.occupied_cells)
    blocked.update(cell for wall in walls for cell in wall.occupied_cells)
    blocked.update(weapon.position for weapon in weapons)
    return frozenset(blocked)
