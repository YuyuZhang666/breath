from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Callable, Mapping

from future_war_agent.protocol.models import Position

from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .geometry import (
    Ray,
    bresenham_cells,
    center_intersection_cells,
    is_legal_cone,
    supercover_cells,
)
from .state import SimRobot, SimState, SimWeapon


_RAY_FUNCTIONS: tuple[Callable[[Position, Position], Ray], ...] = (
    bresenham_cells,
    supercover_cells,
    center_intersection_cells,
)


@dataclass(frozen=True, slots=True)
class WeaponAttack:
    weapon_id: int
    controller_id: int
    targets: tuple[Position, ...]

    @property
    def stable_key(self) -> tuple[object, ...]:
        return (
            self.weapon_id,
            self.controller_id,
            tuple((target.x, target.y) for target in self.targets),
        )


def generate_weapon_attacks(
    state: SimState,
    weapon: SimWeapon,
    controller_id: int,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> tuple[WeaponAttack, ...]:
    if weapon.cooldown != 0 or not state.robots:
        return ()

    candidate_cells = _candidate_cells(state, weapon)
    if weapon.role_type == "railgun":
        target_groups = tuple((cell,) for cell in candidate_cells)
    elif weapon.role_type in {"gatling", "rocket"}:
        if len(candidate_cells) < weapon.level:
            return ()
        target_groups = tuple(combinations(candidate_cells, weapon.level))
    else:
        return ()

    attacks: list[WeaponAttack] = []
    for targets in target_groups:
        normalized_targets = tuple(targets)
        if (
            weapon.role_type == "gatling"
            and not is_legal_cone(weapon.position, normalized_targets)
        ):
            continue
        attacks.append(
            WeaponAttack(
                weapon_id=weapon.unit_id,
                controller_id=controller_id,
                targets=normalized_targets,
            )
        )

    ranked = sorted(
        attacks,
        key=lambda attack: _attack_rank(state, attack, config),
    )
    return tuple(ranked[: config.max_weapon_candidates])


def weapon_damage(
    state: SimState,
    attack: WeaponAttack,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> Mapping[int, int]:
    weapon = next(
        (item for item in state.weapons if item.unit_id == attack.weapon_id),
        None,
    )
    if weapon is None or not _is_legal_attack(state, weapon, attack):
        return MappingProxyType({})

    if weapon.role_type == "rocket":
        return MappingProxyType(_rocket_damage(state, attack, config))

    ledgers = tuple(
        _ray_damage(state, weapon, attack, ray_function, config)
        for ray_function in _RAY_FUNCTIONS
    )
    robot_ids = set().union(*(ledger.keys() for ledger in ledgers))
    conservative = {
        robot_id: min(ledger.get(robot_id, 0) for ledger in ledgers)
        for robot_id in robot_ids
    }
    return MappingProxyType(
        {
            robot_id: damage
            for robot_id, damage in sorted(conservative.items())
            if damage > 0
        }
    )


def _candidate_cells(
    state: SimState,
    weapon: SimWeapon,
) -> tuple[Position, ...]:
    living_robots = tuple(robot for robot in state.robots if robot.health > 0)
    if weapon.role_type == "rocket":
        raw_cells = {
            Position(robot.position.x + delta_x, robot.position.y + delta_y)
            for robot in living_robots
            for delta_x in (-1, 0, 1)
            for delta_y in (-1, 0, 1)
        }
    else:
        raw_cells = {robot.position for robot in living_robots}
    return tuple(
        sorted(
            (
                cell
                for cell in raw_cells
                if _in_bounds(state, cell)
                and weapon.position.chebyshev_distance(cell)
                <= weapon.attack_range
            ),
            key=lambda cell: (cell.x, cell.y),
        )
    )


def _is_legal_attack(
    state: SimState,
    weapon: SimWeapon,
    attack: WeaponAttack,
) -> bool:
    if weapon.cooldown != 0 or len(set(attack.targets)) != len(attack.targets):
        return False
    if any(
        not _in_bounds(state, target)
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


def _ray_damage(
    state: SimState,
    weapon: SimWeapon,
    attack: WeaponAttack,
    ray_function: Callable[[Position, Position], Ray],
    config: Phase3Config,
) -> dict[int, int]:
    robots_by_position: dict[Position, tuple[SimRobot, ...]] = {}
    for robot in sorted(state.robots, key=lambda item: item.robot_id):
        if robot.health <= 0:
            continue
        robots_by_position.setdefault(robot.position, ())
        robots_by_position[robot.position] += (robot,)

    ledger: dict[int, int] = {}
    if weapon.role_type == "gatling":
        for target in attack.targets:
            for cell in ray_function(weapon.position, target):
                robots = robots_by_position.get(cell, ())
                if robots:
                    robot = robots[0]
                    ledger[robot.robot_id] = (
                        ledger.get(robot.robot_id, 0) + config.gatling_damage
                    )
                    break
        return ledger

    if weapon.role_type == "railgun":
        remaining_energy = weapon.attack_power
        for cell in ray_function(weapon.position, attack.targets[0]):
            for robot in robots_by_position.get(cell, ()):
                damage = min(remaining_energy, robot.health)
                if damage > 0:
                    ledger[robot.robot_id] = damage
                    remaining_energy -= damage
                if remaining_energy == 0:
                    return ledger
        return ledger

    return ledger


def _rocket_damage(
    state: SimState,
    attack: WeaponAttack,
    config: Phase3Config,
) -> dict[int, int]:
    ledger: dict[int, int] = {}
    for robot in sorted(state.robots, key=lambda item: item.robot_id):
        if robot.health <= 0:
            continue
        damage = 0
        for target in attack.targets:
            distance = robot.position.chebyshev_distance(target)
            if distance == 0:
                damage += config.rocket_center_damage
            elif distance == 1:
                damage += config.rocket_splash_damage
        if damage > 0:
            ledger[robot.robot_id] = damage
    return ledger


def _attack_rank(
    state: SimState,
    attack: WeaponAttack,
    config: Phase3Config,
) -> tuple[object, ...]:
    damage = weapon_damage(state, attack, config)
    robots_by_id = {robot.robot_id: robot for robot in state.robots}
    killed = tuple(
        robots_by_id[robot_id]
        for robot_id, amount in damage.items()
        if robot_id in robots_by_id and amount >= robots_by_id[robot_id].health
    )
    killed_score = sum(robot.kill_score for robot in killed)
    threat_removed = sum(robot.attack_power for robot in killed)
    total_damage = sum(
        min(amount, robots_by_id[robot_id].health)
        for robot_id, amount in damage.items()
        if robot_id in robots_by_id
    )
    return (
        -killed_score,
        -threat_removed,
        -total_damage,
        attack.stable_key,
    )


def _in_bounds(state: SimState, position: Position) -> bool:
    return 0 <= position.x < state.width and 0 <= position.y < state.height
