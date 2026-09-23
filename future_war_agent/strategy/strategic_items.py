from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, RobotState, UnitState
from future_war_agent.protocol.time import Phase

from .defense import core_weapon_readiness
from .items import has_item
from .jobs import Job, JobKind
from .layout import DefensiveLayout, build_defensive_layout
from .pathfinding import path_to_interaction
from .policy import StrategicIntent, StrategyProfile
from .rules import station_footprint
from .simulation.config import ROBOT_SPECS
from .world import WorldGrid


STATION_UPGRADES = {1: 'StationUpgradeVoucher1', 2: 'StationUpgradeVoucher2'}
WEAPON_UPGRADES = {1: 'WeaponUpgradeVoucher1', 2: 'WeaponUpgradeVoucher2'}
WALL_FIXER = 'WallFixer'
DIZZY_WEAPON = 'DizzyWeapon'
BOMB = 'Bomb'

_MAX_HEALTH = {
    'station': {1: 1500, 2: 3000, 3: 4500},
    'wall': {1: 1000, 2: 1500, 3: 2000},
    'gatling': {1: 1000, 2: 1500, 3: 2000},
    'railgun': {1: 1000, 2: 1500, 3: 2000},
    'rocket': {1: 1000, 2: 1500, 3: 2000},
}
@dataclass(frozen=True, slots=True)
class _ItemGoal:
    item_name: str
    target: Position
    priority: int
    value: float
    purchase_only: bool = False


def generate_strategic_item_jobs(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
    *,
    expected_wall_losses: int = 0,
) -> Mapping[int, tuple[Job, ...]]:
    """Return bounded day jobs without competing with unfinished core defense.

    Item jobs are deliberately unavailable until all configured core weapons and
    critical walls exist. Their priorities remain below wall construction and
    twilight recall, so later fill/rebuild work also wins naturally.
    """
    roles = tuple(world.friendly_roles)
    empty = {role.unit_id: () for role in roles}
    if observation.time.phase is not Phase.DAY or not roles:
        return MappingProxyType(empty)
    if not _core_defense_complete(observation, world, layout, intent):
        return MappingProxyType(empty)

    high_risk = _high_risk_window(intent, expected_wall_losses)
    goal = _select_day_goal(world, layout, intent, high_risk)
    if goal is None:
        return MappingProxyType(empty)

    holders = tuple(
        role for role in roles if has_item(role.backpack, goal.item_name)
    )
    jobs: dict[int, tuple[Job, ...]] = dict(empty)
    if holders:
        if goal.purchase_only:
            return MappingProxyType(jobs)
        holder = min(
            holders,
            key=lambda role: (
                _interaction_cost(world, role, goal.target),
                role.unit_id,
            ),
        )
        if _interaction_cost(world, holder, goal.target) < 10**9:
            jobs[holder.unit_id] = (
                Job(
                    role_id=holder.unit_id,
                    kind=JobKind.USE_ITEM,
                    target=goal.target,
                    priority=goal.priority,
                    value=goal.value,
                    name=goal.item_name,
                    targeted=True,
                ),
            )
        return MappingProxyType(jobs)

    price = _shop_price(observation, goal.item_name)
    spendable = max(0, observation.our.gold - intent.gold_reserve)
    if price is None or price > spendable:
        return MappingProxyType(jobs)

    shops = world.positions_for_zone('weaponShop')
    buyers = tuple(role for role in roles if role.role_type == 'worker')
    choices: list[tuple[int, int, UnitState, Position]] = []
    for buyer in buyers:
        if len(buyer.backpack) >= buyer.backpack_capacity:
            continue
        for shop in shops:
            path = path_to_interaction(world, buyer.position, shop)
            if path is not None:
                choices.append((path.cost, buyer.unit_id, buyer, shop))
    if not choices:
        return MappingProxyType(jobs)
    _, _, buyer, shop = min(choices)
    jobs[buyer.unit_id] = (
        Job(
            role_id=buyer.unit_id,
            kind=JobKind.BUY,
            target=shop,
            priority=max(1, intent.day_priorities.purchase - 1),
            value=goal.value,
            name=goal.item_name,
            quantity=1,
        ),
    )
    return MappingProxyType(jobs)


def merge_strategic_item_jobs(
    base: Mapping[int, tuple[Job, ...]],
    item_jobs: Mapping[int, tuple[Job, ...]],
) -> Mapping[int, tuple[Job, ...]]:
    merged = {
        role_id: tuple(
            sorted(
                (*base.get(role_id, ()), *item_jobs.get(role_id, ())),
                key=lambda job: job.sort_key,
            )
        )
        for role_id in set(base) | set(item_jobs)
    }
    return MappingProxyType(merged)


def apply_emergency_combat_items(
    observation: Observation,
    baseline: Decision,
    intent: StrategicIntent,
) -> Decision:
    """Use a held emergency item only through a currently idle role.

    This is intentionally an overlay rather than a fire-plan competitor: a role
    controlling any baseline weapon attack is never selected, so no attack is
    removed or weakened.
    """
    if observation.time.phase is not Phase.NIGHT:
        return baseline
    flags = intent.feature_flags
    if not (
        flags.enable_stun or flags.enable_bomb or flags.enable_repairs
    ):
        return baseline

    station = next(
        (
            unit
            for unit in observation.our.units
            if unit.health > 0 and unit.role_type == 'station'
        ),
        None,
    )
    if station is None:
        return baseline
    threats = tuple(
        robot
        for robot in observation.robots
        if robot.health > 0
        and _targets_us(robot, observation.our.team_type)
        and _can_hit_station(robot, station)
    )
    used_roles = {
        actor_id
        for actor_id, action in baseline.commands.items()
        if action.kind is not ActionKind.ATTACK
    }
    used_roles.update(
        action.controller_id
        for action in baseline.commands.values()
        if action.kind is ActionKind.ATTACK and action.controller_id is not None
    )
    roles = tuple(
        unit
        for unit in observation.our.units
        if unit.health > 0
        and unit.role_type in {'worker', 'pioneer'}
        and unit.unit_id not in used_roles
    )
    if not roles:
        return baseline

    candidates: list[tuple[int, int, int, str, Position, UnitState]] = []
    stun_name = intent.item_policy.stun_name or DIZZY_WEAPON
    bomb_name = intent.item_policy.bomb_name or BOMB
    repair_name = intent.item_policy.repair_name or WALL_FIXER
    repair_target, repair_missing = (
        _best_night_repair_target(observation, intent)
        if flags.enable_repairs
        and any(has_item(role.backpack, repair_name) for role in roles)
        else (None, 0)
    )
    for role in roles:
        if threats:
            if flags.enable_stun and has_item(role.backpack, stun_name):
                target, prevented = _best_stun_target(observation, threats)
                if target is not None and prevented > 0:
                    candidates.append(
                        (prevented * 5, prevented, -role.unit_id, stun_name, target, role)
                    )
            if flags.enable_bomb and has_item(role.backpack, bomb_name):
                target, prevented, kills = _best_bomb_target(observation, threats)
                if target is not None and prevented > 0 and kills > 0:
                    candidates.append(
                        (prevented, kills, -role.unit_id, bomb_name, target, role)
                    )
        if (
            repair_target is not None
            and repair_missing > 0
            and has_item(role.backpack, repair_name)
            and role.position.chebyshev_distance(repair_target) <= 1
        ):
            candidates.append(
                (
                    repair_missing,
                    repair_missing,
                    -role.unit_id,
                    repair_name,
                    repair_target,
                    role,
                )
            )
    if not candidates:
        return baseline

    _, _, _, name, target, role = max(candidates)
    commands = dict(baseline.commands)
    commands[role.unit_id] = Action.use(name, target)
    return Decision(
        commands=commands,
        prompt=baseline.prompt,
        execute_command=baseline.execute_command,
    )


def _core_defense_complete(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
) -> bool:
    readiness = core_weapon_readiness(
        observation.our.units,
        intent.build_plan.weapon_loadout,
    )
    walls = {wall.position for wall in world.walls}
    return (
        readiness.ready_count >= readiness.required_count
        and set(layout.critical_wall_sites).issubset(walls)
    )


def _high_risk_window(
    intent: StrategicIntent,
    expected_wall_losses: int,
) -> bool:
    return (
        expected_wall_losses > 0
        or intent.profile in {StrategyProfile.SURVIVE, StrategyProfile.DESPERATION}
    )


def _select_day_goal(
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
    high_risk: bool,
) -> _ItemGoal | None:
    flags = intent.feature_flags
    station = world.our_station()
    if flags.enable_upgrades and station is not None:
        goal = _upgrade_goal(station, STATION_UPGRADES, 0.75, 344, 10_000.0)
        if goal is not None:
            return goal
    if not high_risk:
        return None
    if flags.enable_upgrades:
        damaged_weapons = tuple(
            goal
            for weapon in world.weapons
            if (goal := _upgrade_goal(
                weapon, WEAPON_UPGRADES, 0.60, 342, 5_000.0
            )) is not None
        )
        if damaged_weapons:
            return max(
                damaged_weapons,
                key=lambda goal: (goal.value, -goal.target.x, -goal.target.y),
            )
    if flags.enable_repairs:
        critical = set(layout.critical_wall_sites)
        damaged_walls = tuple(
            wall
            for wall in world.walls
            if wall.position in critical
            and wall.health < _max_health(wall) * 0.50
        )
        if damaged_walls:
            wall = min(damaged_walls, key=lambda unit: (unit.health, unit.unit_id))
            return _ItemGoal(
                intent.item_policy.repair_name or WALL_FIXER,
                wall.position,
                340,
                float(_max_health(wall) - wall.health),
            )
    station_target = station.position if station is not None else Position(0, 0)
    if (
        flags.enable_stun
        and not any(
            has_item(role.backpack, intent.item_policy.stun_name or DIZZY_WEAPON)
            for role in world.friendly_roles
        )
    ):
        return _ItemGoal(
            intent.item_policy.stun_name or DIZZY_WEAPON,
            station_target,
            320,
            2_500.0,
            purchase_only=True,
        )
    if (
        flags.enable_bomb
        and not any(
            has_item(role.backpack, intent.item_policy.bomb_name or BOMB)
            for role in world.friendly_roles
        )
    ):
        return _ItemGoal(
            intent.item_policy.bomb_name or BOMB,
            station_target,
            320,
            2_000.0,
            purchase_only=True,
        )
    return None


def _upgrade_goal(
    unit: UnitState,
    names: Mapping[int, str],
    health_ratio: float,
    priority: int,
    base_value: float,
) -> _ItemGoal | None:
    level = unit.level or 1
    maximum = _max_health(unit)
    name = names.get(level)
    if name is None or unit.health >= maximum * health_ratio:
        return None
    return _ItemGoal(
        name,
        unit.position,
        priority,
        base_value + maximum - unit.health,
    )


def _max_health(unit: UnitState) -> int:
    levels = _MAX_HEALTH.get(unit.role_type, {})
    return levels.get(unit.level or 1, max(unit.health, 1))


def _shop_price(observation: Observation, name: str) -> int | None:
    return min(
        (item.price for item in observation.weapon_shop if item.name == name),
        default=None,
    )


def _interaction_cost(world: WorldGrid, role: UnitState, target: Position) -> int:
    if role.position.chebyshev_distance(target) <= 1:
        return 0
    path = path_to_interaction(world, role.position, target)
    return path.cost if path is not None else 10**9


def _targets_us(robot: RobotState, team_type: str) -> bool:
    return robot.target_team in {None, '', team_type}


def _robot_spec(robot: RobotState):
    return ROBOT_SPECS.get(robot.role_type, max(
        ROBOT_SPECS.values(), key=lambda spec: spec.attack_power
    ))


def _can_hit_station(robot: RobotState, station: UnitState) -> bool:
    footprint = station_footprint(station.position)
    return min(robot.position.chebyshev_distance(cell) for cell in footprint) <= (
        _robot_spec(robot).attack_range
    )


def _covered(center: Position, robot: RobotState) -> bool:
    return center.chebyshev_distance(robot.position) <= 1


def _best_night_repair_target(
    observation: Observation,
    intent: StrategicIntent,
) -> tuple[Position | None, int]:
    world = WorldGrid.from_observation(observation)
    layout = build_defensive_layout(world, intent.build_plan)
    critical = set(layout.critical_wall_sites)
    damaged = tuple(
        wall
        for wall in world.walls
        if wall.position in critical
        and wall.health < _max_health(wall) * 0.50
    )
    if not damaged:
        return None, 0
    wall = min(damaged, key=lambda unit: (unit.health, unit.unit_id))
    return wall.position, _max_health(wall) - wall.health


def _best_stun_target(
    observation: Observation,
    threats: tuple[RobotState, ...],
) -> tuple[Position | None, int]:
    active = tuple(robot for robot in threats if robot.abnormal_state != 'dizzy')
    if not active:
        return None, 0
    centers = tuple(robot.position for robot in observation.robots if robot.health > 0)
    scored = (
        (
            sum(
                _robot_spec(robot).attack_power
                for robot in active
                if _covered(center, robot)
            ),
            -center.x,
            -center.y,
            center,
        )
        for center in centers
    )
    prevented, _, _, center = max(scored)
    return center, prevented


def _best_bomb_target(
    observation: Observation,
    threats: tuple[RobotState, ...],
) -> tuple[Position | None, int, int]:
    centers = tuple(robot.position for robot in observation.robots if robot.health > 0)
    scored = []
    for center in centers:
        killed = tuple(
            robot
            for robot in threats
            if _covered(center, robot) and robot.health <= 100
        )
        prevented = sum(_robot_spec(robot).attack_power for robot in killed)
        scored.append((prevented, len(killed), -center.x, -center.y, center))
    if not scored:
        return None, 0, 0
    prevented, kills, _, _, center = max(scored)
    return center, prevented, kills
