from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import Phase

from .defense import core_weapon_readiness
from .items import has_item
from .jobs import Job, JobKind
from .layout import DefensiveLayout
from .opponent_memory import OpponentMemory
from .pathfinding import path_to_interaction
from .policy import StrategicIntent
from .simulation.config import ROBOT_SPECS
from .world import WorldGrid


_PERSONAL_ROLES = frozenset({'worker', 'pioneer'})
_WEAPON_ROLES = frozenset({'gatling', 'railgun', 'rocket'})
_STATION_MAX_HEALTH = {1: 1500, 2: 3000, 3: 4500}
_SUMMONS = (
    ('SmallRobotSummonOrder', 'smallRobot', 20),
    ('MiddleRobotSummonOrder', 'middleRobot', 30),
    ('LargeRobotSummonOrder', 'largeRobot', 100),
    ('BossRobotSummonOrder', 'bossRobot', 200),
)
_NIGHT_HORIZON = 60
_SPAWN_TRAVEL_ROUNDS = 3
_MIN_FIRST_FALL_ADVANCE = 3
_CRITICAL_STATION_RATIO = Fraction(1, 4)
_MIN_SUSTAINED_EVENTS = 2
_MIN_SUSTAINED_DAMAGE = 100


@dataclass(frozen=True, slots=True)
class CollapseSummonPlan:
    item_name: str
    robot_type: str
    price: int
    baseline_first_fall: int
    summoned_first_fall: int
    advance_rounds: int
    added_damage: int
    evidence: tuple[str, ...]


def generate_collapse_summon_jobs(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
    opponent_memory: OpponentMemory | None,
) -> Mapping[int, tuple[Job, ...]]:
    roles = tuple(world.friendly_roles)
    empty = {role.unit_id: () for role in roles}
    if (
        observation.time.phase is not Phase.DAY
        or not intent.feature_flags.enable_collapse_summons
        or opponent_memory is None
        or not roles
        or not _core_survival_complete(observation, world, layout, intent)
    ):
        return MappingProxyType(empty)

    available = {item.name for item in observation.weapon_shop}
    available.update(
        item
        for role in roles
        for item in role.backpack
        if item.endswith('RobotSummonOrder')
    )
    plans = evaluate_collapse_summons(
        observation, opponent_memory, available_names=frozenset(available)
    )
    if not plans:
        return MappingProxyType(empty)

    jobs: dict[int, tuple[Job, ...]] = dict(empty)
    held = tuple(
        (plan, role)
        for plan in plans
        for role in roles
        if has_item(role.backpack, plan.item_name)
    )
    if held:
        plan, holder = min(
            held,
            key=lambda value: (value[0].price, value[0].summoned_first_fall, value[1].unit_id),
        )
        jobs[holder.unit_id] = (
            Job(
                role_id=holder.unit_id,
                kind=JobKind.USE_ITEM,
                target=holder.position,
                priority=max(1, intent.day_priorities.purchase - 2),
                value=float(plan.advance_rounds * 100 + plan.added_damage),
                name=plan.item_name,
            ),
        )
        return MappingProxyType(jobs)

    spendable = max(0, observation.our.gold - intent.gold_reserve)
    purchasable = tuple(
        plan
        for plan in plans
        if plan.price <= spendable
        and any(
            item.name == plan.item_name and item.price == plan.price
            for item in observation.weapon_shop
        )
    )
    if not purchasable:
        return MappingProxyType(jobs)
    plan = purchasable[0]
    shops = world.positions_for_zone('weaponShop')
    choices: list[tuple[int, int, UnitState, Position]] = []
    for buyer in roles:
        if buyer.role_type != 'worker' or len(buyer.backpack) >= buyer.backpack_capacity:
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
            priority=max(1, intent.day_priorities.purchase - 2),
            value=float(plan.advance_rounds * 100 + plan.added_damage),
            name=plan.item_name,
            quantity=1,
        ),
    )
    return MappingProxyType(jobs)


def evaluate_collapse_summons(
    observation: Observation,
    opponent_memory: OpponentMemory,
    *,
    available_names: frozenset[str],
) -> tuple[CollapseSummonPlan, ...]:
    if observation.time.phase is not Phase.DAY:
        return ()
    stations = tuple(
        unit
        for unit in observation.enemy.units
        if unit.health > 0 and unit.role_type == 'station'
    )
    if len(stations) != 1:
        return ()
    station = stations[0]
    if station.level not in _STATION_MAX_HEALTH:
        return ()

    weapons = tuple(
        unit
        for unit in observation.enemy.units
        if unit.health > 0 and unit.role_type in _WEAPON_ROLES
    )
    if any(unit.level not in {1, 2, 3} for unit in weapons):
        return ()
    role_samples = tuple(
        unit for unit in observation.enemy.units if unit.role_type in _PERSONAL_ROLES
    )
    if not role_samples:
        return ()
    living_roles = sum(unit.health > 0 for unit in role_samples)
    walls = tuple(
        unit
        for unit in observation.enemy.units
        if unit.health > 0 and unit.role_type == 'wall'
    )

    max_health = _STATION_MAX_HEALTH[station.level]
    low_station = station.health * _CRITICAL_STATION_RATIO.denominator <= (
        max_health * _CRITICAL_STATION_RATIO.numerator
    )
    sustained_events = tuple(
        item
        for item in opponent_memory.station_damage_history
        if item.day_no == observation.time.day_no - 1
        and item.phase == Phase.NIGHT.value
        and item.half_index == opponent_memory.half_index
    )
    sustained_damage = sum(item.health_loss for item in sustained_events)
    sustained_elapsed = sum(item.elapsed_ticks for item in sustained_events)
    sustained = (
        len(sustained_events) >= _MIN_SUSTAINED_EVENTS
        and sustained_damage >= _MIN_SUSTAINED_DAMAGE
        and sustained_elapsed >= _MIN_SUSTAINED_EVENTS
    )
    defense_gap = (
        len(weapons) <= 1
        or living_roles < len(weapons)
        or len(walls) <= 2
    )
    evidence = tuple(
        label
        for condition, label in (
            (low_station, 'station_critical'),
            (defense_gap, 'defense_or_controller_gap'),
            (sustained, 'previous_night_sustained_damage'),
        )
        if condition
    )
    if not evidence:
        return ()

    active_dps = _active_defense_dps(weapons, living_roles)
    baseline_rate = (
        Fraction(sustained_damage, sustained_elapsed)
        if sustained
        else Fraction(0, 1)
    )
    effective_health = station.health + sum(wall.health for wall in walls)
    baseline_fall = _first_fall_round(effective_health, baseline_rate, 0, 0)

    candidates: list[CollapseSummonPlan] = []
    prices = {item.name: item.price for item in observation.weapon_shop}
    for item_name, robot_type, fallback_price in _SUMMONS:
        if item_name not in available_names:
            continue
        price = prices.get(item_name, fallback_price)
        if price <= 0:
            continue
        spec = ROBOT_SPECS[robot_type]
        attack_rounds = _surviving_attack_rounds(spec.max_health, active_dps)
        added_damage = attack_rounds * spec.attack_power
        summoned_fall = _first_fall_round(
            effective_health,
            baseline_rate,
            spec.attack_power,
            attack_rounds,
        )
        advance = baseline_fall - summoned_fall
        if (
            summoned_fall > _NIGHT_HORIZON
            or advance < _MIN_FIRST_FALL_ADVANCE
            or added_damage <= 0
        ):
            continue
        candidates.append(
            CollapseSummonPlan(
                item_name=item_name,
                robot_type=robot_type,
                price=price,
                baseline_first_fall=baseline_fall,
                summoned_first_fall=summoned_fall,
                advance_rounds=advance,
                added_damage=added_damage,
                evidence=evidence,
            )
        )
    return tuple(sorted(
        candidates,
        key=lambda plan: (
            plan.price,
            plan.summoned_first_fall,
            -plan.advance_rounds,
            plan.item_name,
        ),
    ))


def _core_survival_complete(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
) -> bool:
    readiness = core_weapon_readiness(
        observation.our.units, intent.build_plan.weapon_loadout
    )
    wall_positions = {wall.position for wall in world.walls}
    return (
        readiness.ready_count >= readiness.required_count
        and set(layout.critical_wall_sites).issubset(wall_positions)
    )


def _active_defense_dps(
    weapons: tuple[UnitState, ...],
    living_roles: int,
) -> int:
    values = sorted(
        (_weapon_dps(weapon) for weapon in weapons), reverse=True
    )
    return sum(values[:living_roles])


def _weapon_dps(weapon: UnitState) -> int:
    level = weapon.level or 1
    if weapon.role_type == 'rocket':
        return max(1, (20 * level) // 3)
    return 10 * level


def _surviving_attack_rounds(robot_health: int, defense_dps: int) -> int:
    available = _NIGHT_HORIZON - _SPAWN_TRAVEL_ROUNDS
    if defense_dps <= 0:
        return available
    rounds_until_destroyed = (robot_health + defense_dps - 1) // defense_dps
    return max(0, min(available, rounds_until_destroyed - 1))


def _first_fall_round(
    effective_health: int,
    baseline_rate: Fraction,
    summon_attack: int,
    summon_attack_rounds: int,
) -> int:
    for round_index in range(1, _NIGHT_HORIZON + 1):
        summon_rounds = max(
            0,
            min(summon_attack_rounds, round_index - _SPAWN_TRAVEL_ROUNDS),
        )
        damage = baseline_rate * round_index + summon_attack * summon_rounds
        if damage >= effective_health:
            return round_index
    return _NIGHT_HORIZON + 1
