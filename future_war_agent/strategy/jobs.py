from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Observation, Position, UnitState

from .items import count_item, has_item
from .layout import DefensiveLayout
from .pathfinding import path_to_interaction
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .world import WorldGrid


class JobKind(StrEnum):
    BUY = 'buy'
    USE_ITEM = 'use_item'
    RECALL = "recall"
    BUILD_WEAPON = "build_weapon"
    BUILD_WALL = "build_wall"
    SELL = "sell"
    COLLECT = "collect"
    PREPOSITION = "preposition"
    ATTACK = "attack"


@dataclass(frozen=True, slots=True)
class Job:
    role_id: int
    kind: JobKind
    target: Position
    priority: int
    value: float
    name: str | None = None
    quantity: int | None = None
    weapon_id: int | None = None

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            -self.priority,
            -self.value,
            self.kind.value,
            self.target.x,
            self.target.y,
            self.name or "",
        )


RECALL_PRIORITY = 500
BUILD_WEAPON_PRIORITY = 400
BUILD_WALL_PRIORITY = 350
SELL_PRIORITY = 300
COLLECT_PRIORITY = 200
PREPOSITION_PRIORITY = 100


def generate_day_jobs(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
) -> Mapping[int, tuple[Job, ...]]:
    priorities = intent.day_priorities
    roles = tuple(sorted(world.friendly_roles, key=lambda value: value.unit_id))
    workers = tuple(role for role in roles if role.role_type == "worker")
    pioneers = tuple(role for role in roles if role.role_type == "pioneer")
    result: dict[int, list[Job]] = {role.unit_id: [] for role in roles}
    medicine_buyer_ids = _medicine_buyer_ids(
        workers,
        observation,
        world,
        intent,
    )

    existing_weapon_sites = {
        (weapon.position, weapon.role_type) for weapon in world.weapons
    }
    missing_weapon_sites = tuple(
        site
        for site in layout.weapon_sites
        if (site.position, site.weapon_type) not in existing_weapon_sites
    ) if intent.build_plan.build_weapons else ()
    occupied_weapon_sites = not missing_weapon_sites and bool(layout.weapon_sites)
    existing_wall_positions = {wall.position for wall in world.walls}
    missing_wall_sites = tuple(
        position
        for position in layout.wall_sites
        if position not in existing_wall_positions
    ) if intent.build_plan.build_walls else ()

    for role in roles:
        policy = intent.item_policy
        if (
            policy.medicine_health_threshold > 0
            and role.health <= policy.medicine_health_threshold
            and has_item(role.backpack, policy.medicine_name)
        ):
            result[role.unit_id].append(
                Job(
                    role_id=role.unit_id,
                    kind=JobKind.USE_ITEM,
                    target=role.position,
                    priority=priorities.emergency_item,
                    value=float(policy.medicine_health_threshold - role.health),
                    name=policy.medicine_name,
                )
            )

    for worker in workers:
        _add_recall_jobs(
            result[worker.unit_id],
            worker,
            observation,
            world,
            priority=priorities.recall,
        )
        spendable_gold = max(0, observation.our.gold - intent.gold_reserve)
        if spendable_gold >= world.rules.weapon_build_cost:
            for site in missing_weapon_sites:
                path = path_to_interaction(world, worker.position, site.position)
                if path is not None:
                    result[worker.unit_id].append(
                        Job(
                            role_id=worker.unit_id,
                            kind=JobKind.BUILD_WEAPON,
                            target=site.position,
                            priority=priorities.build_weapon,
                            value=-float(path.cost),
                            name=site.weapon_type,
                        )
                    )

        backpack = Counter(worker.backpack)
        if occupied_weapon_sites and backpack[world.rules.wall_material] > 0:
            for position in missing_wall_sites:
                path = path_to_interaction(world, worker.position, position)
                if path is not None:
                    result[worker.unit_id].append(
                        Job(
                            role_id=worker.unit_id,
                            kind=JobKind.BUILD_WALL,
                            target=position,
                            priority=priorities.build_wall,
                            value=-float(path.cost),
                            name="wall",
                            quantity=world.rules.wall_material_cost,
                        )
                    )

        full = (
            worker.backpack_capacity > 0
            and len(worker.backpack) >= worker.backpack_capacity
        )
        sale_needed = full or (
            bool(missing_weapon_sites)
            and observation.our.gold < world.rules.weapon_build_cost
        )
        if sale_needed:
            _add_sell_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                priority=priorities.sell,
            )
        if worker.unit_id in medicine_buyer_ids:
            _add_medicine_purchase_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                intent,
            )
        if not full and intent.allow_mining:
            _add_mining_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                walls_missing=bool(missing_wall_sites),
                priority=priorities.collect,
            )

    for pioneer in pioneers:
        _add_recall_jobs(
            result[pioneer.unit_id],
            pioneer,
            observation,
            world,
            priority=priorities.recall,
        )
        targets = tuple((weapon.position, weapon.unit_id) for weapon in world.weapons)
        if not targets and layout.entrance is not None:
            targets = ((layout.entrance, None),)
        for target, weapon_id in targets:
            path = path_to_interaction(world, pioneer.position, target)
            if path is not None:
                result[pioneer.unit_id].append(
                    Job(
                        role_id=pioneer.unit_id,
                        kind=JobKind.PREPOSITION,
                        target=target,
                        priority=priorities.preposition,
                        value=-float(path.cost),
                        weapon_id=weapon_id,
                    )
                )

    frozen = {
        role_id: tuple(sorted(jobs, key=lambda value: value.sort_key))
        for role_id, jobs in result.items()
    }
    return MappingProxyType(frozen)


def _add_recall_jobs(
    jobs: list[Job],
    role: UnitState,
    observation: Observation,
    world: WorldGrid,
    *,
    priority: int,
) -> None:
    rounds_left = 71 - observation.time.round_in_phase
    for weapon in world.weapons:
        path = path_to_interaction(world, role.position, weapon.position)
        if path is None:
            continue
        if rounds_left <= path.cost + world.rules.twilight_safety_margin:
            jobs.append(
                Job(
                    role_id=role.unit_id,
                    kind=JobKind.RECALL,
                    target=weapon.position,
                    priority=priority,
                    value=-float(path.cost),
                    weapon_id=weapon.unit_id,
                )
            )


def _add_sell_jobs(
    jobs: list[Job],
    worker: UnitState,
    observation: Observation,
    world: WorldGrid,
    *,
    priority: int,
) -> None:
    backpack = Counter(worker.backpack)
    prices = {item.name: item.price for item in observation.vendor_shop}
    sale_options = tuple(
        sorted(
            (
                (name, count, prices[name] * count)
                for name, count in backpack.items()
                if count > 0 and name in prices
            ),
            key=lambda value: (-value[2], value[0]),
        )
    )
    if not sale_options:
        return
    name, quantity, total = sale_options[0]
    for vendor in world.positions_for_zone("vendor"):
        path = path_to_interaction(world, worker.position, vendor)
        if path is not None:
            jobs.append(
                Job(
                    role_id=worker.unit_id,
                    kind=JobKind.SELL,
                    target=vendor,
                    priority=priority,
                    value=float(total) - path.cost / 1000,
                    name=name,
                    quantity=quantity,
                )
            )


def _medicine_buyer_ids(
    workers: tuple[UnitState, ...],
    observation: Observation,
    world: WorldGrid,
    intent: StrategicIntent,
) -> frozenset[int]:
    policy = intent.item_policy
    current_stock = sum(
        count_item(role.backpack, policy.medicine_name)
        for role in world.friendly_roles
    )
    shortage = max(0, policy.medicine_stock - current_stock)
    prices = {item.name: item.price for item in observation.weapon_shop}
    price = prices.get(policy.medicine_name)
    if shortage == 0 or price is None or price <= 0:
        return frozenset()
    affordable = max(0, observation.our.gold - intent.gold_reserve) // price
    purchase_count = min(shortage, affordable)
    if purchase_count == 0:
        return frozenset()

    shops = world.positions_for_zone('weaponShop')
    candidates: list[tuple[int, int]] = []
    for worker in workers:
        costs = tuple(
            path.cost
            for shop in shops
            if (path := path_to_interaction(world, worker.position, shop))
            is not None
        )
        if costs:
            candidates.append((min(costs), worker.unit_id))
    candidates.sort()
    return frozenset(
        worker_id for _, worker_id in candidates[:purchase_count]
    )


def _add_medicine_purchase_jobs(
    jobs: list[Job],
    worker: UnitState,
    observation: Observation,
    world: WorldGrid,
    intent: StrategicIntent,
) -> None:
    policy = intent.item_policy
    if policy.medicine_stock <= 0:
        return
    current_stock = sum(
        count_item(role.backpack, policy.medicine_name)
        for role in world.friendly_roles
    )
    if current_stock >= policy.medicine_stock:
        return
    prices = {item.name: item.price for item in observation.weapon_shop}
    price = prices.get(policy.medicine_name)
    if price is None or observation.our.gold - intent.gold_reserve < price:
        return
    for shop in world.positions_for_zone('weaponShop'):
        path = path_to_interaction(world, worker.position, shop)
        if path is not None:
            jobs.append(
                Job(
                    role_id=worker.unit_id,
                    kind=JobKind.BUY,
                    target=shop,
                    priority=intent.day_priorities.purchase,
                    value=float(-path.cost),
                    name=policy.medicine_name,
                    quantity=1,
                )
            )


def _add_mining_jobs(
    jobs: list[Job],
    worker: UnitState,
    observation: Observation,
    world: WorldGrid,
    *,
    walls_missing: bool,
    priority: int,
) -> None:
    prices = {item.name: item.price for item in observation.vendor_shop}
    for zone in observation.zones:
        if zone.neutral_type not in {"stone", "iron", "copper"}:
            continue
        path = path_to_interaction(world, worker.position, zone.position)
        if path is None:
            continue
        score = prices.get(zone.neutral_type, 0) / (path.cost + 1)
        if walls_missing and zone.neutral_type == world.rules.wall_material:
            score += 1_000_000
        jobs.append(
            Job(
                role_id=worker.unit_id,
                kind=JobKind.COLLECT,
                target=zone.position,
                priority=priority,
                value=float(score),
                name=zone.neutral_type,
                quantity=1,
            )
        )
