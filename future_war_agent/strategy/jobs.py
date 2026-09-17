from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Observation, Position, UnitState

from .layout import DefensiveLayout
from .pathfinding import path_to_interaction
from .world import WorldGrid


class JobKind(StrEnum):
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
) -> Mapping[int, tuple[Job, ...]]:
    roles = tuple(sorted(world.friendly_roles, key=lambda value: value.unit_id))
    workers = tuple(role for role in roles if role.role_type == "worker")
    pioneers = tuple(role for role in roles if role.role_type == "pioneer")
    result: dict[int, list[Job]] = {role.unit_id: [] for role in roles}

    existing_weapon_sites = {
        (weapon.position, weapon.role_type) for weapon in world.weapons
    }
    missing_weapon_sites = tuple(
        site
        for site in layout.weapon_sites
        if (site.position, site.weapon_type) not in existing_weapon_sites
    )
    occupied_weapon_sites = not missing_weapon_sites and bool(layout.weapon_sites)
    existing_wall_positions = {wall.position for wall in world.walls}
    missing_wall_sites = tuple(
        position
        for position in layout.wall_sites
        if position not in existing_wall_positions
    )

    for worker in workers:
        _add_recall_jobs(result[worker.unit_id], worker, observation, world)
        if observation.our.gold >= world.rules.weapon_build_cost:
            for site in missing_weapon_sites:
                path = path_to_interaction(world, worker.position, site.position)
                if path is not None:
                    result[worker.unit_id].append(
                        Job(
                            role_id=worker.unit_id,
                            kind=JobKind.BUILD_WEAPON,
                            target=site.position,
                            priority=BUILD_WEAPON_PRIORITY,
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
                            priority=BUILD_WALL_PRIORITY,
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
            _add_sell_jobs(result[worker.unit_id], worker, observation, world)
        if not full:
            _add_mining_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                walls_missing=bool(missing_wall_sites),
            )

    for pioneer in pioneers:
        _add_recall_jobs(result[pioneer.unit_id], pioneer, observation, world)
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
                        priority=PREPOSITION_PRIORITY,
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
                    priority=RECALL_PRIORITY,
                    value=-float(path.cost),
                    weapon_id=weapon.unit_id,
                )
            )


def _add_sell_jobs(
    jobs: list[Job],
    worker: UnitState,
    observation: Observation,
    world: WorldGrid,
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
                    priority=SELL_PRIORITY,
                    value=float(total) - path.cost / 1000,
                    name=name,
                    quantity=quantity,
                )
            )


def _add_mining_jobs(
    jobs: list[Job],
    worker: UnitState,
    observation: Observation,
    world: WorldGrid,
    *,
    walls_missing: bool,
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
                priority=COLLECT_PRIORITY,
                value=float(score),
                name=zone.neutral_type,
                quantity=1,
            )
        )
