from dataclasses import dataclass
from enum import StrEnum

from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import DAY_ROUNDS, Phase

from .defense import core_weapon_readiness
from .items import count_item
from .layout import DefensiveLayout
from .pathfinding import path_to_interaction, shortest_path
from .policy import StrategicIntent
from .rules import station_footprint
from .world import WorldGrid


class OpeningStage(StrEnum):
    INACTIVE = 'inactive'
    STONE_SUPPLY = 'opening_stone_supply'
    RETURN_TO_BASE = 'opening_return_to_base'
    BUILD_WALLS = 'opening_build_walls'
    RECALL = 'opening_recall'
    COMPLETE = 'complete'


@dataclass(frozen=True, slots=True)
class OpeningPlan:
    active: bool = False
    stage: OpeningStage = OpeningStage.INACTIVE
    stone_target: int = 0
    weapon_count: int = 0
    wall_count: int = 0
    workers_with_stone: int = 0
    workers_returned: int = 0
    controllers_ready: int = 0
    gate_closed: bool = False
    remaining_daylight: int = 0
    mine_eta: int = -1
    return_eta: int = -1
    construction_eta: int = 0
    recall_eta: int = -1

    def assignment_for(self, worker_id: int) -> str:
        if self.active:
            return 'opening_dynamic_builder'
        return 'ordinary'


def build_opening_plan(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
) -> OpeningPlan:
    remaining_daylight = (
        DAY_ROUNDS - observation.time.round_in_phase
        if observation.time.phase is Phase.DAY
        else 0
    )
    workers = tuple(
        sorted(
            (role for role in world.friendly_roles if role.role_type == 'worker'),
            key=lambda role: role.unit_id,
        )
    )
    stone_worker = workers[0] if workers else None
    existing_walls = {wall.position for wall in world.walls}
    wall_count = sum(position in existing_walls for position in layout.wall_sites)
    critical_wall_count = sum(
        position in existing_walls for position in layout.critical_wall_sites
    )
    readiness = core_weapon_readiness(
        observation.our.units,
        intent.build_plan.weapon_loadout,
    )
    workers_with_stone = sum(
        count_item(worker.backpack, world.rules.wall_material)
        >= world.rules.wall_material_cost
        for worker in workers
    )
    workers_returned = sum(
        _inside_defense(world, worker.position) for worker in workers
    )
    controllers_ready = _ready_controller_count(world)
    gate_closed = layout.entrance is None or layout.entrance in existing_walls

    common = dict(
        weapon_count=readiness.ready_count,
        wall_count=wall_count,
        workers_with_stone=workers_with_stone,
        workers_returned=workers_returned,
        controllers_ready=controllers_ready,
        gate_closed=gate_closed,
        remaining_daylight=remaining_daylight,
    )
    if (
        observation.time.day_no != 1
        or observation.time.phase is not Phase.DAY
        or stone_worker is None
        or world.our_station() is None
    ):
        return OpeningPlan(**common)

    recall_eta = _recall_eta(world)
    common.update(recall_eta=recall_eta)
    hard_recall = (
        recall_eta >= 0
        and remaining_daylight
        <= recall_eta + world.rules.twilight_safety_margin
    )
    if hard_recall:
        return OpeningPlan(
            active=True,
            stage=OpeningStage.RECALL,
            **common,
        )

    required_walls = min(
        intent.build_plan.opening_critical_wall_count,
        len(layout.critical_wall_sites),
    )
    if (
        readiness.ready_count >= readiness.required_count
        and critical_wall_count >= required_walls
    ):
        return OpeningPlan(
            active=False,
            stage=OpeningStage.COMPLETE,
            **common,
        )

    missing_critical = tuple(
        position
        for position in layout.critical_wall_sites
        if position not in existing_walls
    )
    primary_missing_critical = missing_critical
    stone_count = count_item(
        stone_worker.backpack,
        world.rules.wall_material,
    )
    stone_target = min(
        intent.build_plan.opening_stone_batch_target,
        max(0, stone_worker.backpack_capacity),
        len(primary_missing_critical) * world.rules.wall_material_cost,
    )
    mine_eta, return_eta = _stone_route_eta(
        world,
        stone_worker,
        primary_missing_critical,
    )
    construction_eta = (
        len(primary_missing_critical) * world.rules.wall_material_cost
    )
    common.update(
        stone_target=stone_target,
        mine_eta=mine_eta,
        return_eta=return_eta,
        construction_eta=construction_eta,
    )

    if not primary_missing_critical:
        return OpeningPlan(
            active=True,
            stage=OpeningStage.RETURN_TO_BASE,
            **common,
        )

    defense_started = (
        readiness.ready_count
        >= min(
            intent.build_plan.minimum_weapons_before_walls,
            len(layout.weapon_sites),
        )
        or observation.time.round_in_phase
        >= intent.build_plan.opening_wall_force_round
        or wall_count > 0
    )
    construction_started = critical_wall_count > 0
    batch_ready = stone_count >= stone_target > 0
    needed_stone = max(0, stone_target - stone_count)
    supply_eta = _sum_known(mine_eta, needed_stone, return_eta)
    can_finish_supply = (
        mine_eta >= 0
        and return_eta >= 0
        and supply_eta
        + construction_eta
        + max(0, recall_eta)
        + world.rules.twilight_safety_margin
        <= remaining_daylight
    )
    if (
        primary_missing_critical
        and not construction_started
        and not batch_ready
        and can_finish_supply
    ):
        stage = OpeningStage.STONE_SUPPLY
    elif (
        primary_missing_critical
        and stone_count >= world.rules.wall_material_cost
    ):
        stage = (
            OpeningStage.BUILD_WALLS
            if defense_started
            else OpeningStage.RETURN_TO_BASE
        )
    elif primary_missing_critical and can_finish_supply:
        stage = OpeningStage.STONE_SUPPLY
    else:
        stage = OpeningStage.RECALL
    return OpeningPlan(active=True, stage=stage, **common)


def _stone_route_eta(
    world: WorldGrid,
    worker: UnitState,
    wall_targets: tuple[Position, ...],
) -> tuple[int, int]:
    mines = world.positions_for_zone(world.rules.wall_material)
    mine_routes = tuple(
        (path.cost, mine)
        for mine in mines
        if (path := path_to_interaction(world, worker.position, mine)) is not None
    )
    if not mine_routes:
        return -1, -1
    mine_eta, mine = min(
        mine_routes,
        key=lambda item: (item[0], item[1].x, item[1].y),
    )
    if not wall_targets:
        return mine_eta, 0
    starts = world.interaction_cells(mine)
    goals = tuple(
        cell
        for target in wall_targets
        for cell in world.interaction_cells(target)
    )
    return_paths = tuple(
        path.cost
        for start in starts
        if (path := shortest_path(world, start, goals)) is not None
    )
    return mine_eta, min(return_paths, default=-1)


def _recall_eta(world: WorldGrid) -> int:
    station = world.our_station()
    if station is None:
        return -1
    targets = tuple(weapon.position for weapon in world.weapons)
    if not targets:
        targets = tuple(station_footprint(station.position, world.rules))
    costs: list[int] = []
    for role in world.friendly_roles:
        routes = tuple(
            path.cost
            for target in targets
            if (path := path_to_interaction(world, role.position, target)) is not None
        )
        if not routes:
            return -1
        costs.append(min(routes))
    return max(costs, default=0)


def _inside_defense(world: WorldGrid, position: Position) -> bool:
    station = world.our_station()
    if station is None:
        return False
    footprint = station_footprint(station.position, world.rules)
    return min(position.chebyshev_distance(cell) for cell in footprint) <= 1


def _ready_controller_count(world: WorldGrid) -> int:
    available = list(world.friendly_roles)
    ready = 0
    for weapon in sorted(world.weapons, key=lambda item: item.unit_id):
        controller = next(
            (
                role
                for role in available
                if role.position.chebyshev_distance(weapon.position) <= 1
            ),
            None,
        )
        if controller is not None:
            available.remove(controller)
            ready += 1
    return ready


def _sum_known(*values: int) -> int:
    return sum(value for value in values if value >= 0)
