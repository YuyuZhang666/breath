from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import Phase
from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .build_recovery import EMPTY_BUILD_RECOVERY_STATE, BuildRecoveryState
from .defense import core_weapon_readiness
from .items import count_item, has_item
from .layout import DefensiveLayout, WeaponSite
from .market import MarketView, PriceDirection
from .opening import OpeningPlan, OpeningStage, build_opening_plan
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
    reserve_eligible: bool = False
    targeted: bool = False

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
    *,
    expected_wall_losses: int = 0,
    market_view: MarketView | None = None,
    previous_decision: Decision | None = None,
    previously_built_wall_sites: frozenset[Position] = frozenset(),
    build_recovery: BuildRecoveryState = EMPTY_BUILD_RECOVERY_STATE,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
) -> Mapping[int, tuple[Job, ...]]:
    priorities = intent.day_priorities
    roles = tuple(sorted(world.friendly_roles, key=lambda value: value.unit_id))
    workers = tuple(role for role in roles if role.role_type == "worker")
    pioneers = tuple(role for role in roles if role.role_type == "pioneer")
    result: dict[int, list[Job]] = {role.unit_id: [] for role in roles}
    opening = build_opening_plan(observation, world, layout, intent)
    medicine_buyer_ids = _medicine_buyer_ids(
        workers,
        observation,
        world,
        intent,
    )

    weapon_readiness = core_weapon_readiness(
        observation.our.units,
        intent.build_plan.weapon_loadout,
    )
    missing_weapon_types = Counter(weapon_readiness.missing_types)
    missing_weapon_sites_list = []
    if intent.build_plan.build_weapons:
        for site in layout.weapon_sites:
            if missing_weapon_types[site.weapon_type] <= 0:
                continue
            missing_weapon_sites_list.append(site)
            missing_weapon_types[site.weapon_type] -= 1
    weapon_build_reroute_log: list[str] = []
    resolved_weapon_sites: list[WeaponSite] = []
    all_failed_build_positions = frozenset(
        record.target for record in build_recovery.failures
    )
    occupied_build_targets = world.hard_blocked | world.soft_friendly
    for site in missing_weapon_sites_list:
        cooldown_targets = build_recovery.cooldown_targets(
            site.weapon_type,
            observation.time.round_no,
        )
        reroute_targets = build_recovery.reroute_targets(
            site.weapon_type,
            observation.time.round_no,
        )
        reroute_reason = None
        if site.position in reroute_targets:
            reroute_reason = 'repeated_failure'
        elif site.position in occupied_build_targets:
            reroute_reason = 'occupied'
        elif site.position in cooldown_targets:
            continue
        if reroute_reason is None:
            resolved_weapon_sites.append(site)
            continue
        alternative = _alternate_weapon_site(
            world,
            layout,
            workers,
            site,
            excluded_positions=(
                all_failed_build_positions | occupied_build_targets
            ),
        )
        if alternative is None:
            weapon_build_reroute_log.append(
                f'{site.weapon_type}@({site.position.x},{site.position.y}):'
                f'{reroute_reason}:no_alternative'
            )
            continue
        resolved_weapon_sites.append(
            WeaponSite(alternative, site.weapon_type)
        )
        weapon_build_reroute_log.append(
            f'{site.weapon_type}@({site.position.x},{site.position.y})->'
            f'({alternative.x},{alternative.y}):{reroute_reason}'
        )
    missing_weapon_sites = tuple(resolved_weapon_sites)
    early_weapon_target = min(
        intent.build_plan.opening_early_weapon_target,
        weapon_readiness.required_count,
        len(layout.weapon_sites),
    )
    early_weapon_urgent = (
        observation.time.day_no == 1
        and observation.time.phase is Phase.DAY
        and weapon_readiness.ready_count < early_weapon_target
    )
    early_weapon_priority = priorities.build_weapon
    if early_weapon_urgent:
        deadline_bonus = max(
            0,
            observation.time.round_in_phase
            - intent.build_plan.opening_early_weapon_deadline_round,
        )
        early_weapon_priority = max(
            priorities.build_weapon,
            min(
                priorities.recall - 1,
                priorities.build_weapon
                + intent.build_plan.opening_early_weapon_priority_boost
                + deadline_bonus,
            ),
            min(
                priorities.recall - 1,
                priorities.build_wall
                + intent.build_plan.critical_wall_priority_boost
                + 1,
            ),
        )
    required_weapons_before_walls = min(
        intent.build_plan.minimum_weapons_before_walls,
        len(layout.weapon_sites),
    )
    existing_wall_positions = {wall.position for wall in world.walls}
    missing_wall_sites = tuple(
        position
        for position in layout.wall_sites
        if position not in existing_wall_positions
    ) if intent.build_plan.build_walls else ()
    existing_planned_wall_count = len(layout.wall_sites) - len(missing_wall_sites)
    rebuild_wall_sites = frozenset(
        position
        for position in missing_wall_sites
        if position in previously_built_wall_sites
    )
    opening_wall_forced = (
        observation.time.day_no == 1
        and observation.time.round_in_phase
        >= intent.build_plan.opening_wall_force_round
    )
    defense_started = (
        weapon_readiness.ready_count >= required_weapons_before_walls
        or opening_wall_forced
        or observation.time.day_no > 1
        or existing_planned_wall_count > 0
        or bool(rebuild_wall_sites)
    )
    failed_wall_targets, failed_wall_log = _failed_wall_builds(
        observation,
        previous_decision,
    )
    recovery_wall_targets = (
        build_recovery.cooldown_targets('wall', observation.time.round_no)
        | build_recovery.reroute_targets('wall', observation.time.round_no)
    )
    failed_wall_targets = failed_wall_targets | recovery_wall_targets
    occupied_wall_targets = world.hard_blocked | world.soft_friendly
    temporary_gate_waiting = (
        layout.entrance is not None
        and layout.entrance in missing_wall_sites
        and not _temporary_gate_ready(
            observation,
            world,
            layout,
            existing_wall_positions,
            intent,
        )
    )
    wall_rank_by_position = {
        position: rank for rank, position in enumerate(layout.wall_sites)
    }
    eligible_wall_sites = tuple(
        sorted(
            (
                position
                for position in missing_wall_sites
                if position not in occupied_wall_targets
                and position not in failed_wall_targets
                and (
                    position != layout.entrance
                    or not temporary_gate_waiting
                )
            ),
            key=lambda position: (
                position not in rebuild_wall_sites,
                wall_rank_by_position[position],
            ),
        )
    )
    stone_reserve = calculate_stone_reserve(
        world,
        layout,
        intent,
        expected_wall_losses=expected_wall_losses,
    )
    reserved_stone_by_worker = _stone_reserve_by_worker(
        workers,
        world,
        missing_wall_sites,
        stone_reserve,
    )
    current_stone = sum(
        count_item(worker.backpack, world.rules.wall_material)
        for worker in workers
    )
    wall_path_costs_by_worker: dict[int, dict[Position, int]] = {}
    for worker in workers:
        path_costs: dict[Position, int] = {}
        for position in eligible_wall_sites:
            if (
                position == layout.entrance
                and not world.is_inside_defense(worker.position)
            ):
                continue
            path = path_to_interaction(world, worker.position, position)
            if path is not None:
                path_costs[position] = path.cost
        wall_path_costs_by_worker[worker.unit_id] = path_costs
    reachable_wall_sites_by_worker = {
        worker_id: tuple(path_costs)
        for worker_id, path_costs in wall_path_costs_by_worker.items()
    }
    actionable_wall_sites = {
        position
        for worker in workers
        if count_item(worker.backpack, world.rules.wall_material)
        >= world.rules.wall_material_cost
        for position in reachable_wall_sites_by_worker[worker.unit_id]
    }
    has_actionable_stone_worker = bool(actionable_wall_sites)
    missing_critical_count = sum(
        position in missing_wall_sites for position in layout.critical_wall_sites
    )
    urgent_opening_stone = (
        opening_wall_forced
        and missing_critical_count > 0
        and not has_actionable_stone_worker
    )
    wall_quota_per_worker = (
        (
            missing_critical_count
            + max(1, len(workers))
            - 1
        )
        // max(1, len(workers))
        * world.rules.wall_material_cost
    )
    worker_count = max(1, len(workers))
    stone_shortfall = max(
        0,
        missing_critical_count * world.rules.wall_material_cost
        - current_stone,
    )
    supply_travel = 0
    if stone_shortfall:
        if opening.mine_eta < 0 or opening.return_eta < 0:
            supply_travel = 1_000_000
        else:
            supply_travel = opening.mine_eta + opening.return_eta
    estimated_wall_rounds = (
        supply_travel
        + (stone_shortfall + worker_count - 1) // worker_count
        + (missing_critical_count + worker_count - 1) // worker_count
    )
    wall_round_budget = (
        opening.remaining_daylight
        - max(0, opening.recall_eta)
        - world.rules.twilight_safety_margin
    )
    stone_pipeline_ready = (
        has_actionable_stone_worker
        and estimated_wall_rounds < wall_round_budget
    )

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
        opening_assignment = opening.assignment_for(worker.unit_id)
        opening_wall_lead = (
            opening.active
            and opening_assignment == 'opening_wall_supply'
            and missing_critical_count > 0
            and len(workers) > 1
        )
        worker_weapon_priority = early_weapon_priority
        if opening_wall_lead:
            worker_weapon_priority = min(
                worker_weapon_priority,
                priorities.build_wall - 1,
            )
        elif (
            opening.active
            and opening_assignment == 'opening_flex_builder'
            and missing_weapon_sites
            and stone_pipeline_ready
        ):
            worker_weapon_priority = max(
                worker_weapon_priority,
                min(
                    priorities.recall - 1,
                    priorities.build_wall
                    + intent.build_plan.critical_wall_priority_boost
                    + 2,
                ),
            )
        _add_recall_jobs(
            result[worker.unit_id],
            worker,
            observation,
            world,
            priority=priorities.recall,
        )
        if opening.active and opening.stage is OpeningStage.RECALL:
            _add_opening_recall_jobs(
                result[worker.unit_id],
                worker,
                world,
                priority=priorities.recall + 1,
            )
        spendable_gold = max(0, observation.our.gold - intent.gold_reserve)
        for site in missing_weapon_sites:
            reserve_eligible = (
                ('build', site.weapon_type)
                in intent.reserve_eligible_actions
            )
            if (
                spendable_gold >= world.rules.weapon_build_cost
                or (
                    reserve_eligible
                    and observation.our.gold >= world.rules.weapon_build_cost
                )
            ):
                path = path_to_interaction(world, worker.position, site.position)
                if path is not None:
                    result[worker.unit_id].append(
                        Job(
                            role_id=worker.unit_id,
                            kind=JobKind.BUILD_WEAPON,
                            target=site.position,
                            priority=worker_weapon_priority,
                            value=-float(path.cost),
                            name=site.weapon_type,
                            reserve_eligible=reserve_eligible,
                        )
                    )

        backpack = Counter(worker.backpack)
        worker_wall_sites = reachable_wall_sites_by_worker[worker.unit_id]
        worker_wall_sites = tuple(
            sorted(
                worker_wall_sites,
                key=lambda position: (
                    position not in rebuild_wall_sites,
                    position not in layout.critical_wall_sites,
                    wall_path_costs_by_worker[worker.unit_id][position],
                    wall_rank_by_position[position],
                ),
            )
        )
        if (
            defense_started
            and backpack[world.rules.wall_material]
            >= world.rules.wall_material_cost
        ):
            critical_sites = set(layout.critical_wall_sites)
            for position in worker_wall_sites[
                : intent.build_plan.max_wall_job_candidates
            ]:
                path_cost = wall_path_costs_by_worker[
                    worker.unit_id
                ].get(position)
                if path_cost is not None:
                    wall_rank = wall_rank_by_position[position]
                    priority = priorities.build_wall
                    if position in critical_sites:
                        priority += intent.build_plan.critical_wall_priority_boost
                    else:
                        priority += max(
                            0,
                            intent.build_plan.threat_wall_priority_boost
                            - wall_rank,
                        )
                    if position in rebuild_wall_sites:
                        priority += intent.build_plan.rebuild_wall_priority_boost
                    if (
                        position == layout.entrance
                        and not temporary_gate_waiting
                    ):
                        priority = max(priority, priorities.recall - 1)
                    result[worker.unit_id].append(
                        Job(
                            role_id=worker.unit_id,
                            kind=JobKind.BUILD_WALL,
                            target=position,
                            priority=priority,
                            value=-float(path_cost) - wall_rank / 1000,
                            name="wall",
                            quantity=world.rules.wall_material_cost,
                        )
                    )

        near_full = _backpack_near_full(worker)
        defense_funding_needed = (
            bool(missing_weapon_sites)
            and observation.our.gold < world.rules.weapon_build_cost
        )
        worker_reaches_wall = bool(
            reachable_wall_sites_by_worker[worker.unit_id]
        )
        defense_material_needed = (
            defense_started
            and missing_critical_count > 0
            and worker_reaches_wall
            and backpack[world.rules.wall_material] == 0
        )
        sale_needed = near_full or defense_funding_needed
        if sale_needed:
            _add_sell_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                priority=priorities.sell,
                reserved_stone=reserved_stone_by_worker.get(
                    worker.unit_id,
                    0,
                ),
                market_view=market_view,
                allow_held_sale=(
                    defense_funding_needed or defense_material_needed
                ),
            )
        if worker.unit_id in medicine_buyer_ids:
            _add_medicine_purchase_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                intent,
            )
        if not near_full and intent.allow_mining:
            dynamic_stone_target = min(
                max(0, worker.backpack_capacity),
                wall_quota_per_worker,
            )
            dynamic_stone_supply = (
                observation.time.day_no == 1
                and observation.time.phase is Phase.DAY
                and missing_critical_count > 0
                and worker_reaches_wall
                and backpack[world.rules.wall_material] < dynamic_stone_target
            )
            dynamic_stone_urgent = (
                dynamic_stone_supply
                and backpack[world.rules.wall_material]
                < world.rules.wall_material_cost
            )
            opening_supply_urgent = (
                opening_wall_lead
                and backpack[world.rules.wall_material]
                < world.rules.wall_material_cost
            )
            _add_mining_jobs(
                result[worker.unit_id],
                worker,
                observation,
                world,
                stone_needed=(
                    opening_wall_lead
                    or dynamic_stone_supply
                    or (
                        defense_started
                        and bool(missing_wall_sites)
                        and worker_reaches_wall
                        and (
                            current_stone < stone_reserve
                            or not has_actionable_stone_worker
                        )
                    )
                ),
                priority=(
                    min(
                        priorities.recall - 1,
                        early_weapon_priority + 1,
                    )
                    if opening_supply_urgent
                    else
                    priorities.build_wall
                    + intent.build_plan.critical_wall_priority_boost
                    + 1
                    if (
                        urgent_opening_stone
                        or (dynamic_stone_urgent and not early_weapon_urgent)
                    )
                    else priorities.collect
                ),
                market_view=market_view,
            )

    for pioneer in pioneers:
        _add_recall_jobs(
            result[pioneer.unit_id],
            pioneer,
            observation,
            world,
            priority=priorities.recall,
        )
        if opening.active and opening.stage is OpeningStage.RECALL:
            _add_opening_recall_jobs(
                result[pioneer.unit_id],
                pioneer,
                world,
                priority=priorities.recall + 1,
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
    wall_job_count = sum(
        job.kind is JobKind.BUILD_WALL
        for role_jobs in frozen.values()
        for job in role_jobs
    )
    planned_wall_target, planned_build_position = _planned_wall_positions(
        workers,
        frozen,
        world,
    )
    telemetry.set(
        opening_stage=opening.stage.value,
        opening_weapon_count=opening.weapon_count,
        opening_wall_count=opening.wall_count,
        opening_workers_with_stone=opening.workers_with_stone,
        opening_workers_returned=opening.workers_returned,
        opening_controllers_ready=opening.controllers_ready,
        opening_gate_closed=opening.gate_closed,
        opening_remaining_daylight=opening.remaining_daylight,
        opening_mine_eta=opening.mine_eta,
        opening_return_eta=opening.return_eta,
        opening_construction_eta=opening.construction_eta,
        opening_recall_eta=opening.recall_eta,
        opening_worker_log=_opening_worker_log(workers, frozen, opening, world),
        planned_wall_target=planned_wall_target,
        planned_build_position=planned_build_position,
        wall_plan_stage=_wall_plan_stage(
            intent,
            layout,
            weapon_readiness.ready_count,
            weapon_readiness.required_count,
            defense_started,
            missing_wall_sites,
            missing_critical_count,
            rebuild_wall_sites,
        ),
        core_weapon_ready_count=weapon_readiness.ready_count,
        core_weapon_required_count=weapon_readiness.required_count,
        planned_wall_count=len(layout.wall_sites),
        existing_planned_wall_count=existing_planned_wall_count,
        missing_wall_count=len(missing_wall_sites),
        missing_critical_wall_count=missing_critical_count,
        new_wall_gap_count=(
            len(missing_wall_sites) - len(rebuild_wall_sites)
        ),
        rebuild_wall_gap_count=len(rebuild_wall_sites),
        actionable_wall_count=len(actionable_wall_sites),
        wall_job_count=wall_job_count,
        worker_stone_count=current_stone,
        stone_reserve=stone_reserve,
        wall_blocker=_wall_blocker(
            intent=intent,
            layout=layout,
            workers=workers,
            defense_started=defense_started,
            missing_wall_sites=missing_wall_sites,
            eligible_wall_sites=eligible_wall_sites,
            failed_wall_targets=failed_wall_targets,
            current_stone=current_stone,
            wall_material_cost=world.rules.wall_material_cost,
            wall_job_count=wall_job_count,
            temporary_gate_waiting=temporary_gate_waiting,
        ),
        wall_failed_build_log=failed_wall_log,
        weapon_build_reroute_log=tuple(weapon_build_reroute_log[:8]),
    )
    return MappingProxyType(frozen)


def _alternate_weapon_site(
    world: WorldGrid,
    layout: DefensiveLayout,
    workers: tuple[UnitState, ...],
    planned: WeaponSite,
    *,
    excluded_positions: frozenset[Position],
) -> Position | None:
    station = world.our_station()
    if station is None or not workers:
        return None
    reserved_targets = {
        *(site.position for site in layout.weapon_sites),
        *layout.wall_sites,
        *layout.controller_sites,
    }
    candidates: list[tuple[int, int, int, int, Position]] = []
    for x in range(world.observation.width):
        for y in range(world.observation.height):
            position = Position(x, y)
            if (
                not world.is_weapon_build_site(position)
                or position in reserved_targets
                or position in excluded_positions
                or position in world.hard_blocked
                or position in world.soft_friendly
            ):
                continue
            controller_cells = tuple(
                cell
                for cell in world.interaction_cells(position)
                if cell not in layout.wall_sites
                and cell not in {
                    site.position for site in layout.weapon_sites
                }
            )
            if not controller_cells:
                continue
            costs = tuple(
                path.cost
                for worker in workers
                if (
                    path := path_to_interaction(
                        world,
                        worker.position,
                        position,
                    )
                ) is not None
            )
            if not costs:
                continue
            candidates.append(
                (
                    min(costs),
                    position.chebyshev_distance(planned.position),
                    position.x,
                    position.y,
                    position,
                )
            )
    return min(candidates)[-1] if candidates else None


def _failed_wall_builds(
    observation: Observation,
    previous_decision: Decision | None,
) -> tuple[frozenset[Position], tuple[str, ...]]:
    if previous_decision is None:
        return frozenset(), ()
    failed: set[Position] = set()
    log: list[str] = []
    for actor_id, succeeded in sorted(observation.last_action_results.items()):
        action = previous_decision.commands.get(actor_id)
        if (
            succeeded
            or action is None
            or action.kind is not ActionKind.BUILD
            or action.name != 'wall'
            or len(action.target_positions) != 1
        ):
            continue
        target = action.target_positions[0]
        failed.add(target)
        log.append(f'actor={actor_id}:wall@({target.x},{target.y}):failed')
    return frozenset(failed), tuple(log[:8])


def _wall_plan_stage(
    intent: StrategicIntent,
    layout: DefensiveLayout,
    ready_weapons: int,
    required_weapons: int,
    defense_started: bool,
    missing_wall_sites: tuple[Position, ...],
    missing_critical_count: int,
    rebuild_wall_sites: frozenset[Position],
) -> str:
    if not intent.build_plan.build_walls:
        return 'disabled'
    if not layout.wall_sites:
        return 'no_planned_sites'
    if not missing_wall_sites:
        return 'complete'
    if (
        layout.entrance is not None
        and missing_wall_sites == (layout.entrance,)
    ):
        return 'temporary_gate'
    if not defense_started:
        return 'waiting_for_first_weapon'
    if missing_critical_count:
        return (
            'critical_rebuild'
            if any(
                position in rebuild_wall_sites
                for position in layout.critical_wall_sites
            )
            else 'opening_critical'
        )
    if rebuild_wall_sites:
        return 'daily_rebuild'
    if ready_weapons < required_weapons:
        return 'complete_core_weapons'
    return 'daily_infill'


def _wall_blocker(
    *,
    intent: StrategicIntent,
    layout: DefensiveLayout,
    workers: tuple[UnitState, ...],
    defense_started: bool,
    missing_wall_sites: tuple[Position, ...],
    eligible_wall_sites: tuple[Position, ...],
    failed_wall_targets: frozenset[Position],
    current_stone: int,
    wall_material_cost: int,
    wall_job_count: int,
    temporary_gate_waiting: bool,
) -> str:
    if not intent.build_plan.build_walls:
        return 'walls_disabled'
    if not layout.wall_sites:
        return 'no_planned_sites'
    if not missing_wall_sites:
        return 'none'
    if not workers:
        return 'no_workers'
    if not defense_started:
        return 'waiting_for_first_weapon'
    if current_stone < wall_material_cost:
        return 'worker_without_stone'
    if temporary_gate_waiting and not eligible_wall_sites:
        return 'temporary_gate_open'
    if failed_wall_targets and not eligible_wall_sites:
        return 'failed_build_cooldown'
    if not eligible_wall_sites or wall_job_count <= 0:
        return 'blocked_or_unreachable'
    return 'none'


def _temporary_gate_ready(
    observation: Observation,
    world: WorldGrid,
    layout: DefensiveLayout,
    existing_walls: set[Position],
    intent: StrategicIntent,
) -> bool:
    gate = layout.entrance
    if (
        gate is None
        or observation.time.phase is not Phase.DAY
        or observation.time.round_in_phase
        < intent.build_plan.opening_gate_close_round
    ):
        return False
    ordinary_walls = set(layout.wall_sites) - {gate}
    if not ordinary_walls.issubset(existing_walls):
        return False
    if not world.friendly_roles or not all(
        world.is_inside_defense(role.position) for role in world.friendly_roles
    ):
        return False
    return any(
        worker.role_type == 'worker'
        and world.is_inside_defense(worker.position)
        for worker in world.friendly_roles
    )


def _add_opening_recall_jobs(
    jobs: list[Job],
    role: UnitState,
    world: WorldGrid,
    *,
    priority: int,
) -> None:
    station = world.our_station()
    targets = tuple(weapon.position for weapon in world.weapons)
    if not targets and station is not None:
        targets = (station.position,)
    for target in targets:
        path = path_to_interaction(world, role.position, target)
        if path is None:
            continue
        jobs.append(
            Job(
                role_id=role.unit_id,
                kind=JobKind.RECALL,
                target=target,
                priority=priority,
                value=-float(path.cost),
            )
        )


def _opening_worker_log(
    workers: tuple[UnitState, ...],
    jobs: Mapping[int, tuple[Job, ...]],
    opening: OpeningPlan,
    world: WorldGrid,
) -> tuple[str, ...]:
    entries: list[str] = []
    for worker in workers:
        intended = jobs.get(worker.unit_id, ())
        top = intended[0] if intended else None
        action = top.kind.value if top is not None else 'none'
        target = (
            f'{top.target.x},{top.target.y}' if top is not None else 'none'
        )
        entries.append(
            f'id={worker.unit_id}:pos={worker.position.x},{worker.position.y}:'
            f'stone={count_item(worker.backpack, world.rules.wall_material)}:'
            f'assigned={opening.assignment_for(worker.unit_id)}:'
            f'intended={action}:target={target}'
        )
    return tuple(entries)


def _planned_wall_positions(
    workers: tuple[UnitState, ...],
    jobs: Mapping[int, tuple[Job, ...]],
    world: WorldGrid,
) -> tuple[str, str]:
    worker_by_id = {worker.unit_id: worker for worker in workers}
    candidates = tuple(
        job
        for role_jobs in jobs.values()
        for job in role_jobs
        if job.kind is JobKind.BUILD_WALL
    )
    if not candidates:
        return '', ''
    job = min(candidates, key=lambda item: item.sort_key)
    worker = worker_by_id.get(job.role_id)
    if worker is None:
        return f'{job.target.x},{job.target.y}', ''
    path = path_to_interaction(world, worker.position, job.target)
    stand = path.path[-1] if path is not None else None
    return (
        f'{job.target.x},{job.target.y}',
        f'{stand.x},{stand.y}' if stand is not None else '',
    )


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
    reserved_stone: int = 0,
    market_view: MarketView | None = None,
    allow_held_sale: bool = False,
) -> None:
    backpack = Counter(worker.backpack)
    prices = {item.name: item.price for item in observation.vendor_shop}
    sale_options = tuple(
        sorted(
            (
                (
                    name,
                    (
                        max(0, count - reserved_stone)
                        if name == world.rules.wall_material
                        else count
                    ),
                    prices[name]
                    * (
                        max(0, count - reserved_stone)
                        if name == world.rules.wall_material
                        else count
                    ),
                )
                for name, count in backpack.items()
                if count > 0
                and name in prices
                and (
                    allow_held_sale
                    or market_view is None
                    or name.casefold() not in market_view.hold_items
                )
                and (
                    name != world.rules.wall_material
                    or count > reserved_stone
                )
            ),
            key=lambda value: (
                not (
                    market_view is not None
                    and value[0].casefold() in market_view.sell_items
                ),
                -value[2],
                value[0],
            ),
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
    stone_needed: bool,
    priority: int,
    market_view: MarketView | None = None,
) -> None:
    prices = {item.name: item.price for item in observation.vendor_shop}
    for zone in observation.zones:
        if zone.neutral_type not in {"stone", "iron", "copper"}:
            continue
        path = path_to_interaction(world, worker.position, zone.position)
        if path is None:
            continue
        score = prices.get(zone.neutral_type, 0) / (path.cost + 1)
        if market_view is not None:
            direction = market_view.direction(zone.neutral_type.casefold())
            if direction is PriceDirection.UP:
                score *= 1.15
            elif direction is PriceDirection.DOWN:
                score *= 0.85
        if stone_needed and zone.neutral_type == world.rules.wall_material:
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


def calculate_stone_reserve(
    world: WorldGrid,
    layout: DefensiveLayout,
    intent: StrategicIntent,
    *,
    expected_wall_losses: int = 0,
) -> int:
    if not intent.build_plan.build_walls or not layout.wall_sites:
        return 0
    existing_wall_positions = {wall.position for wall in world.walls}
    missing_wall_count = sum(
        position not in existing_wall_positions
        for position in layout.wall_sites
    )
    material_cost = world.rules.wall_material_cost
    return max(
        intent.build_plan.minimum_wall_stock,
        max(0, expected_wall_losses) * material_cost,
        missing_wall_count * material_cost,
    )


def _stone_reserve_by_worker(
    workers: tuple[UnitState, ...],
    world: WorldGrid,
    missing_wall_sites: tuple[Position, ...],
    stone_reserve: int,
) -> dict[int, int]:
    def distance_to_gap(worker: UnitState) -> tuple[int, int]:
        costs = tuple(
            path.cost
            for position in missing_wall_sites
            if (path := path_to_interaction(
                world,
                worker.position,
                position,
            )) is not None
        )
        return (min(costs) if costs else 1_000_000, worker.unit_id)

    remaining = stone_reserve
    reserved: dict[int, int] = {}
    for worker in sorted(workers, key=distance_to_gap):
        count = count_item(worker.backpack, world.rules.wall_material)
        held = min(count, remaining)
        reserved[worker.unit_id] = held
        remaining -= held
    return reserved


def _backpack_near_full(worker: UnitState) -> bool:
    if worker.backpack_capacity <= 0:
        return False
    remaining = worker.backpack_capacity - len(worker.backpack)
    margin = max(1, worker.backpack_capacity // 10)
    return remaining <= margin
