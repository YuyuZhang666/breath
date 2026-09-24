import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.build_recovery import (
    BuildFailureRecord,
    BuildRecoveryState,
)
from future_war_agent.strategy.build_recovery import (
    BuildFailureRecord,
    BuildRecoveryState,
)
from future_war_agent.strategy.jobs import (
    JobKind,
    calculate_stone_reserve,
    generate_day_jobs,
)
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.policy import (
    BuildPlan,
    DayPriorities,
    ItemPolicy,
    StrategicIntent,
)
from future_war_agent.strategy.world import WorldGrid
from future_war_agent.telemetry import TelemetryRecorder
from tests.strategy_helpers import observation, robot, unit


class DayJobTests(unittest.TestCase):
    def jobs(self, observed):
        world = WorldGrid.from_observation(observed)
        return generate_day_jobs(observed, world, build_defensive_layout(world))

    def test_twilight_recall_outranks_ordinary_jobs(self) -> None:
        observed = observation(
            round_no=70,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
            ),
            zones=(Zone(Position(2, 2), "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.RECALL)

    def test_affordable_missing_weapon_is_a_build_job(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, "worker"),
                unit(10013, 5, 5, "station", level=1),
            ),
            gold=25,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.BUILD_WEAPON, {job.kind for job in jobs[10010]})

    def test_first_weapon_build_failure_cools_down_same_target(self) -> None:
        observed = observation(
            round_no=10,
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 7, 7, 'station', level=1),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        planned = next(
            site for site in layout.weapon_sites
            if site.weapon_type == 'rocket'
        )
        recovery = BuildRecoveryState(
            failures=(
                BuildFailureRecord(
                    name='rocket',
                    target=planned.position,
                    consecutive_failures=1,
                    last_failure_round=10,
                    cooldown_until_round=10,
                ),
            )
        )

        jobs = generate_day_jobs(
            observed,
            world,
            layout,
            build_recovery=recovery,
        )

        self.assertFalse(
            any(
                job.kind is JobKind.BUILD_WEAPON
                and job.name == 'rocket'
                for job in jobs[10010]
            )
        )

    def test_repeated_weapon_build_failure_reroutes_target(self) -> None:
        observed = observation(
            round_no=10,
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 7, 7, 'station', level=1),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        planned = next(
            site for site in layout.weapon_sites
            if site.weapon_type == 'rocket'
        )
        recovery = BuildRecoveryState(
            failures=(
                BuildFailureRecord(
                    name='rocket',
                    target=planned.position,
                    consecutive_failures=2,
                    last_failure_round=10,
                    cooldown_until_round=11,
                ),
            )
        )
        recorder = TelemetryRecorder()
        token = recorder.begin()

        jobs = generate_day_jobs(
            observed,
            world,
            layout,
            build_recovery=recovery,
            telemetry=recorder,
        )
        sample = recorder.finish(token)

        rocket_job = next(
            job
            for job in jobs[10010]
            if job.kind is JobKind.BUILD_WEAPON
            and job.name == 'rocket'
        )
        self.assertNotEqual(rocket_job.target, planned.position)
        self.assertNotIn(rocket_job.target, layout.wall_sites)
        self.assertNotIn(rocket_job.target, layout.controller_sites)
        self.assertTrue(sample.weapon_build_reroute_log)

    def test_full_backpack_generates_unload_job(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    "worker",
                    backpack_capacity=2,
                    backpack=("iron", "iron"),
                ),
            ),
            zones=(Zone(Position(5, 5), "vendor"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.SELL)
        self.assertEqual(jobs[10010][0].quantity, 2)

    def test_low_gold_with_minerals_generates_proactive_sale(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker", backpack=("copper",)),
                unit(10013, 5, 5, "station", level=1),
            ),
            zones=(Zone(Position(5, 2), "vendor"),),
            vendor_shop=(ShopItem("copper", 8),),
            gold=10,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.SELL, {job.kind for job in jobs[10010]})

    def test_stone_is_preferred_while_wall_is_missing(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(
                Zone(Position(3, 2), "stone"),
                Zone(Position(2, 3), "copper"),
            ),
            vendor_shop=(ShopItem("stone", 1), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        mining = next(job for job in jobs[10010] if job.kind is JobKind.COLLECT)
        self.assertEqual(mining.name, "stone")

    def test_empty_stone_pipeline_boosts_stone_mining_priority(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(
                Zone(Position(3, 2), "stone"),
                Zone(Position(2, 3), "iron"),
            ),
            vendor_shop=(ShopItem("stone", 1), ShopItem("iron", 5)),
        )

        jobs = self.jobs(observed)

        collect = [job for job in jobs[10010] if job.kind is JobKind.COLLECT]
        stone = next(job for job in collect if job.name == "stone")
        iron = next(job for job in collect if job.name == "iron")
        self.assertGreater(stone.priority, iron.priority)

    def test_partial_batch_keeps_collecting_above_wall_build(self) -> None:
        observed = observation(
            round_no=20,
            our_units=(
                unit(10010, 2, 2, "worker", backpack=("stone",)),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(Zone(Position(3, 2), "stone"),),
            vendor_shop=(ShopItem("stone", 1),),
        )

        jobs = self.jobs(observed)

        collect = next(job for job in jobs[10010] if job.kind is JobKind.COLLECT)
        wall = next(job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL)
        self.assertGreater(collect.priority, wall.priority)

    def test_late_day_with_stone_builds_wall_instead_of_batching(self) -> None:
        observed = observation(
            round_no=64,
            our_units=(
                unit(10010, 4, 3, "worker", backpack=("stone",)),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(Zone(Position(3, 2), "stone"),),
            vendor_shop=(ShopItem("stone", 1),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)

    def test_reserved_stone_is_not_sold_when_backpack_is_full(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    'worker',
                    backpack_capacity=5,
                    backpack=('stone',) * 5,
                ),
                unit(10013, 7, 7, 'station', level=1),
                unit(10020, 6, 6, 'gatling', level=1),
            ),
            zones=(Zone(Position(3, 2), 'vendor'),),
            vendor_shop=(ShopItem('stone', 1),),
        )

        jobs = self.jobs(observed)

        self.assertFalse(
            any(
                job.kind is JobKind.SELL and job.name == 'stone'
                for job in jobs[10010]
            )
        )

    def test_only_stone_above_reserve_is_sold(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    'worker',
                    backpack_capacity=20,
                    backpack=('stone',) * 20,
                ),
                unit(10013, 7, 7, 'station', level=1),
                unit(10020, 6, 6, 'gatling', level=1),
            ),
            zones=(Zone(Position(3, 2), 'vendor'),),
            vendor_shop=(ShopItem('stone', 1),),
        )

        jobs = self.jobs(observed)

        sale = next(
            job
            for job in jobs[10010]
            if job.kind is JobKind.SELL and job.name == 'stone'
        )
        self.assertEqual(sale.quantity, 6)

    def test_sufficient_stone_stock_does_not_force_more_stone_mining(
        self,
    ) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    'worker',
                    backpack=('stone',) * 14,
                ),
                unit(10013, 7, 7, 'station', level=1),
                unit(10020, 6, 6, 'gatling', level=1),
            ),
            zones=(
                Zone(Position(3, 2), 'stone'),
                Zone(Position(2, 4), 'copper'),
            ),
            vendor_shop=(ShopItem('stone', 1), ShopItem('copper', 20)),
        )

        jobs = self.jobs(observed)

        mining = next(
            job for job in jobs[10010] if job.kind is JobKind.COLLECT
        )
        self.assertEqual(mining.name, 'copper')

    def test_forecast_wall_losses_raise_stone_reserve(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, 'worker'),
                unit(10013, 7, 7, 'station', level=1),
            ),
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        reserve = calculate_stone_reserve(
            world,
            layout,
            StrategicIntent(),
            expected_wall_losses=8,
        )

        self.assertEqual(reserve, 14)

    def test_early_weapon_priority_does_not_hide_wall_jobs(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=75,
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WEAPON)
        wall_job = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )
        self.assertGreater(jobs[10010][0].priority, wall_job.priority)

    def test_critical_wall_is_available_immediately_without_weapon(
        self,
    ) -> None:
        observed = observation(
            round_no=1,
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=0,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertIn(jobs[10010][0].target, layout.critical_wall_sites)

    def test_early_two_tower_target_precedes_stone_collection(
        self,
    ) -> None:
        observed = observation(
            round_no=10,
            our_units=(
                unit(10010, 2, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            zones=(Zone(Position(3, 5), 'stone'),),
            gold=75,
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WEAPON)
        stone_job = next(
            job
            for job in jobs[10010]
            if job.kind is JobKind.COLLECT and job.name == 'stone'
        )
        self.assertGreater(jobs[10010][0].priority, stone_job.priority)

    def test_one_core_weapon_keeps_second_tower_deadline_and_wall_option(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
                unit(10020, 7, 7, 'gatling', level=1),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WEAPON)
        wall_job = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )
        self.assertIn(wall_job.target, layout.critical_wall_sites)

    def test_day_two_rebuilds_critical_walls_even_if_weapons_were_lost(
        self,
    ) -> None:
        observed = observation(
            round_no=131,
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        recorder = TelemetryRecorder()
        token = recorder.begin()

        jobs = generate_day_jobs(
            observed,
            world,
            layout,
            telemetry=recorder,
        )
        sample = recorder.finish(token)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertIn(jobs[10010][0].target, layout.critical_wall_sites)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.wall_plan_stage, 'opening_critical')
        self.assertEqual(sample.wall_blocker, 'none')
        self.assertEqual(sample.core_weapon_ready_count, 0)
        self.assertGreater(sample.wall_job_count, 0)

    def test_occupied_front_candidates_do_not_starve_later_wall_sites(
        self,
    ) -> None:
        base_units = (
            unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
            unit(10013, 5, 5, 'station', level=1),
            unit(10020, 7, 7, 'gatling', level=1),
        )
        initial = observation(our_units=base_units, gold=75)
        initial_layout = build_defensive_layout(
            WorldGrid.from_observation(initial)
        )
        occupied = initial_layout.wall_sites[:4]
        blockers = tuple(
            unit(11000 + index, site.x, site.y, 'pioneer')
            for index, site in enumerate(occupied)
        )
        observed = observation(
            our_units=base_units + blockers,
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        jobs = generate_day_jobs(observed, world, layout)
        wall_jobs = tuple(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )

        self.assertTrue(wall_jobs)
        self.assertNotIn(wall_jobs[0].target, occupied)
        self.assertIn(wall_jobs[0].target, layout.wall_sites[4:])

    def test_failed_wall_target_cools_down_and_next_gap_is_used(self) -> None:
        units = (
            unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
            unit(10013, 5, 5, 'station', level=1),
            unit(10020, 7, 7, 'gatling', level=1),
        )
        initial = observation(our_units=units, gold=75)
        initial_world = WorldGrid.from_observation(initial)
        initial_layout = build_defensive_layout(initial_world)
        failed_target = initial_layout.critical_wall_sites[0]
        previous = Decision(
            commands={10010: Action.build('wall', failed_target)}
        )
        observed = observation(
            round_no=2,
            our_units=units,
            gold=75,
            last_action_results={10010: False},
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        recorder = TelemetryRecorder()
        token = recorder.begin()

        jobs = generate_day_jobs(
            observed,
            world,
            layout,
            previous_decision=previous,
            telemetry=recorder,
        )
        sample = recorder.finish(token)
        wall_jobs = tuple(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )

        self.assertTrue(wall_jobs)
        self.assertNotIn(failed_target, {job.target for job in wall_jobs})
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample.wall_failed_build_log), 1)

    def test_critical_opening_walls_start_after_core_weapons(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
                unit(10020, 7, 7, 'gatling', level=1),
                unit(10030, 4, 4, 'railgun', level=1),
                unit(10040, 4, 7, 'rocket', level=1),
            ),
            gold=0,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertIn(jobs[10010][0].target, layout.critical_wall_sites)

    def test_custom_short_loadout_opens_walls_after_its_last_weapon(self) -> None:
        plan = BuildPlan(
            weapon_loadout=('gatling',),
            minimum_weapons_before_walls=3,
        )
        initial = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=0,
        )
        initial_world = WorldGrid.from_observation(initial)
        layout = build_defensive_layout(initial_world, plan)
        site = layout.weapon_sites[0]
        observed = observation(
            our_units=initial.our.units + (
                unit(10020, site.position.x, site.position.y, 'gatling'),
            ),
            gold=0,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world, plan),
            StrategicIntent(build_plan=plan),
        )

        self.assertIn(
            JobKind.BUILD_WALL,
            {job.kind for job in jobs[10010]},
        )

    def test_after_critical_walls_remaining_weapon_build_resumes(self) -> None:
        initial = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
                unit(10020, 7, 7, 'gatling', level=1),
            ),
            gold=75,
        )
        initial_world = WorldGrid.from_observation(initial)
        initial_layout = build_defensive_layout(initial_world)
        walls = tuple(
            unit(10100 + index, site.x, site.y, 'wall')
            for index, site in enumerate(initial_layout.critical_wall_sites)
        )
        observed = observation(
            our_units=initial.our.units + walls,
            gold=75,
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WEAPON)

    def test_after_twelve_walls_workers_continue_three_side_infill(self) -> None:
        base = observation(
            our_units=(
                unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=0,
        )
        layout = build_defensive_layout(WorldGrid.from_observation(base))
        walls = tuple(
            unit(10100 + index, site.x, site.y, 'wall')
            for index, site in enumerate(layout.critical_wall_sites)
        )
        weapons = tuple(
            unit(
                10200 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites)
        )
        observed = observation(
            our_units=base.our.units + walls + weapons,
            gold=0,
        )

        jobs = self.jobs(observed)

        wall_job = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )
        self.assertIn(wall_job.target, layout.wall_sites[12:])

    def test_destroyed_wall_rebuild_outranks_remaining_core_weapons(self) -> None:
        base_units = (
            unit(10010, 2, 5, 'worker', backpack=('stone',) * 5),
            unit(10013, 5, 5, 'station', level=1),
            unit(10020, 7, 7, 'gatling', level=1),
        )
        initial = observation(round_no=131, our_units=base_units, gold=75)
        initial_layout = build_defensive_layout(
            WorldGrid.from_observation(initial)
        )
        critical_walls = tuple(
            unit(10100 + index, site.x, site.y, 'wall')
            for index, site in enumerate(initial_layout.critical_wall_sites)
        )
        rebuild_target = initial_layout.wall_sites[
            len(initial_layout.critical_wall_sites)
        ]
        observed = observation(
            round_no=132,
            our_units=base_units + critical_walls,
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        recorder = TelemetryRecorder()
        token = recorder.begin()

        jobs = generate_day_jobs(
            observed,
            world,
            layout,
            previously_built_wall_sites=frozenset({rebuild_target}),
            telemetry=recorder,
        )
        sample = recorder.finish(token)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertEqual(jobs[10010][0].target, rebuild_target)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.wall_plan_stage, 'daily_rebuild')
        self.assertEqual(sample.rebuild_wall_gap_count, 1)
        self.assertGreater(sample.new_wall_gap_count, 0)

    def test_later_rebuild_prefers_attack_lane_over_nearer_rear_gap(self) -> None:
        initial = observation(
            our_units=(
                unit(10010, 2, 4, 'worker', backpack=('stone',) * 5),
                unit(10013, 5, 5, 'station', level=1),
                unit(10020, 7, 7, 'gatling', level=1),
                unit(10030, 4, 4, 'railgun', level=1),
                unit(10040, 4, 7, 'rocket', level=1),
            ),
            robots=(robot(501, 12, 7), robot(502, 11, 8)),
            gold=75,
        )
        initial_world = WorldGrid.from_observation(initial)
        layout = build_defensive_layout(initial_world)
        attack_lane_gap = layout.wall_sites[3]
        rear_gap = layout.wall_sites[-2]
        walls = tuple(
            unit(10100 + index, site.x, site.y, 'wall')
            for index, site in enumerate(layout.wall_sites)
            if site not in {attack_lane_gap, rear_gap}
        )
        observed = observation(
            our_units=initial.our.units + walls,
            robots=initial.robots,
            gold=75,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
        )

        rebuilds = tuple(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WALL
        )
        self.assertEqual(rebuilds[0].target, attack_lane_gap)
        self.assertGreater(rebuilds[0].priority, rebuilds[1].priority)

    def test_rear_access_lane_is_not_closed_at_twilight(self) -> None:
        base = observation(
            round_no=60,
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        base_world = WorldGrid.from_observation(base)
        layout = build_defensive_layout(base_world)
        gate = layout.entrance
        self.assertIsNotNone(gate)
        walls = tuple(
            unit(10100 + index, site.x, site.y, 'wall')
            for index, site in enumerate(layout.wall_sites)
        )
        observed = observation(
            round_no=60,
            our_units=(
                unit(
                    10010,
                    5,
                    5,
                    'worker',
                    backpack=('stone',),
                ),
                unit(10013, 5, 5, 'station', level=1),
            ) + walls,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
        )

        self.assertTrue(all(
            job.kind is not JobKind.BUILD_WALL or job.target != gate
            for job in jobs[10010]
        ))

    def test_price_per_distance_selects_mineral_after_defense(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, "worker"),),
            zones=(
                Zone(Position(3, 1), "iron"),
                Zone(Position(6, 1), "copper"),
            ),
            vendor_shop=(ShopItem("iron", 2), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].name, "copper")

    def test_two_workers_may_receive_same_adjacent_mine(self) -> None:
        mine = Position(4, 4)
        observed = observation(
            our_units=(
                unit(10010, 3, 4, "worker"),
                unit(10012, 4, 3, "worker"),
            ),
            zones=(Zone(mine, "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].target, mine)
        self.assertEqual(jobs[10012][0].target, mine)

    def test_gold_reserve_prevents_weapon_build_job(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            StrategicIntent(gold_reserve=25),
        )

        self.assertNotIn(JobKind.BUILD_WEAPON, {job.kind for job in jobs[10010]})

    def test_last_core_weapon_can_consume_opening_reserve(self) -> None:
        initial = observation(
            round_no=20,
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=25,
        )
        initial_world = WorldGrid.from_observation(initial)
        layout = build_defensive_layout(initial_world)
        built = tuple(
            unit(
                10020 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites[:2])
        )
        observed = observation(
            round_no=20,
            our_units=initial.our.units + built,
            gold=25,
        )
        world = WorldGrid.from_observation(observed)
        missing_type = layout.weapon_sites[2].weapon_type

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            StrategicIntent(
                gold_reserve=25,
                reserve_eligible_actions=frozenset(
                    {('build', missing_type)}
                ),
            ),
        )

        build = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WEAPON
        )
        self.assertEqual(build.name, missing_type)
        self.assertTrue(build.reserve_eligible)

    def test_intent_priorities_flow_into_generated_jobs(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            StrategicIntent(day_priorities=DayPriorities(build_weapon=777)),
        )

        weapon_job = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WEAPON
        )
        self.assertEqual(weapon_job.priority, 777)

    def test_medicine_purchase_respects_stock_and_reserve(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, 'worker'),),
            zones=(Zone(Position(2, 2), 'weaponShop'),),
            weapon_shop=(ShopItem('Medicine', 10),),
            gold=40,
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            gold_reserve=25,
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            intent,
        )

        purchase = next(job for job in jobs[10010] if job.kind is JobKind.BUY)
        self.assertEqual((purchase.name, purchase.quantity), ('Medicine', 1))

    def test_medicine_stock_target_caps_concurrent_purchase_jobs(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 1, 2, 'worker'),
                unit(10011, 2, 1, 'worker'),
            ),
            zones=(Zone(Position(2, 2), 'weaponShop'),),
            weapon_shop=(ShopItem('Medicine', 10),),
            gold=40,
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            intent,
        )

        purchases = [
            job
            for role_jobs in jobs.values()
            for job in role_jobs
            if job.kind is JobKind.BUY and job.name == 'Medicine'
        ]
        self.assertEqual(len(purchases), 1)

    def test_lowercase_inventory_medicine_counts_toward_stock_target(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 1, 1, "worker", backpack=("medicine",)),
            ),
            zones=(Zone(Position(2, 2), "weaponShop"),),
            weapon_shop=(ShopItem("Medicine", 10),),
            gold=40,
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            intent,
        )

        self.assertNotIn(
            JobKind.BUY,
            {job.kind for role_jobs in jobs.values() for job in role_jobs},
        )


    def test_wall_sites_are_partitioned_between_workers(self) -> None:
        observed = observation(
            round_no=131,
            our_units=(
                unit(10010, 3, 3, 'worker', backpack=('stone',) * 4),
                unit(10012, 3, 3, 'worker', backpack=('stone',) * 4),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        jobs = generate_day_jobs(observed, world, layout)

        jobs_by_worker = {
            role_id: {
                job.target: job
                for job in worker_jobs
                if job.kind is JobKind.BUILD_WALL
            }
            for role_id, worker_jobs in jobs.items()
        }
        shared = (
            set(jobs_by_worker[10010]) & set(jobs_by_worker[10012])
        )
        self.assertTrue(shared)
        for target in shared:
            rank = layout.wall_sites.index(target)
            owner_id = 10010 if rank % 2 == 0 else 10012
            other_id = 10012 if owner_id == 10010 else 10010
            self.assertGreater(
                jobs_by_worker[owner_id][target].value,
                jobs_by_worker[other_id][target].value,
            )


if __name__ == "__main__":
    unittest.main()
