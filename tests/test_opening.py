import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.jobs import JobKind, generate_day_jobs
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.opening import OpeningStage, build_opening_plan
from future_war_agent.strategy.planner import plan_turn
from future_war_agent.strategy.policy import DEFAULT_STRATEGIC_INTENT
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class OpeningDefenseTests(unittest.TestCase):
    def test_round_one_locks_stone_and_weapon_workers(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 5, 'worker'),
                unit(10012, 3, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            zones=(Zone(Position(1, 5), 'stone'),),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        opening = build_opening_plan(
            observed,
            world,
            layout,
            DEFAULT_STRATEGIC_INTENT,
        )
        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(opening.stage, OpeningStage.STONE_SUPPLY)
        self.assertEqual(opening.stone_worker_id, 10010)
        self.assertEqual(opening.weapon_worker_id, 10012)
        self.assertEqual(jobs[10010][0].kind, JobKind.COLLECT)
        self.assertEqual(jobs[10010][0].name, 'stone')
        self.assertEqual(jobs[10012][0].kind, JobKind.BUILD_WEAPON)

    def test_worker_with_stone_owns_wall_build_through_response(self) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        initial_world = WorldGrid.from_observation(base)
        layout = build_defensive_layout(initial_world)
        wall_target = layout.critical_wall_sites[0]
        stand = initial_world.interaction_cells(wall_target)[0]
        first_weapon = layout.weapon_sites[0]
        observed = observation(
            round_no=10,
            our_units=(
                unit(10010, stand.x, stand.y, 'worker', backpack=('stone',)),
                unit(10012, 2, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                unit(
                    10020,
                    first_weapon.position.x,
                    first_weapon.position.y,
                    first_weapon.weapon_type,
                    level=1,
                ),
            ),
            gold=50,
        )

        planned = plan_turn(observed)
        validated = validate_decision(observed, planned)
        payload = decision_to_payload(validated)

        action = validated.commands[10010]
        self.assertEqual(action.kind, ActionKind.BUILD)
        self.assertEqual(action.name, 'wall')
        self.assertEqual(action.target_positions, (wall_target,))
        self.assertNotEqual(
            validated.commands.get(10012),
            action,
        )
        self.assertEqual(
            payload['roleCommandMap']['10010']['action'],
            'build',
        )

    def test_second_worker_collects_own_stone_after_two_weapons(self) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        layout = build_defensive_layout(WorldGrid.from_observation(base))
        weapons = tuple(
            unit(
                20000 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites[:2])
        )
        observed = observation(
            our_units=(
                unit(10010, 2, 4, 'worker', backpack=('stone',) * 6),
                unit(10012, 2, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                *weapons,
            ),
            zones=(Zone(Position(3, 5), 'stone'),),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        opening = build_opening_plan(
            observed,
            world,
            layout,
            DEFAULT_STRATEGIC_INTENT,
        )
        jobs = generate_day_jobs(observed, world, layout)

        self.assertTrue(opening.wall_support_active)
        self.assertEqual(opening.wall_support_worker_id, 10012)
        self.assertEqual(len(opening.wall_support_targets), 3)
        self.assertEqual(
            opening.assignment_for(10012),
            'opening_wall_support',
        )
        self.assertEqual(jobs[10012][0].kind, JobKind.COLLECT)
        self.assertEqual(jobs[10012][0].name, 'stone')

    def test_two_wall_workers_receive_disjoint_wall_chains(self) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        initial_world = WorldGrid.from_observation(base)
        initial_layout = build_defensive_layout(initial_world)
        support_target = initial_layout.critical_wall_sites[-1]
        support_stand = initial_world.interaction_cells(support_target)[0]
        weapons = tuple(
            unit(
                20000 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(initial_layout.weapon_sites[:2])
        )
        observed = observation(
            our_units=(
                unit(10010, 2, 4, 'worker', backpack=('stone',) * 6),
                unit(
                    10012,
                    support_stand.x,
                    support_stand.y,
                    'worker',
                    backpack=('stone',) * 3,
                ),
                unit(10013, 5, 5, 'station', level=1),
                *weapons,
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        opening = build_opening_plan(
            observed,
            world,
            layout,
            DEFAULT_STRATEGIC_INTENT,
        )

        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertNotIn(
            jobs[10010][0].target,
            opening.wall_support_targets,
        )
        self.assertEqual(jobs[10012][0].kind, JobKind.BUILD_WALL)
        self.assertIn(
            jobs[10012][0].target,
            opening.wall_support_targets,
        )
        self.assertNotEqual(
            jobs[10010][0].target,
            jobs[10012][0].target,
        )

    def test_support_without_stone_source_resumes_third_weapon(self) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        layout = build_defensive_layout(WorldGrid.from_observation(base))
        weapons = tuple(
            unit(
                20000 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites[:2])
        )
        observed = observation(
            our_units=(
                unit(10010, 2, 4, 'worker', backpack=('stone',) * 6),
                unit(10012, 2, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                *weapons,
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)
        opening = build_opening_plan(
            observed,
            world,
            layout,
            DEFAULT_STRATEGIC_INTENT,
        )

        jobs = generate_day_jobs(observed, world, layout)

        self.assertFalse(opening.wall_support_active)
        self.assertEqual(jobs[10012][0].kind, JobKind.BUILD_WEAPON)
        self.assertEqual(jobs[10010][0].kind, JobKind.BUILD_WALL)
        self.assertEqual(opening.stone_target, 6)

    def test_three_walls_do_not_complete_nine_wall_opening(self) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        layout = build_defensive_layout(WorldGrid.from_observation(base))
        weapons = tuple(
            unit(
                20000 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites)
        )
        walls = tuple(
            unit(
                30000 + index,
                position.x,
                position.y,
                'wall',
                level=1,
            )
            for index, position in enumerate(
                layout.critical_wall_sites[:3]
            )
        )
        observed = observation(
            our_units=(
                unit(10010, 2, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                *weapons,
                *walls,
            ),
        )
        world = WorldGrid.from_observation(observed)

        opening = build_opening_plan(
            observed,
            world,
            build_defensive_layout(world),
            DEFAULT_STRATEGIC_INTENT,
        )

        self.assertTrue(opening.active)
        self.assertNotEqual(opening.stage, OpeningStage.COMPLETE)
        self.assertEqual(opening.wall_count, 3)

    def test_opening_hard_recall_outranks_mining(self) -> None:
        observed = observation(
            round_no=66,
            our_units=(
                unit(10010, 1, 1, 'worker'),
                unit(10012, 4, 5, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                unit(10020, 4, 4, 'gatling', level=1),
            ),
            zones=(Zone(Position(2, 1), 'stone'),),
        )
        world = WorldGrid.from_observation(observed)
        layout = build_defensive_layout(world)

        opening = build_opening_plan(
            observed,
            world,
            layout,
            DEFAULT_STRATEGIC_INTENT,
        )
        jobs = generate_day_jobs(observed, world, layout)

        self.assertEqual(opening.stage, OpeningStage.RECALL)
        self.assertEqual(jobs[10010][0].kind, JobKind.RECALL)

    def test_twilight_recall_reactivates_after_minimum_defense_completes(
        self,
    ) -> None:
        base = observation(
            our_units=(unit(10013, 5, 5, 'station', level=1),),
        )
        layout = build_defensive_layout(WorldGrid.from_observation(base))
        weapons = tuple(
            unit(
                20000 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites)
        )
        walls = tuple(
            unit(
                30000 + index,
                position.x,
                position.y,
                'wall',
                level=1,
            )
            for index, position in enumerate(layout.critical_wall_sites)
        )
        observed = observation(
            round_no=66,
            our_units=(
                unit(10010, 1, 1, 'worker'),
                unit(10012, 2, 1, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
                *weapons,
                *walls,
            ),
        )
        world = WorldGrid.from_observation(observed)

        opening = build_opening_plan(
            observed,
            world,
            build_defensive_layout(world),
            DEFAULT_STRATEGIC_INTENT,
        )

        self.assertTrue(opening.active)
        self.assertEqual(opening.stage, OpeningStage.RECALL)


if __name__ == '__main__':
    unittest.main()
