import unittest
from dataclasses import replace

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.night import (
    ControllerAssignmentCache,
    assign_controllers,
    generate_night_candidates,
    plan_emergency_night_fire,
)
from future_war_agent.strategy.policy import ItemPolicy, StrategicIntent
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class NightPolicyTests(unittest.TestCase):
    def test_emergency_fire_uses_only_ready_adjacent_weapons(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, 'worker'),
                unit(10011, 8, 5, 'pioneer'),
                unit(10013, 1, 1, 'station', level=1),
                unit(
                    10020,
                    5,
                    5,
                    'gatling',
                    level=1,
                    attack_range=5,
                ),
                unit(
                    10021,
                    9,
                    5,
                    'railgun',
                    level=1,
                    attack_range=6,
                ),
                unit(
                    10022,
                    5,
                    8,
                    'rocket',
                    level=1,
                    attack_range=6,
                    cooldown=1,
                ),
            ),
            robots=(robot(30001, 6, 5), robot(30002, 7, 5)),
        )

        plan = plan_emergency_night_fire(
            observed,
            clock=lambda: 0.0,
            deadline=0.02,
        )

        self.assertFalse(plan.deadline_hit)
        self.assertEqual(set(plan.decision.commands), {10020, 10021})
        self.assertTrue(
            all(
                action.kind is ActionKind.ATTACK
                for action in plan.decision.commands.values()
            )
        )
        self.assertEqual(validate_decision(observed, plan.decision), plan.decision)

    def test_emergency_fire_never_targets_other_team_robots(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, 'worker'),
                unit(10013, 1, 1, 'station', level=1),
                unit(
                    10020,
                    5,
                    5,
                    'railgun',
                    level=1,
                    attack_range=6,
                ),
            ),
            robots=(robot(30001, 6, 5, target_team='opponent'),),
        )

        plan = plan_emergency_night_fire(
            observed,
            clock=lambda: 0.0,
            deadline=0.02,
        )

        self.assertEqual(plan.decision.commands, {})

    def test_emergency_fire_honors_an_expired_micro_deadline(self) -> None:
        observed = observation(round_no=71)

        plan = plan_emergency_night_fire(
            observed,
            clock=lambda: 1.0,
            deadline=1.0,
        )

        self.assertTrue(plan.deadline_hit)
        self.assertEqual(plan.decision.commands, {})

    def test_assignment_minimizes_total_distance(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 10, 10, "pioneer"),
                unit(10020, 2, 2, "gatling", level=1, attack_range=4),
                unit(10030, 9, 9, "railgun", level=1, attack_range=6),
            ),
        )
        world = WorldGrid.from_observation(observed)

        assignments = assign_controllers(observed, world)

        paired = {(item.role_id, item.weapon_id) for item in assignments}
        self.assertEqual(paired, {(10010, 10020), (10011, 10030)})

    def test_assignment_stand_cells_are_unique(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 1, 3, "pioneer"),
                unit(10020, 4, 2, "gatling", level=1, attack_range=4),
                unit(10030, 4, 3, "railgun", level=1, attack_range=6),
            ),
        )
        assignments = assign_controllers(
            observed, WorldGrid.from_observation(observed)
        )
        self.assertEqual(
            len({item.stand for item in assignments}),
            len(assignments),
        )

    def test_role_moves_toward_assigned_weapon(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10020, 5, 5, "gatling", level=1, attack_range=4),
            ),
        )
        world = WorldGrid.from_observation(observed)

        choices = generate_night_candidates(observed, world)

        self.assertTrue(
            any(
                item.action and item.action.kind is ActionKind.MOVE
                for item in choices[10010]
            )
        )

    def test_ready_level_one_weapon_attacks_nearest_station_threat(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10013, 1, 1, "station", level=1),
                unit(10020, 5, 5, "gatling", level=1, attack_range=5),
            ),
            robots=(robot(30001, 3, 3), robot(30002, 7, 5)),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        attack = next(
            item
            for item in choices[10010]
            if item.action and item.action.kind is ActionKind.ATTACK
        )
        self.assertEqual(attack.action.target_positions, (Position(3, 3),))

    def test_cooling_weapon_does_not_attack(self) -> None:
        for role_type in ('gatling', 'railgun', 'rocket'):
            with self.subTest(role_type=role_type):
                observed = observation(
                    round_no=71,
                    our_units=(
                        unit(10010, 4, 5, "worker"),
                        unit(
                            10020,
                            5,
                            5,
                            role_type,
                            level=2,
                            cooldown=2,
                            attack_range=5,
                        ),
                    ),
                    robots=(robot(30001, 3, 3),),
                )
                choices = generate_night_candidates(
                    observed,
                    WorldGrid.from_observation(observed),
                )
                self.assertFalse(
                    any(
                        item.action and item.action.kind is ActionKind.ATTACK
                        for item in choices[10010]
                    )
                )

    def test_upgraded_gatling_emits_level_sized_targets(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, 'worker'),
                unit(10020, 5, 5, 'gatling', level=3, attack_range=6),
            ),
            robots=(
                robot(30001, 7, 4),
                robot(30002, 7, 5),
                robot(30003, 7, 6),
            ),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        attack = next(
            item.action
            for item in choices[10010]
            if item.action and item.action.kind is ActionKind.ATTACK
        )

        self.assertEqual(len(attack.target_positions), 3)
        self.assertEqual(len(set(attack.target_positions)), 3)

    def test_upgraded_rocket_emits_level_sized_targets(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, 'worker'),
                unit(10020, 5, 5, 'rocket', level=2, attack_range=6),
            ),
            robots=(robot(30001, 7, 4), robot(30002, 7, 6)),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        attack = next(
            item.action
            for item in choices[10010]
            if item.action and item.action.kind is ActionKind.ATTACK
        )

        self.assertEqual(len(attack.target_positions), 2)
        self.assertEqual(len(set(attack.target_positions)), 2)

    def test_assignment_cache_hits_and_invalidates_on_role_move(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, 'worker'),
                unit(10020, 5, 5, 'gatling', level=1, attack_range=5),
            ),
        )
        cache = ControllerAssignmentCache()

        first, first_hit = cache.resolve(
            observed,
            WorldGrid.from_observation(observed),
            mode_key='economy',
        )
        second, second_hit = cache.resolve(
            observed,
            WorldGrid.from_observation(observed),
            mode_key='economy',
        )
        moved = replace(
            observed,
            our=replace(
                observed.our,
                units=(
                    replace(observed.our.units[0], position=Position(2, 1)),
                    observed.our.units[1],
                ),
            ),
        )
        _, moved_hit = cache.resolve(
            moved,
            WorldGrid.from_observation(moved),
            mode_key='economy',
        )

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(first, second)
        self.assertFalse(moved_hit)

    def test_assignment_cache_keys_task_role_exclusions(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, 'worker'),
                unit(10011, 4, 6, 'pioneer'),
                unit(10020, 5, 5, 'gatling', level=1, attack_range=5),
                unit(10030, 5, 6, 'railgun', level=1, attack_range=6),
            ),
        )
        world = WorldGrid.from_observation(observed)
        cache = ControllerAssignmentCache()

        full, _ = cache.resolve(observed, world, mode_key='score')
        reserved, first_hit = cache.resolve(
            observed,
            world,
            mode_key='score',
            excluded_role_ids=frozenset({10011}),
        )
        repeated, second_hit = cache.resolve(
            observed,
            world,
            mode_key='score',
            excluded_role_ids=frozenset({10011}),
        )

        self.assertEqual({item.role_id for item in full}, {10010, 10011})
        self.assertEqual({item.role_id for item in reserved}, {10010})
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(reserved, repeated)

    def test_unassigned_role_gets_safe_defensive_candidate(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10011, 2, 2, "pioneer"),
                unit(10013, 5, 5, "station", level=1),
                unit(10020, 5, 4, "gatling", level=1, attack_range=4),
            ),
            robots=(robot(30001, 1, 1),),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        self.assertIn(10011, choices)
        self.assertGreaterEqual(len(choices[10011]), 1)

    def test_injured_role_with_medicine_gets_emergency_use_candidate(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(
                    10010,
                    4,
                    5,
                    'worker',
                    health=50,
                    backpack=('Medicine',),
                ),
                unit(10020, 5, 5, 'gatling', level=1, attack_range=5),
            ),
        )
        intent = StrategicIntent(
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            )
        )

        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
            intent,
        )

        self.assertEqual(choices[10010][0].action.kind, ActionKind.USE)
        self.assertEqual(choices[10010][0].action.name, 'Medicine')

    def test_lowercase_inventory_medicine_emits_official_action_name(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(
                    10010,
                    4,
                    5,
                    "worker",
                    health=50,
                    backpack=("medicine",),
                ),
            ),
        )
        intent = StrategicIntent(
            item_policy=ItemPolicy(medicine_health_threshold=60)
        )

        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
            intent,
        )

        self.assertEqual(choices[10010][0].action.kind, ActionKind.USE)
        self.assertEqual(choices[10010][0].action.name, "Medicine")


if __name__ == "__main__":
    unittest.main()
