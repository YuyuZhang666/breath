import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.night import (
    assign_controllers,
    generate_night_candidates,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class NightPolicyTests(unittest.TestCase):
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

    def test_cooldown_and_high_level_weapons_do_not_attack(self) -> None:
        for level, cooldown in ((1, 2), (2, 0)):
            with self.subTest(level=level, cooldown=cooldown):
                observed = observation(
                    round_no=71,
                    our_units=(
                        unit(10010, 4, 5, "worker"),
                        unit(
                            10020,
                            5,
                            5,
                            "gatling",
                            level=level,
                            cooldown=cooldown,
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


if __name__ == "__main__":
    unittest.main()
