import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.rules import RulesConfig, station_footprint
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class WorldGridTests(unittest.TestCase):
    def test_station_footprint_uses_formal_top_left_convention(self) -> None:
        self.assertEqual(
            station_footprint(Position(4, 5)),
            frozenset(
                {
                    Position(4, 5),
                    Position(5, 5),
                    Position(4, 6),
                    Position(5, 6),
                }
            ),
        )

    def test_station_direction_is_configurable(self) -> None:
        rules = RulesConfig(station_y_direction=-1)
        self.assertEqual(
            station_footprint(Position(4, 5), rules),
            frozenset(
                {
                    Position(4, 5),
                    Position(5, 5),
                    Position(4, 4),
                    Position(5, 4),
                }
            ),
        )

    def test_builds_separate_hard_and_friendly_occupancy(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 5, 5, "station", level=1),
                unit(10020, 7, 5, "gatling", level=1),
            ),
            enemy_units=(unit(20010, 10, 10, "worker"),),
            robots=(robot(30001, 9, 9),),
            zones=(Zone(Position(3, 3), "stone"),),
        )

        world = WorldGrid.from_observation(observed)

        self.assertEqual(world.soft_friendly, frozenset({Position(2, 2)}))
        for blocked in (
            Position(3, 3),
            Position(5, 5),
            Position(6, 6),
            Position(7, 5),
            Position(9, 9),
            Position(10, 10),
        ):
            self.assertIn(blocked, world.hard_blocked)

    def test_interaction_cells_are_adjacent_and_traversable(self) -> None:
        target = Position(3, 3)
        world = WorldGrid.from_observation(
            observation(zones=(Zone(target, "stone"),))
        )

        cells = world.interaction_cells(target)

        self.assertEqual(len(cells), 8)
        self.assertTrue(all(cell.chebyshev_distance(target) == 1 for cell in cells))
        self.assertNotIn(target, cells)

    def test_diagonal_is_not_blocked_by_orthogonal_corners(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                zones=(
                    Zone(Position(2, 1), "stone"),
                    Zone(Position(1, 2), "iron"),
                )
            )
        )

        self.assertTrue(world.can_traverse(Position(2, 2)))

    def test_build_zones_follow_station_footprint_rings(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                our_units=(unit(10013, 5, 5, 'station', level=1),),
            )
        )

        self.assertTrue(world.is_weapon_build_site(Position(4, 5)))
        self.assertTrue(world.is_wall_build_site(Position(3, 5)))
        self.assertTrue(
            world.is_legal_build_site('gatling', Position(4, 5))
        )
        self.assertTrue(world.is_legal_build_site('wall', Position(3, 5)))
        self.assertFalse(world.is_legal_build_site('wall', Position(4, 5)))
        self.assertFalse(
            world.is_legal_build_site('rocket', Position(3, 5))
        )


if __name__ == "__main__":
    unittest.main()
