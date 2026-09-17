import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.pathfinding import (
    first_step_options,
    path_to_interaction,
    shortest_path,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation


class PathfindingTests(unittest.TestCase):
    def test_finds_deterministic_diagonal_shortest_path(self) -> None:
        world = WorldGrid.from_observation(observation())

        first = shortest_path(world, Position(1, 1), (Position(4, 4),))
        second = shortest_path(world, Position(1, 1), (Position(4, 4),))

        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        self.assertEqual(first.cost, 3)
        self.assertEqual(first.path[0], Position(1, 1))
        self.assertEqual(first.path[-1], Position(4, 4))

    def test_routes_around_hard_obstacles(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                zones=tuple(
                    Zone(Position(2, y), "stone") for y in range(1, 5)
                )
            )
        )

        result = shortest_path(world, Position(1, 2), (Position(3, 2),))

        self.assertIsNotNone(result)
        self.assertGreater(result.cost, 2)
        self.assertTrue(all(step not in world.hard_blocked for step in result.path))

    def test_returns_none_when_goal_is_unreachable(self) -> None:
        center = Position(2, 2)
        zones = tuple(
            Zone(Position(center.x + dx, center.y + dy), "stone")
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        )
        world = WorldGrid.from_observation(observation(zones=zones))

        self.assertIsNone(shortest_path(world, center, (Position(5, 5),)))

    def test_path_to_interaction_stops_adjacent_to_target(self) -> None:
        mine = Position(6, 6)
        world = WorldGrid.from_observation(
            observation(zones=(Zone(mine, "stone"),))
        )

        result = path_to_interaction(world, Position(1, 1), mine)

        self.assertIsNotNone(result)
        self.assertEqual(result.path[-1].chebyshev_distance(mine), 1)

    def test_first_step_options_are_distinct_and_stable(self) -> None:
        world = WorldGrid.from_observation(observation())

        options = first_step_options(
            world,
            Position(2, 2),
            (Position(6, 4),),
            limit=2,
        )

        self.assertEqual(options, tuple(dict.fromkeys(options)))
        self.assertLessEqual(len(options), 2)
        self.assertEqual(
            options,
            first_step_options(
                world,
                Position(2, 2),
                (Position(6, 4),),
                limit=2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
