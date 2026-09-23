import unittest
from unittest.mock import patch

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy import pathfinding
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

    def test_first_step_options_match_independent_shortest_path_costs(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                width=20,
                height=20,
                zones=tuple(
                    Zone(Position(5, y), 'stone') for y in range(2, 12)
                ),
            )
        )
        start = Position(2, 6)
        goals = (Position(12, 4), Position(12, 9))
        expected = []
        for dx, dy in pathfinding._STEPS:
            neighbor = Position(start.x + dx, start.y + dy)
            if not world.can_traverse(neighbor):
                continue
            remaining = shortest_path(world, neighbor, goals)
            if remaining is not None:
                expected.append(
                    (1 + remaining.cost, neighbor.x, neighbor.y, neighbor)
                )
        expected.sort()

        self.assertEqual(
            first_step_options(world, start, goals, limit=3),
            tuple(item[3] for item in expected[:3]),
        )

    def test_first_step_options_use_one_reverse_search_not_eight_astars(self) -> None:
        world = WorldGrid.from_observation(observation(width=24, height=24))

        with patch.object(
            pathfinding,
            'shortest_path',
            wraps=pathfinding.shortest_path,
        ) as shortest_spy:
            options = first_step_options(
                world,
                Position(2, 2),
                (Position(20, 20),),
                limit=2,
            )

        self.assertEqual(len(options), 2)
        shortest_spy.assert_not_called()

    def test_exact_query_cache_is_scoped_to_world_instance(self) -> None:
        start = Position(1, 2)
        goals = (Position(4, 2),)
        open_world = WorldGrid.from_observation(observation())
        blocked_world = WorldGrid.from_observation(
            observation(zones=(Zone(Position(2, 1), 'stone'),))
        )

        first = shortest_path(open_world, start, goals)
        cached = shortest_path(open_world, start, goals)
        blocked = shortest_path(blocked_world, start, goals)

        self.assertIs(first, cached)
        self.assertNotEqual(first.path, blocked.path)
        self.assertIsNot(open_world._path_cache, blocked_world._path_cache)

    def test_deadline_interrupts_search_without_caching_partial_result(self) -> None:
        world = WorldGrid.from_observation(observation(width=30, height=30))
        calls = 0

        def deadline_check() -> None:
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise TimeoutError('synthetic path deadline')

        with self.assertRaises(TimeoutError):
            shortest_path(
                world,
                Position(1, 1),
                (Position(28, 28),),
                deadline_check=deadline_check,
            )

        self.assertFalse(world._path_cache)

    def test_deadline_is_checked_before_returning_cached_path(self) -> None:
        world = WorldGrid.from_observation(observation())
        shortest_path(world, Position(1, 1), (Position(4, 4),))

        def expired() -> None:
            raise TimeoutError('expired before cache lookup')

        with self.assertRaises(TimeoutError):
            shortest_path(
                world,
                Position(1, 1),
                (Position(4, 4),),
                deadline_check=expired,
            )


if __name__ == "__main__":
    unittest.main()
