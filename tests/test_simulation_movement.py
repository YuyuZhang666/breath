import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.movement import (
    MoveIntent,
    resolve_simultaneous_moves,
)
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimRole,
    SimState,
    SimStructure,
)


class SimulationMovementTests(unittest.TestCase):
    def test_role_and_robot_contesting_one_cell_both_stay(self) -> None:
        state = self._state(
            roles=(self._role(1, 4, 4),),
            robots=(self._robot(9, 6, 6),),
        )
        intents = (
            MoveIntent("role", 1, Position(4, 4), Position(5, 5)),
            MoveIntent("robot", 9, Position(6, 6), Position(5, 5)),
        )

        resolved = resolve_simultaneous_moves(state, intents)

        self.assertEqual(resolved[("role", 1)], Position(4, 4))
        self.assertEqual(resolved[("robot", 9)], Position(6, 6))

    def test_role_robot_swap_is_blocked(self) -> None:
        state = self._state(
            roles=(self._role(1, 4, 4),),
            robots=(self._robot(9, 5, 5),),
        )
        intents = (
            MoveIntent("role", 1, Position(4, 4), Position(5, 5)),
            MoveIntent("robot", 9, Position(5, 5), Position(4, 4)),
        )

        resolved = resolve_simultaneous_moves(state, intents)

        self.assertEqual(resolved[("role", 1)], Position(4, 4))
        self.assertEqual(resolved[("robot", 9)], Position(5, 5))

    def test_robot_robot_swap_is_blocked(self) -> None:
        state = self._state(
            robots=(self._robot(8, 4, 4), self._robot(9, 5, 5)),
        )
        intents = (
            MoveIntent("robot", 8, Position(4, 4), Position(5, 5)),
            MoveIntent("robot", 9, Position(5, 5), Position(4, 4)),
        )

        resolved = resolve_simultaneous_moves(state, intents)

        self.assertEqual(resolved[("robot", 8)], Position(4, 4))
        self.assertEqual(resolved[("robot", 9)], Position(5, 5))

    def test_move_into_stationary_actor_is_blocked(self) -> None:
        state = self._state(
            roles=(self._role(1, 4, 4), self._role(2, 5, 5)),
        )
        intent = MoveIntent("role", 1, Position(4, 4), Position(5, 5))

        resolved = resolve_simultaneous_moves(state, (intent,))

        self.assertEqual(resolved[("role", 1)], Position(4, 4))

    def test_move_into_static_blocker_is_blocked(self) -> None:
        state = self._state(
            roles=(self._role(1, 4, 4),),
            extra_blocked=frozenset({Position(5, 5)}),
        )
        intent = MoveIntent("role", 1, Position(4, 4), Position(5, 5))

        resolved = resolve_simultaneous_moves(state, (intent,))

        self.assertEqual(resolved[("role", 1)], Position(4, 4))

    def test_independent_moves_resolve_from_one_snapshot(self) -> None:
        state = self._state(
            roles=(self._role(1, 2, 2),),
            robots=(self._robot(9, 6, 6),),
        )
        intents = (
            MoveIntent("role", 1, Position(2, 2), Position(3, 2)),
            MoveIntent("robot", 9, Position(6, 6), Position(5, 6)),
        )

        resolved = resolve_simultaneous_moves(state, intents)

        self.assertEqual(resolved[("role", 1)], Position(3, 2))
        self.assertEqual(resolved[("robot", 9)], Position(5, 6))

    def test_three_actor_contention_is_deterministically_blocked(self) -> None:
        state = self._state(
            roles=(self._role(1, 4, 4), self._role(2, 4, 6)),
            robots=(self._robot(9, 6, 5),),
        )
        intents = (
            MoveIntent("role", 1, Position(4, 4), Position(5, 5)),
            MoveIntent("role", 2, Position(4, 6), Position(5, 5)),
            MoveIntent("robot", 9, Position(6, 5), Position(5, 5)),
        )

        first = resolve_simultaneous_moves(state, intents)
        second = resolve_simultaneous_moves(state, tuple(reversed(intents)))

        self.assertEqual(dict(first), dict(second))
        self.assertEqual(set(first.values()), {Position(4, 4), Position(4, 6), Position(6, 5)})

    @staticmethod
    def _role(role_id: int, x: int, y: int) -> SimRole:
        return SimRole(role_id, "worker", Position(x, y), 100, None, None)

    @staticmethod
    def _robot(robot_id: int, x: int, y: int) -> SimRobot:
        return SimRobot(
            robot_id,
            "smallRobot",
            Position(x, y),
            40,
            5,
            3,
            1,
            False,
        )

    @staticmethod
    def _state(
        *,
        roles: tuple[SimRole, ...] = (),
        robots: tuple[SimRobot, ...] = (),
        extra_blocked: frozenset[Position] = frozenset(),
    ) -> SimState:
        station_cells = frozenset(
            {Position(8, 8), Position(9, 8), Position(8, 9), Position(9, 9)}
        )
        station = SimStructure(1, "station", Position(8, 8), station_cells, 500, 1)
        return SimState(
            71,
            12,
            12,
            60,
            "challenger",
            station_cells | extra_blocked,
            roles,
            station,
            (),
            (),
            robots,
        )


if __name__ == "__main__":
    unittest.main()
