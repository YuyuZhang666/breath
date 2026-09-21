import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.config import Phase3Config
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimRole,
    SimState,
    SimStructure,
    SimWeapon,
)
from future_war_agent.strategy.simulation.tail import estimate_tail


class TailEstimatorTests(unittest.TestCase):
    def test_expired_deadline_stops_tail_estimation(self) -> None:
        state = self._state(
            remaining=10,
            station_health=500,
            robots=(
                SimRobot(
                    9,
                    'middleRobot',
                    Position(0, 0),
                    60,
                    10,
                    3,
                    2,
                    False,
                ),
            ),
        )

        with self.assertRaises(DeadlineExceeded):
            estimate_tail(
                state,
                clock=lambda: 1.0,
                deadline=1.0,
            )

    def test_eta_projects_damage_beyond_exact_horizon(self) -> None:
        state = self._state(
            remaining=10,
            station_health=500,
            robots=(SimRobot(9, 'middleRobot', Position(0, 0), 60, 10, 3, 2, False),),
        )

        estimate = estimate_tail(
            state,
            Phase3Config(tail_visible_roster_complete=True),
        )

        self.assertEqual(estimate.incoming_damage, 50)
        self.assertEqual(estimate.expected_station_hp_at_dawn, 450)
        self.assertEqual(estimate.critical_robot_ids, (9,))
        self.assertTrue(estimate.complete)

    def test_open_route_does_not_credit_unrelated_wall(self) -> None:
        wall = SimStructure(
            20,
            'wall',
            Position(2, 2),
            frozenset({Position(2, 2)}),
            40,
            1,
        )
        state = self._state(
            remaining=5,
            walls=(wall,),
            robots=(SimRobot(9, 'middleRobot', Position(0, 4), 60, 10, 1, 2, False),),
        )

        estimate = estimate_tail(
            state,
            Phase3Config(tail_visible_roster_complete=True),
        )

        self.assertEqual(estimate.critical_wall_ids, ())
        self.assertEqual(estimate.effective_hp, state.station.health)

    def test_minimum_closed_barrier_credits_only_one_wall_path(self) -> None:
        walls = tuple(
            SimStructure(
                20 + y,
                'wall',
                Position(2, y),
                frozenset({Position(2, y)}),
                30,
                1,
            )
            for y in range(7)
        )
        state = self._state(
            width=7,
            height=7,
            station_position=Position(4, 4),
            remaining=5,
            walls=walls,
            robots=(SimRobot(9, 'middleRobot', Position(0, 4), 60, 10, 1, 2, False),),
        )

        estimate = estimate_tail(
            state,
            Phase3Config(tail_visible_roster_complete=True),
        )

        self.assertEqual(len(estimate.critical_wall_ids), 1)
        self.assertEqual(estimate.effective_hp, state.station.health + 30)

    def test_only_proven_next_volley_kills_receive_firepower_credit(self) -> None:
        role = SimRole(
            1,
            'worker',
            Position(2, 1),
            100,
            3,
            Position(2, 1),
        )
        weapon = SimWeapon(
            3,
            'gatling',
            Position(2, 2),
            100,
            10,
            8,
            1,
            0,
        )
        robot = SimRobot(
            9,
            'middleRobot',
            Position(4, 4),
            10,
            10,
            3,
            2,
            False,
        )
        state = self._state(
            station_position=Position(5, 5),
            remaining=5,
            roles=(role,),
            weapons=(weapon,),
            robots=(robot,),
        )

        estimate = estimate_tail(
            state,
            Phase3Config(tail_visible_roster_complete=True),
        )

        self.assertEqual(estimate.incoming_damage, 50)
        self.assertEqual(estimate.future_firepower, 40)
        self.assertEqual(estimate.expected_station_hp_at_dawn, 490)

    def test_d3_without_private_calibration_is_explicitly_uncertain(self) -> None:
        state = self._state(day_no=3, remaining=5)

        estimate = estimate_tail(state)

        self.assertFalse(estimate.complete)
        self.assertIn('d3_calibration_unavailable', estimate.uncertainty_reasons)

    @staticmethod
    def _state(
        *,
        width: int = 12,
        height: int = 12,
        station_position: Position = Position(8, 8),
        station_health: int = 500,
        remaining: int = 10,
        roles: tuple[SimRole, ...] = (),
        walls: tuple[SimStructure, ...] = (),
        weapons: tuple[SimWeapon, ...] = (),
        robots: tuple[SimRobot, ...] = (),
        day_no: int = 1,
    ) -> SimState:
        station_cells = frozenset(
            {
                station_position,
                Position(station_position.x + 1, station_position.y),
                Position(station_position.x, station_position.y + 1),
                Position(station_position.x + 1, station_position.y + 1),
            }
        )
        blocked = set(station_cells)
        blocked.update(wall.position for wall in walls)
        blocked.update(weapon.position for weapon in weapons)
        return SimState(
            round_no=71,
            width=width,
            height=height,
            remaining_night_turns=remaining,
            team_type='challenger',
            static_blocked=frozenset(blocked),
            roles=roles,
            station=SimStructure(
                100,
                'station',
                station_position,
                station_cells,
                station_health,
                1,
            ),
            walls=walls,
            weapons=weapons,
            robots=robots,
            day_no=day_no,
        )


if __name__ == '__main__':
    unittest.main()
