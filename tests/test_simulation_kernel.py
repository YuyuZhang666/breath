import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.candidates import (
    RoleMove,
    SimJointAction,
)
from future_war_agent.strategy.simulation.future import choose_future_action
from future_war_agent.strategy.simulation.kernel import step_simulation
from future_war_agent.strategy.simulation.robots import RobotPolicy
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimRole,
    SimState,
    SimStructure,
    SimWeapon,
)
from future_war_agent.strategy.simulation.weapons import WeaponAttack


class SimulationKernelTests(unittest.TestCase):
    def test_robot_receiving_lethal_weapon_damage_still_attacks_this_turn(self) -> None:
        station = self._station(Position(8, 1))
        weapon = self._weapon(10, "gatling", Position(1, 1))
        controller = self._role(1, Position(1, 2), weapon_id=10, stand=Position(1, 2))
        blocker = self._role(2, Position(4, 1), health=100)
        robot = self._robot(9, Position(3, 1), health=10)
        state = self._state(station, roles=(controller, blocker), weapons=(weapon,), robots=(robot,))
        action = SimJointAction(
            weapon_attacks=(WeaponAttack(10, 1, (Position(3, 1),)),)
        )

        result = step_simulation(
            state,
            action,
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(result.robots, ())
        self.assertEqual(result.roles[1].health, 95)
        self.assertEqual(result.owned_kill_score, 1)

    def test_weapon_and_robot_damage_commit_simultaneously(self) -> None:
        station = self._station(Position(8, 1))
        weapon = self._weapon(10, "gatling", Position(4, 1), health=5)
        controller = self._role(1, Position(3, 2), weapon_id=10, stand=Position(3, 2))
        robot = self._robot(9, Position(3, 1), health=10)
        state = self._state(station, roles=(controller,), weapons=(weapon,), robots=(robot,))
        action = SimJointAction(
            weapon_attacks=(WeaponAttack(10, 1, (Position(3, 1),)),)
        )

        result = step_simulation(
            state,
            action,
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(result.weapons, ())
        self.assertEqual(result.robots, ())
        self.assertEqual(result.owned_kill_score, 1)

    def test_role_and_robot_moves_from_same_snapshot_collide(self) -> None:
        station = self._station(Position(8, 8))
        role = self._role(1, Position(4, 3))
        robot = self._robot(9, Position(3, 3))
        state = self._state(station, roles=(role,), robots=(robot,))
        action = SimJointAction(
            role_moves=(RoleMove(1, Position(4, 4)),)
        )

        result = step_simulation(
            state,
            action,
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(result.roles[0].position, Position(4, 3))
        self.assertEqual(result.robots[0].position, Position(3, 3))

    def test_range_three_attack_damages_blocking_role(self) -> None:
        station = self._station(Position(8, 1))
        role = self._role(1, Position(4, 1), health=20)
        robot = self._robot(9, Position(1, 1))
        state = self._state(station, roles=(role,), robots=(robot,))

        result = step_simulation(
            state,
            SimJointAction(),
            RobotPolicy.MAIN_PATH_BLOCKER,
        )

        self.assertEqual(result.roles[0].health, 15)

    def test_dead_controller_cannot_control_weapon_next_turn(self) -> None:
        station = self._station(Position(8, 1))
        weapon = self._weapon(10, "gatling", Position(4, 2))
        controller = self._role(
            1,
            Position(4, 1),
            health=5,
            weapon_id=10,
            stand=Position(4, 1),
        )
        robot = self._robot(9, Position(3, 1))
        state = self._state(station, roles=(controller,), weapons=(weapon,), robots=(robot,))

        next_state = step_simulation(
            state,
            SimJointAction(),
            RobotPolicy.STATION_SHORTEST_PATH,
        )
        future = choose_future_action(next_state)

        self.assertEqual(next_state.roles, ())
        self.assertEqual(future.weapon_attacks, ())

    def test_dizzy_robot_waits_first_step_then_becomes_active(self) -> None:
        station = self._station(Position(8, 8))
        robot = self._robot(9, Position(3, 3), waits=True)
        state = self._state(station, robots=(robot,))

        after_wait = step_simulation(
            state,
            SimJointAction(),
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
        )
        after_active = step_simulation(
            after_wait,
            SimJointAction(),
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
        )

        self.assertEqual(after_wait.robots[0].position, Position(3, 3))
        self.assertFalse(after_wait.robots[0].waits_this_turn)
        self.assertNotEqual(after_active.robots[0].position, Position(3, 3))

    def test_fired_rocket_stays_at_three_then_decrements(self) -> None:
        station = self._station(Position(8, 8))
        rocket = self._weapon(10, "rocket", Position(1, 1), attack_power=20)
        controller = self._role(1, Position(1, 2), weapon_id=10, stand=Position(1, 2))
        robot = self._robot(9, Position(3, 3))
        state = self._state(station, roles=(controller,), weapons=(rocket,), robots=(robot,))
        action = SimJointAction(
            weapon_attacks=(WeaponAttack(10, 1, (Position(3, 3),)),)
        )

        fired = step_simulation(state, action, RobotPolicy.STATION_SHORTEST_PATH)
        cooled = step_simulation(
            fired,
            SimJointAction(),
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(fired.weapons[0].cooldown, 3)
        self.assertEqual(cooled.weapons[0].cooldown, 2)

    def test_nonfired_cooldown_and_remaining_turns_decrement_once(self) -> None:
        station = self._station(Position(8, 8))
        weapon = self._weapon(10, "rocket", Position(1, 1), cooldown=2)
        state = self._state(station, weapons=(weapon,), remaining=7)

        result = step_simulation(
            state,
            SimJointAction(),
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(result.weapons[0].cooldown, 1)
        self.assertEqual(result.remaining_night_turns, 6)
        self.assertEqual(result.round_no, state.round_no + 1)

    def test_end_of_night_clears_robots_without_kill_score(self) -> None:
        station = self._station(Position(8, 8))
        robot = self._robot(9, Position(3, 3))
        state = self._state(station, robots=(robot,), remaining=1, score=4)

        result = step_simulation(
            state,
            SimJointAction(),
            RobotPolicy.STATION_SHORTEST_PATH,
        )

        self.assertEqual(result.remaining_night_turns, 0)
        self.assertEqual(result.robots, ())
        self.assertEqual(result.owned_kill_score, 4)

    def test_future_policy_moves_then_controls_ready_assigned_weapon(self) -> None:
        station = self._station(Position(8, 8))
        weapon = self._weapon(10, "gatling", Position(4, 4))
        remote = self._role(1, Position(1, 1), weapon_id=10, stand=Position(3, 4))
        remote_state = self._state(station, roles=(remote,), weapons=(weapon,))

        moving = choose_future_action(remote_state)

        self.assertEqual(len(moving.role_moves), 1)
        self.assertEqual(moving.role_moves[0].role_id, 1)

        adjacent = self._role(1, Position(3, 4), weapon_id=10, stand=Position(3, 4))
        robot = self._robot(9, Position(6, 4))
        ready_state = self._state(
            station,
            roles=(adjacent,),
            weapons=(weapon,),
            robots=(robot,),
        )

        firing = choose_future_action(ready_state)

        self.assertEqual(len(firing.weapon_attacks), 1)
        self.assertEqual(firing.weapon_attacks[0].controller_id, 1)

    @staticmethod
    def _role(
        unit_id: int,
        position: Position,
        *,
        health: int = 100,
        weapon_id: int | None = None,
        stand: Position | None = None,
    ) -> SimRole:
        return SimRole(unit_id, "worker", position, health, weapon_id, stand)

    @staticmethod
    def _station(position: Position) -> SimStructure:
        cells = frozenset(
            {
                position,
                Position(position.x + 1, position.y),
                Position(position.x, position.y + 1),
                Position(position.x + 1, position.y + 1),
            }
        )
        return SimStructure(100, "station", position, cells, 500, 1)

    @staticmethod
    def _weapon(
        unit_id: int,
        role_type: str,
        position: Position,
        *,
        health: int = 100,
        attack_power: int = 10,
        attack_range: int = 8,
        level: int = 1,
        cooldown: int = 0,
    ) -> SimWeapon:
        return SimWeapon(
            unit_id,
            role_type,
            position,
            health,
            attack_power,
            attack_range,
            level,
            cooldown,
        )

    @staticmethod
    def _robot(
        robot_id: int,
        position: Position,
        *,
        health: int = 40,
        waits: bool = False,
    ) -> SimRobot:
        return SimRobot(
            robot_id,
            "smallRobot",
            position,
            health,
            5,
            3,
            1,
            waits,
        )

    @staticmethod
    def _state(
        station: SimStructure,
        *,
        roles: tuple[SimRole, ...] = (),
        walls: tuple[SimStructure, ...] = (),
        weapons: tuple[SimWeapon, ...] = (),
        robots: tuple[SimRobot, ...] = (),
        remaining: int = 60,
        score: int = 0,
    ) -> SimState:
        blocked = set(station.occupied_cells)
        blocked.update(cell for wall in walls for cell in wall.occupied_cells)
        blocked.update(weapon.position for weapon in weapons)
        return SimState(
            71,
            14,
            12,
            remaining,
            "challenger",
            frozenset(blocked),
            roles,
            station,
            walls,
            weapons,
            robots,
            score,
        )


if __name__ == "__main__":
    unittest.main()
