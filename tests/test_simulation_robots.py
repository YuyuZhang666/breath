import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.robots import (
    ALL_ROBOT_POLICIES,
    RobotPolicy,
    choose_robot_intents,
)
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimRole,
    SimState,
    SimStructure,
    SimWeapon,
)


class SimulationRobotPolicyTests(unittest.TestCase):
    def test_each_policy_attacks_each_supported_blocker_within_range(self) -> None:
        attack_first_policies = (
            RobotPolicy.STATION_SHORTEST_PATH,
            RobotPolicy.MAIN_PATH_BLOCKER,
            RobotPolicy.LOW_HEALTH_BLOCKER,
        )
        for policy in attack_first_policies:
            for target_kind in ("role", "wall", "weapon", "station"):
                with self.subTest(policy=policy, target_kind=target_kind):
                    state, target_id = self._blocked_state(target_kind, distance=2)

                    intent = choose_robot_intents(state, policy)[0]

                    self.assertEqual(intent.attack_target_kind, target_kind)
                    self.assertEqual(intent.attack_target_id, target_id)
                    self.assertIsNone(intent.move_target)

    def test_blocker_at_range_four_is_not_attacked(self) -> None:
        for policy in ALL_ROBOT_POLICIES:
            state, _ = self._blocked_state("wall", distance=4)

            intent = choose_robot_intents(state, policy)[0]

            with self.subTest(policy=policy):
                self.assertIsNone(intent.attack_target_kind)
                self.assertIsNone(intent.attack_target_id)

    def test_off_route_object_within_range_is_not_attacked(self) -> None:
        state, _ = self._blocked_state("role", distance=2)
        off_route = SimRole(1, "worker", Position(1, 5), 20, None, None)
        state = self._replace_state(state, roles=(off_route,))

        for policy in ALL_ROBOT_POLICIES:
            with self.subTest(policy=policy):
                intent = choose_robot_intents(state, policy)[0]
                self.assertIsNone(intent.attack_target_kind)
                self.assertIsNone(intent.attack_target_id)

    def test_dizzy_robot_waits(self) -> None:
        state, _ = self._blocked_state("wall", distance=2, dizzy=True)

        for policy in ALL_ROBOT_POLICIES:
            with self.subTest(policy=policy):
                intent = choose_robot_intents(state, policy)[0]
                self.assertIsNone(intent.move_target)
                self.assertIsNone(intent.attack_target_kind)

    def test_low_health_policy_prefers_lowest_health_blocker(self) -> None:
        state, _ = self._blocked_state("role", distance=2)
        high = state.roles[0]
        low = SimRole(2, "worker", Position(3, 3), 5, None, None)
        state = self._replace_state(
            state,
            roles=(high, low),
        )

        intent = choose_robot_intents(state, RobotPolicy.LOW_HEALTH_BLOCKER)[0]

        self.assertEqual(intent.attack_target_kind, "role")
        self.assertEqual(intent.attack_target_id, 2)

    def test_maximum_progress_compares_move_against_blocker_attack(self) -> None:
        state, target_id = self._blocked_state("role", distance=2)

        blocker_intent = choose_robot_intents(
            state,
            RobotPolicy.MAIN_PATH_BLOCKER,
        )[0]
        progress_intent = choose_robot_intents(
            state,
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
        )[0]

        self.assertEqual(blocker_intent.attack_target_id, target_id)
        self.assertIsNone(progress_intent.attack_target_id)
        self.assertIsNotNone(progress_intent.move_target)

    def test_maximum_progress_can_select_station_attack(self) -> None:
        state, target_id = self._blocked_state("station", distance=2)

        intent = choose_robot_intents(
            state,
            RobotPolicy.MAXIMUM_STATION_PROGRESS,
        )[0]

        self.assertEqual(intent.attack_target_kind, "station")
        self.assertEqual(intent.attack_target_id, target_id)

    def test_stable_ties_choose_the_same_target_repeatedly(self) -> None:
        state, _ = self._blocked_state("role", distance=2)
        first_role = state.roles[0]
        second_role = SimRole(2, "worker", Position(3, 1), 20, None, None)
        state = self._replace_state(state, roles=(first_role, second_role))

        first = choose_robot_intents(state, RobotPolicy.MAIN_PATH_BLOCKER)
        second = choose_robot_intents(state, RobotPolicy.MAIN_PATH_BLOCKER)

        self.assertEqual(first, second)
        self.assertEqual(first[0].attack_target_id, 1)

    def test_policy_does_not_mutate_input_state(self) -> None:
        state, _ = self._blocked_state("weapon", distance=2)
        original = state

        for policy in ALL_ROBOT_POLICIES:
            choose_robot_intents(state, policy)

        self.assertEqual(state, original)

    def test_policy_set_is_exact_and_stable(self) -> None:
        self.assertEqual(
            ALL_ROBOT_POLICIES,
            (
                RobotPolicy.STATION_SHORTEST_PATH,
                RobotPolicy.MAIN_PATH_BLOCKER,
                RobotPolicy.LOW_HEALTH_BLOCKER,
                RobotPolicy.MAXIMUM_STATION_PROGRESS,
            ),
        )

    @classmethod
    def _blocked_state(
        cls,
        target_kind: str,
        *,
        distance: int,
        dizzy: bool = False,
    ):
        robot = SimRobot(
            9,
            "smallRobot",
            Position(1, 2),
            40,
            5,
            3,
            1,
            dizzy,
        )
        blocker_position = Position(1 + distance, 2)
        far_station = cls._station(90, Position(9, 2))
        roles: tuple[SimRole, ...] = ()
        walls: tuple[SimStructure, ...] = ()
        weapons: tuple[SimWeapon, ...] = ()
        station = far_station
        target_id = 1

        if target_kind == "role":
            roles = (SimRole(target_id, "worker", blocker_position, 20, None, None),)
        elif target_kind == "wall":
            walls = (
                SimStructure(
                    target_id,
                    "wall",
                    blocker_position,
                    frozenset({blocker_position}),
                    20,
                    1,
                ),
            )
        elif target_kind == "weapon":
            weapons = (
                SimWeapon(
                    target_id,
                    "gatling",
                    blocker_position,
                    20,
                    10,
                    6,
                    1,
                    0,
                ),
            )
        elif target_kind == "station":
            station = cls._station(target_id, blocker_position)
        else:
            raise AssertionError(target_kind)

        building_cells = station.occupied_cells | frozenset(
            cell for wall in walls for cell in wall.occupied_cells
        ) | frozenset(weapon.position for weapon in weapons)
        return (
            SimState(
                71,
                14,
                10,
                60,
                "challenger",
                building_cells,
                roles,
                station,
                walls,
                weapons,
                (robot,),
            ),
            target_id,
        )

    @staticmethod
    def _station(unit_id: int, position: Position) -> SimStructure:
        cells = frozenset(
            {
                position,
                Position(position.x + 1, position.y),
                Position(position.x, position.y + 1),
                Position(position.x + 1, position.y + 1),
            }
        )
        return SimStructure(unit_id, "station", position, cells, 500, 1)

    @staticmethod
    def _replace_state(
        state: SimState,
        *,
        roles: tuple[SimRole, ...],
    ) -> SimState:
        return SimState(
            state.round_no,
            state.width,
            state.height,
            state.remaining_night_turns,
            state.team_type,
            state.static_blocked,
            roles,
            state.station,
            state.walls,
            state.weapons,
            state.robots,
            state.owned_kill_score,
        )


if __name__ == "__main__":
    unittest.main()
