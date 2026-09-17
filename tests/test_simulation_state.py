import unittest
from dataclasses import FrozenInstanceError, replace

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.simulation.errors import UnsupportedSimulation
from future_war_agent.strategy.simulation.state import AssignedStand, build_sim_state
from tests.strategy_helpers import observation, robot, unit


EXPLICIT_WEAPON_FIELDS = frozenset(
    {"attackPower", "attackRange", "level", "cooldown"}
)


class SimulationStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = unit(10, 1, 1, "worker", health=220)
        self.station = unit(100, 2, 2, "station", health=900, level=1)
        self.wall = unit(101, 6, 6, "wall", health=300, level=1)
        self.gatling = unit(
            10020,
            5,
            5,
            "gatling",
            health=180,
            attack_power=10,
            attack_range=6,
            level=1,
            cooldown=0,
            provided_fields=EXPLICIT_WEAPON_FIELDS,
        )
        self.rocket = unit(
            10021,
            7,
            5,
            "rocket",
            health=170,
            attack_power=20,
            attack_range=7,
            level=2,
            cooldown=1,
            provided_fields=EXPLICIT_WEAPON_FIELDS,
        )
        self.enemy_worker = unit(200, 11, 11, "worker")
        self.enemy_station = unit(201, 12, 2, "station", health=800)
        self.middle_robot = robot(
            9,
            8,
            8,
            role_type="middleRobot",
            health=60,
            abnormal_state="DiZzY",
        )
        self.assignments = (
            AssignedStand(10, 10020, Position(4, 5)),
        )
        self.observed = observation(
            round_no=80,
            width=20,
            height=15,
            our_units=(
                self.worker,
                self.station,
                self.wall,
                self.gatling,
                self.rocket,
            ),
            enemy_units=(self.enemy_worker, self.enemy_station),
            robots=(self.middle_robot,),
            zones=(Zone(Position(0, 0), "mountain"),),
        )

    def test_build_state_keeps_role_health_and_assignments(self) -> None:
        state = build_sim_state(self.observed, self.assignments)

        self.assertEqual(state.roles[0].health, 220)
        self.assertEqual(state.roles[0].assigned_weapon_id, 10020)
        self.assertEqual(state.roles[0].assigned_stand, Position(4, 5))
        with self.assertRaises(FrozenInstanceError):
            state.roles[0].health = 1

    def test_robot_spec_supplies_range_three_kill_score_and_dizzy_wait(self) -> None:
        state = build_sim_state(self.observed, self.assignments)

        self.assertEqual(state.robots[0].attack_power, 10)
        self.assertEqual(state.robots[0].attack_range, 3)
        self.assertEqual(state.robots[0].kill_score, 2)
        self.assertTrue(state.robots[0].waits_this_turn)

    def test_remaining_night_turns_is_inclusive(self) -> None:
        state = build_sim_state(self.observed, self.assignments)

        self.assertEqual(state.round_no, 80)
        self.assertEqual(state.remaining_night_turns, 51)

    def test_unknown_robot_type_is_unsupported(self) -> None:
        observed = replace(
            self.observed,
            robots=(robot(9, 8, 8, role_type="mysteryRobot"),),
        )

        with self.assertRaises(UnsupportedSimulation):
            build_sim_state(observed, self.assignments)

    def test_day_request_and_blank_team_type_are_unsupported(self) -> None:
        day = replace(self.observed, time=TurnTime.from_round(1))
        blank_team = replace(
            self.observed,
            our=replace(self.observed.our, team_type="  "),
        )

        for observed in (day, blank_team):
            with self.subTest(observed=observed):
                with self.assertRaises(UnsupportedSimulation):
                    build_sim_state(observed, self.assignments)

    def test_every_alive_robot_requires_a_nonblank_target_team(self) -> None:
        for target_team in (None, "", "  "):
            observed = replace(
                self.observed,
                robots=(replace(self.middle_robot, target_team=target_team),),
            )
            with self.subTest(target_team=target_team):
                with self.assertRaises(UnsupportedSimulation):
                    build_sim_state(observed, self.assignments)

    def test_invalid_dead_unit_coordinate_is_rejected_before_filtering(self) -> None:
        invalid_dead_role = unit(99, -1, 0, "worker", health=0)
        observed = replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=self.observed.our.units + (invalid_dead_role,),
            ),
        )

        with self.assertRaises(UnsupportedSimulation):
            build_sim_state(observed, self.assignments)

    def test_exactly_one_living_station_is_required(self) -> None:
        without_station = replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=tuple(
                    value
                    for value in self.observed.our.units
                    if value.role_type != "station"
                ),
            ),
        )
        extra_station = unit(102, 14, 10, "station", health=700)
        with_two = replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=self.observed.our.units + (extra_station,),
            ),
        )

        for observed in (without_station, with_two):
            with self.subTest(observed=observed):
                with self.assertRaises(UnsupportedSimulation):
                    build_sim_state(observed, self.assignments)

    def test_weapon_fields_must_be_explicit(self) -> None:
        for field_name in EXPLICIT_WEAPON_FIELDS:
            weapon = replace(
                self.gatling,
                provided_fields=EXPLICIT_WEAPON_FIELDS - {field_name},
            )
            observed = self._replace_our_unit(self.gatling.unit_id, weapon)
            with self.subTest(field_name=field_name):
                with self.assertRaises(UnsupportedSimulation):
                    build_sim_state(observed, self.assignments)

    def test_weapon_values_must_match_supported_semantics(self) -> None:
        railgun = unit(
            10022,
            9,
            5,
            "railgun",
            attack_power=0,
            attack_range=8,
            level=1,
            cooldown=0,
            provided_fields=EXPLICIT_WEAPON_FIELDS,
        )
        invalid_weapons = (
            replace(self.gatling, attack_power=11),
            replace(self.rocket, attack_power=21),
            replace(self.gatling, attack_range=0),
            replace(self.gatling, level=0),
            replace(self.gatling, cooldown=-1),
            railgun,
        )

        for weapon in invalid_weapons:
            if weapon.unit_id in {self.gatling.unit_id, self.rocket.unit_id}:
                observed = self._replace_our_unit(weapon.unit_id, weapon)
            else:
                observed = replace(
                    self.observed,
                    our=replace(
                        self.observed.our,
                        units=self.observed.our.units + (weapon,),
                    ),
                )
            with self.subTest(weapon=weapon):
                with self.assertRaises(UnsupportedSimulation):
                    build_sim_state(observed, self.assignments)

    def test_unknown_living_weapon_type_is_unsupported(self) -> None:
        unknown = replace(self.gatling, role_type="laser")
        observed = self._replace_our_unit(self.gatling.unit_id, unknown)

        with self.assertRaises(UnsupportedSimulation):
            build_sim_state(observed, self.assignments)

    def test_robots_targeting_another_team_are_ignored(self) -> None:
        observed = replace(
            self.observed,
            robots=(
                replace(
                    self.middle_robot,
                    role_type="mysteryRobot",
                    target_team="opponent",
                ),
            ),
        )

        state = build_sim_state(observed, self.assignments)

        self.assertEqual(state.robots, ())

    def test_static_blocked_contains_terrain_and_buildings_only(self) -> None:
        state = build_sim_state(self.observed, self.assignments)

        expected_station_cells = frozenset(
            {Position(2, 2), Position(3, 2), Position(2, 3), Position(3, 3)}
        )
        expected_enemy_station_cells = frozenset(
            {
                Position(12, 2),
                Position(13, 2),
                Position(12, 3),
                Position(13, 3),
            }
        )
        self.assertEqual(state.station.occupied_cells, expected_station_cells)
        self.assertTrue(expected_station_cells <= state.static_blocked)
        self.assertTrue(expected_enemy_station_cells <= state.static_blocked)
        self.assertIn(Position(0, 0), state.static_blocked)
        self.assertIn(self.wall.position, state.static_blocked)
        self.assertIn(self.gatling.position, state.static_blocked)
        self.assertNotIn(self.worker.position, state.static_blocked)
        self.assertNotIn(self.enemy_worker.position, state.static_blocked)
        self.assertNotIn(self.middle_robot.position, state.static_blocked)

    def _replace_our_unit(self, unit_id: int, replacement):
        return replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=tuple(
                    replacement if value.unit_id == unit_id else value
                    for value in self.observed.our.units
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
