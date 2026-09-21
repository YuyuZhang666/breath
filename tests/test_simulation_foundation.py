import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

from future_war_agent.strategy.simulation.config import (
    DEFAULT_PHASE3_CONFIG,
    ROBOT_SPECS,
    Phase3Config,
    Phase3Level,
    RobotSpec,
)
from future_war_agent.strategy.simulation.errors import (
    DeadlineExceeded,
    UnsupportedSimulation,
)
from future_war_agent.strategy.simulation.objective import (
    DEFAULT_NIGHT_OBJECTIVE,
    NightObjective,
)
from tests.strategy_helpers import robot, unit


class SimulationFoundationTests(unittest.TestCase):
    def test_helpers_freeze_phase_3_values(self) -> None:
        observed_unit = unit(
            1,
            2,
            3,
            "gatling",
            attack_power=10,
            backpack=["stone"],
            provided_fields=["attackPower", "cooldown"],
        )
        observed_robot = robot(9, 4, 5, abnormal_state="dizzy")

        self.assertEqual(observed_unit.attack_power, 10)
        self.assertEqual(observed_unit.backpack, ("stone",))
        self.assertEqual(
            observed_unit.provided_fields,
            frozenset({"attackPower", "cooldown"}),
        )
        self.assertEqual(observed_robot.abnormal_state, "dizzy")

    def test_robot_specs_define_the_supported_protocol_types(self) -> None:
        self.assertEqual(
            dict(ROBOT_SPECS),
            {
                "smallRobot": RobotSpec(5, 3, 40, 1),
                "middleRobot": RobotSpec(10, 3, 60, 2),
                "largeRobot": RobotSpec(20, 3, 500, 4),
                "bossRobot": RobotSpec(40, 3, 800, 10),
            },
        )
        with self.assertRaises(TypeError):
            ROBOT_SPECS["smallRobot"] = RobotSpec(99, 3, 40, 1)

    def test_phase_3_defaults_expose_lite_and_full_budgets(self) -> None:
        config = DEFAULT_PHASE3_CONFIG

        self.assertEqual(config.max_root_actions, 8)
        self.assertEqual(config.scenario_count, 2)
        self.assertEqual(config.max_horizon, 4)
        self.assertEqual(config.watchdog_seconds, 0.250)
        self.assertEqual(config.budget_for(Phase3Level.LITE).root_candidates, 4)
        self.assertEqual(config.budget_for(Phase3Level.LITE).scenarios, 1)
        self.assertEqual(config.budget_for(Phase3Level.LITE).exact_horizon, 2)
        self.assertEqual(config.budget_for(Phase3Level.LITE).seconds, 0.080)
        self.assertEqual(config.scenario_weight_floor, Fraction(1, 20))
        self.assertEqual(config.gatling_damage, 10)
        self.assertEqual(config.rocket_center_damage, 20)
        self.assertEqual(config.rocket_splash_damage, 10)
        self.assertEqual(config.rocket_cooldown_rounds, 3)
        self.assertEqual(config.max_weapon_candidates, 3)

    def test_phase_3_rejects_invalid_search_budget_values(self) -> None:
        invalid_configs = (
            {"max_root_actions": 9},
            {"scenario_count": 3},
            {"max_horizon": 0},
            {"max_horizon": 5},
            {"watchdog_seconds": 0},
            {"lite_root_actions": 5},
            {"lite_scenario_count": 2},
            {"lite_horizon": 3},
            {"lite_watchdog_seconds": 0},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    Phase3Config(**values)

    def test_night_objective_is_immutable_and_conservative_by_default(self) -> None:
        objective = DEFAULT_NIGHT_OBJECTIVE

        self.assertEqual(objective.minimum_station_health, 1)
        self.assertTrue(objective.protect_all_controllers)
        self.assertTrue(objective.protect_all_weapons)
        self.assertTrue(objective.require_post_horizon_buffer)
        self.assertTrue(objective.enable_score_band)
        with self.assertRaises(FrozenInstanceError):
            objective.minimum_station_health = 2

    def test_night_objective_requires_positive_station_health(self) -> None:
        with self.assertRaises(ValueError):
            NightObjective(minimum_station_health=0)

    def test_simulation_errors_are_runtime_failures(self) -> None:
        self.assertTrue(issubclass(UnsupportedSimulation, RuntimeError))
        self.assertTrue(issubclass(DeadlineExceeded, RuntimeError))


if __name__ == "__main__":
    unittest.main()
