import unittest
from fractions import Fraction

from future_war_agent.strategy.features import extract_features
from future_war_agent.strategy.simulation.certificate import (
    ScenarioOutcome,
    WaveClassification,
    build_certificate,
)
from future_war_agent.strategy.simulation.objective import DEFAULT_NIGHT_OBJECTIVE
from tests.strategy_helpers import observation, robot, unit


def certificate(*, station_health: int, secured: bool):
    outcome = ScenarioOutcome(
        weight=Fraction(1),
        station_health=station_health,
        surviving_controlled_role_count=3,
        surviving_controller_count=3 if secured else 2,
        controller_losses=0 if secured else 1,
        surviving_key_weapon_count=3,
        key_weapon_losses=0,
        wall_losses=0,
        weapon_losses=0,
        minimum_controlled_role_health=100,
        surviving_wall_non_key_weapon_value=1000,
        owned_kill_score=2,
        remaining_threat=0 if secured else 40,
        remaining_one_turn_damage=0 if secured else 40,
        ended_with_night=False,
    )
    return build_certificate((outcome,), DEFAULT_NIGHT_OBJECTIVE)


class StrategyFeatureTests(unittest.TestCase):
    def test_robot_threat_uses_canonical_power_and_one_turn_range(self) -> None:
        observed = observation(
            our_units=(unit(10, 5, 5, 'station', health=1000, level=1),),
            robots=(
                robot(20, 8, 8, role_type='smallRobot'),
                robot(21, 12, 12, role_type='middleRobot'),
                robot(22, 14, 14, role_type='largeRobot'),
            ),
        )

        features = extract_features(observed)

        self.assertEqual(features.targeted_robot_attack_power, 35)
        self.assertEqual(features.targeted_robot_one_turn_attack_power, 5)

    def test_extracts_station_delta_and_visible_targeted_threat(self) -> None:
        previous = observation(
            round_no=70,
            our_units=(unit(10, 5, 5, "station", health=1000, level=1),),
        )
        current = observation(
            round_no=71,
            our_units=(unit(10, 5, 5, "station", health=900, level=1),),
            robots=(
                robot(20, 7, 7, role_type="bossRobot"),
                robot(21, 10, 10, target_team="enemy"),
            ),
        )

        features = extract_features(current, previous_observation=previous)

        self.assertEqual(features.our_station_health, 900)
        self.assertEqual(features.station_health_loss, 100)
        self.assertEqual(features.targeted_robot_count, 1)
        self.assertEqual(features.targeted_robot_attack_power, 40)
        self.assertEqual(features.nearest_targeted_robot_distance, 1)

    def test_extracts_defense_completeness_and_certificate_scope(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10, 5, 5, "station", health=1000, level=1),
                unit(11, 3, 3, "gatling", level=1),
                unit(12, 4, 3, "railgun", level=1),
                unit(13, 5, 3, "rocket", level=1),
            ),
        )
        wave = certificate(station_health=900, secured=True)

        features = extract_features(observed, certificate=wave)

        self.assertTrue(features.defense_complete)
        self.assertEqual(features.wave_classification, WaveClassification.WAVE_SAFE)
        self.assertTrue(features.wave_secured)
        self.assertEqual(features.living_weapon_count, 3)

    def test_missing_required_weapon_keeps_defense_incomplete(self) -> None:
        observed = observation(
            our_units=(
                unit(10, 5, 5, "station", level=1),
                unit(11, 3, 3, "gatling", level=1),
                unit(12, 4, 3, "railgun", level=1),
            ),
        )

        features = extract_features(observed)

        self.assertFalse(features.defense_complete)


if __name__ == "__main__":
    unittest.main()
