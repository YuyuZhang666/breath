import unittest

from future_war_agent.strategy.director import DirectorConfig, StrategicDirector
from future_war_agent.strategy.policy import RuleFeatureFlags, StrategyProfile
from tests.strategy_helpers import observation, robot, unit


def defended_observation(
    *,
    round_no: int = 20,
    station_health: int = 1000,
    gold: int = 200,
    enemy_station_health: int | None = None,
    robots=(),
):
    enemy_units = ()
    if enemy_station_health is not None:
        enemy_units = (
            unit(90, 10, 10, "station", health=enemy_station_health, level=1),
        )
    return observation(
        round_no=round_no,
        our_units=(
            unit(10, 5, 5, "station", health=station_health, level=1),
            unit(11, 3, 3, "gatling", level=1),
            unit(12, 4, 3, "railgun", level=1),
            unit(13, 5, 3, "rocket", level=1),
            unit(14, 2, 2, "worker"),
        ),
        enemy_units=enemy_units,
        robots=robots,
        gold=gold,
    )


class StrategicDirectorTests(unittest.TestCase):
    def test_day_one_missing_core_weapon_can_consume_economy_reserve(self) -> None:
        observed = observation(
            round_no=20,
            our_units=(
                unit(10, 5, 5, 'station', level=1),
                unit(11, 2, 2, 'worker'),
                unit(12, 3, 3, 'gatling', level=1),
                unit(13, 4, 3, 'railgun', level=1),
            ),
            gold=25,
        )

        decision = StrategicDirector().select(observed)

        self.assertIs(decision.intent.profile, StrategyProfile.ECONOMY)
        self.assertEqual(decision.intent.gold_reserve, 25)
        self.assertEqual(
            decision.intent.reserve_eligible_actions,
            frozenset({('build', 'rocket')}),
        )

    def test_opening_reserve_exception_does_not_extend_into_day_two(self) -> None:
        observed = observation(
            round_no=131,
            our_units=(
                unit(10, 5, 5, 'station', level=1),
                unit(11, 2, 2, 'worker'),
            ),
            gold=75,
        )

        decision = StrategicDirector().select(observed)

        self.assertEqual(decision.intent.gold_reserve, 25)
        self.assertEqual(decision.intent.reserve_eligible_actions, frozenset())

    def test_selects_economy_until_defense_and_reserve_are_ready(self) -> None:
        observed = observation(
            round_no=20,
            our_units=(unit(10, 5, 5, "station", level=1),),
            gold=20,
        )

        decision = StrategicDirector(
            config=DirectorConfig(required_personal_roles=0)
        ).select(observed)

        self.assertEqual(decision.intent.profile, StrategyProfile.ECONOMY)
        self.assertEqual(decision.state.reason, "defense or reserve incomplete")

    def test_selects_score_when_safe_defense_is_funded(self) -> None:
        decision = StrategicDirector().select(defended_observation())

        self.assertEqual(decision.intent.profile, StrategyProfile.SCORE)
        self.assertTrue(decision.intent.allow_tasks)

    def test_visible_threat_selects_survive(self) -> None:
        observed = defended_observation(
            robots=(robot(20, 7, 7, role_type="mediumRobot"),)
        )

        decision = StrategicDirector().select(observed)

        self.assertEqual(decision.intent.profile, StrategyProfile.SURVIVE)

    def test_fatal_visible_threat_selects_desperation(self) -> None:
        observed = defended_observation(
            station_health=30,
            robots=(robot(20, 7, 7, role_type="bossRobot"),),
        )

        decision = StrategicDirector().select(observed)

        self.assertEqual(decision.intent.profile, StrategyProfile.DESPERATION)

    def test_fatal_power_outside_one_turn_range_only_selects_survive(self) -> None:
        observed = defended_observation(
            station_health=40,
            robots=(robot(20, 14, 14, role_type='bossRobot'),),
        )

        decision = StrategicDirector().select(observed)

        self.assertEqual(decision.intent.profile, StrategyProfile.SURVIVE)

    def test_pressure_requires_explicit_flag_and_visible_weakness(self) -> None:
        observed = defended_observation(enemy_station_health=100)
        disabled = StrategicDirector().select(observed)
        enabled = StrategicDirector(
            feature_flags=RuleFeatureFlags(enable_pressure=True)
        ).select(observed)

        self.assertEqual(disabled.intent.profile, StrategyProfile.SCORE)
        self.assertEqual(enabled.intent.profile, StrategyProfile.PRESSURE)
        self.assertTrue(enabled.intent.allow_pressure)

    def test_zero_hold_allows_immediate_non_emergency_transition(self) -> None:
        director = StrategicDirector(
            config=DirectorConfig(
                evaluation_interval=1,
                required_personal_roles=0,
            )
        )
        first = director.select(
            defended_observation(round_no=20, station_health=100)
        )
        incomplete = observation(
            round_no=21,
            our_units=(unit(10, 5, 5, "station", level=1),),
            gold=10,
        )

        held = director.select(
            incomplete,
            previous_state=first.state,
            previous_observation=defended_observation(
                round_no=20,
                station_health=100,
            ),
        )

        self.assertEqual(held.intent.profile, StrategyProfile.ECONOMY)
        self.assertEqual(held.state.since_round, 21)

    def test_station_damage_overrides_hold_period(self) -> None:
        director = StrategicDirector(config=DirectorConfig(evaluation_interval=10))
        previous_observation = defended_observation(round_no=20)
        first = director.select(previous_observation)
        damaged = defended_observation(round_no=21, station_health=900)

        override = director.select(
            damaged,
            previous_state=first.state,
            previous_observation=previous_observation,
        )

        self.assertEqual(override.intent.profile, StrategyProfile.SURVIVE)
        self.assertEqual(override.state.since_round, 21)

    def test_non_material_round_reuses_previous_evaluation(self) -> None:
        director = StrategicDirector(config=DirectorConfig(evaluation_interval=5))
        previous_observation = defended_observation(round_no=20)
        first = director.select(previous_observation)

        reused = director.select(
            defended_observation(round_no=21),
            previous_state=first.state,
            previous_observation=previous_observation,
        )

        self.assertEqual(reused.state.last_evaluated_round, 20)
        self.assertEqual(reused.intent.profile, StrategyProfile.SCORE)


if __name__ == "__main__":
    unittest.main()
