import unittest
from dataclasses import replace
from fractions import Fraction

from future_war_agent.strategy.forecast import (
    ForecastUpdateKind,
    RiskLevel,
    build_night_forecast,
    classify_risk,
    rebase_day_forecast,
    refresh_night_forecast,
)
from future_war_agent.strategy.simulation.config import Phase3Config
from tests.strategy_helpers import observation, robot, unit


class NightForecastTests(unittest.TestCase):
    def test_risk_threshold_boundaries_are_exact(self) -> None:
        self.assertIs(
            classify_risk(Fraction(11, 20), 1, None, 71),
            RiskLevel.WATCH,
        )
        self.assertIs(
            classify_risk(Fraction(17, 20), 1, None, 71),
            RiskLevel.CRITICAL,
        )
        self.assertIs(
            classify_risk(Fraction(1, 10), 100, 73, 71),
            RiskLevel.LETHAL,
        )
        self.assertIs(
            classify_risk(
                Fraction(1, 10),
                100,
                None,
                71,
                complete=False,
            ),
            RiskLevel.UNKNOWN,
        )

    def test_long_range_d3_wave_is_visible_beyond_four_round_prefix(self) -> None:
        observed = observation(
            round_no=331,
            width=16,
            height=16,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 14, 14, 'station', health=500, level=1),
            ),
            robots=(
                robot(
                    9,
                    0,
                    0,
                    role_type='bossRobot',
                    health=800,
                ),
            ),
        )
        config = Phase3Config(
            tail_visible_roster_complete=True,
            tail_late_wave_calibration_source='synthetic-test',
        )

        forecast = build_night_forecast(observed, config=config)

        self.assertEqual(observed.our.units[1].health, 500)
        self.assertGreater(forecast.predicted_damage_before_dawn, 500)
        self.assertLess(forecast.survival_margin, 0)
        self.assertIn(
            forecast.risk_level,
            {RiskLevel.CRITICAL, RiskLevel.LETHAL},
        )

    def test_ordinary_round_uses_incremental_update(self) -> None:
        first = self._quiet_night(71)
        second = self._quiet_night(72)
        config = Phase3Config(tail_visible_roster_complete=True)
        initial = refresh_night_forecast(first, config=config)

        refreshed = refresh_night_forecast(
            second,
            previous_observation=first,
            previous_forecast=initial.forecast,
            config=config,
        )

        self.assertTrue(initial.recomputed)
        self.assertFalse(refreshed.recomputed)
        self.assertIs(
            refreshed.forecast.update_kind,
            ForecastUpdateKind.INCREMENTAL,
        )
        self.assertEqual(
            refreshed.forecast.generated_round,
            initial.forecast.generated_round,
        )
        self.assertEqual(refreshed.forecast.updated_round, 72)

    def test_wall_loss_forces_full_recompute(self) -> None:
        first = self._quiet_night(
            71,
            extra_units=(unit(20, 6, 6, 'wall', health=40, level=1),),
        )
        second = self._quiet_night(72)
        config = Phase3Config(tail_visible_roster_complete=True)
        initial = refresh_night_forecast(first, config=config)

        refreshed = refresh_night_forecast(
            second,
            previous_observation=first,
            previous_forecast=initial.forecast,
            config=config,
        )

        self.assertTrue(refreshed.recomputed)
        self.assertEqual(refreshed.reason, 'wall destroyed')
        self.assertEqual(refreshed.forecast.generated_round, 72)

    def test_day_rebase_clears_stale_immediate_lethal_prediction(self) -> None:
        config = Phase3Config(tail_visible_roster_complete=True)
        forecast = build_night_forecast(
            self._quiet_night(71),
            config=config,
        )
        lethal = replace(
            forecast,
            predicted_damage_before_dawn=600,
            effective_defense_hp=500,
            survival_margin=-100,
            risk_ratio=Fraction(6, 5),
            risk_level=RiskLevel.LETHAL,
            lethal_round=72,
        )
        day = observation(
            round_no=131,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 10, 10, 'station', health=500, level=1),
            ),
        )

        rebased = rebase_day_forecast(day, lethal)

        self.assertIs(rebased.risk_level, RiskLevel.CRITICAL)
        self.assertIsNone(rebased.lethal_round)
        self.assertIs(rebased.update_kind, ForecastUpdateKind.REBASED_DAY)

    @staticmethod
    def _quiet_night(round_no: int, *, extra_units=()):
        return observation(
            round_no=round_no,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 10, 10, 'station', health=500, level=1),
                *extra_units,
            ),
        )


if __name__ == '__main__':
    unittest.main()
