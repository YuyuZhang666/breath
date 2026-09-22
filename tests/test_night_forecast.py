import unittest
from dataclasses import replace
from fractions import Fraction

from future_war_agent.strategy.forecast import (
    ForecastUpdateKind,
    RiskLevel,
    build_lightweight_forecast,
    build_night_forecast,
    classify_risk,
    rebase_day_forecast,
    refresh_night_forecast,
)
from future_war_agent.strategy.simulation.config import Phase3Config
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.strategy.simulation.errors import UnsupportedSimulation
from tests.strategy_helpers import observation, robot, unit


class NightForecastTests(unittest.TestCase):
    def test_lightweight_forecast_rejects_missing_station(self) -> None:
        with self.assertRaises(UnsupportedSimulation):
            build_lightweight_forecast(observation(round_no=71))

    def test_full_forecast_honors_expired_deadline(self) -> None:
        with self.assertRaises(DeadlineExceeded):
            build_night_forecast(
                self._quiet_night(71),
                clock=lambda: 1.0,
                deadline=1.0,
            )

    def test_full_forecast_can_be_interrupted_during_work(self) -> None:
        calls = 0

        def clock() -> float:
            nonlocal calls
            calls += 1
            return 0.0 if calls < 8 else 2.0

        observed = observation(
            round_no=71,
            width=32,
            height=32,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 20, 20, 'station', health=500, level=1),
            ),
            robots=(robot(9, 0, 0, role_type='bossRobot'),),
        )

        with self.assertRaises(DeadlineExceeded):
            build_night_forecast(observed, clock=clock, deadline=1.0)
        self.assertGreaterEqual(calls, 8)

    def test_lightweight_forecast_reports_conservative_critical_risk(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 10, 10, 'station', health=100, level=1),
            ),
            robots=(robot(9, 0, 0, role_type='bossRobot'),),
        )

        forecast = build_lightweight_forecast(observed)

        self.assertIs(forecast.update_kind, ForecastUpdateKind.LIGHTWEIGHT)
        self.assertFalse(forecast.complete)
        self.assertLess(forecast.survival_margin, 0)
        self.assertIs(forecast.risk_level, RiskLevel.CRITICAL)
        self.assertIn(
            'lightweight_conservative_estimate',
            forecast.uncertainty_reasons,
        )

    def test_full_disabled_uses_lightweight_without_cache(self) -> None:
        refreshed = refresh_night_forecast(
            self._quiet_night(71),
            allow_full=False,
        )

        self.assertTrue(refreshed.recomputed)
        self.assertIs(
            refreshed.forecast.update_kind,
            ForecastUpdateKind.LIGHTWEIGHT,
        )
        self.assertIn('full forecast disabled', refreshed.reason)

    def test_full_disabled_corrects_invalid_cache_with_lightweight(self) -> None:
        first = self._quiet_night(
            71,
            extra_units=(unit(20, 6, 6, 'wall', health=40, level=1),),
        )
        second = self._quiet_night(72)
        initial = refresh_night_forecast(first)

        refreshed = refresh_night_forecast(
            second,
            previous_observation=first,
            previous_forecast=initial.forecast,
            allow_full=False,
        )

        self.assertTrue(refreshed.recomputed)
        self.assertIs(
            refreshed.forecast.update_kind,
            ForecastUpdateKind.LIGHTWEIGHT,
        )
        self.assertIn(
            'incremental_cache_lightweight_correction',
            refreshed.forecast.uncertainty_reasons,
        )

    def test_expired_lightweight_correction_keeps_incremental_cache(self) -> None:
        first = self._quiet_night(
            71,
            extra_units=(unit(20, 6, 6, 'wall', health=40, level=1),),
        )
        second = self._quiet_night(72)
        initial = refresh_night_forecast(first)

        refreshed = refresh_night_forecast(
            second,
            previous_observation=first,
            previous_forecast=initial.forecast,
            allow_full=False,
            clock=lambda: 1.0,
            deadline=1.0,
        )

        self.assertFalse(refreshed.recomputed)
        self.assertIs(
            refreshed.forecast.update_kind,
            ForecastUpdateKind.INCREMENTAL,
        )
        self.assertIn('deadline expired', refreshed.reason)

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

    def test_active_task_transition_forces_full_recompute(self) -> None:
        first = self._quiet_night(71)
        second = replace(
            self._quiet_night(72),
            phase_task='Return the exact answer.',
        )
        config = Phase3Config(tail_visible_roster_complete=True)
        initial = refresh_night_forecast(first, config=config)

        refreshed = refresh_night_forecast(
            second,
            previous_observation=first,
            previous_forecast=initial.forecast,
            config=config,
        )

        self.assertTrue(refreshed.recomputed)
        self.assertEqual(
            refreshed.reason,
            'active task changed controller availability',
        )
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
