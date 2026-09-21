import unittest
from dataclasses import replace
from fractions import Fraction

from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.forecast import (
    ForecastUpdateKind,
    NightForecast,
    RiskLevel,
)
from future_war_agent.strategy.policy import (
    DEFAULT_STRATEGIC_INTENT,
    StrategyProfile,
    intent_for_profile,
)
from future_war_agent.strategy.simulation.config import Phase3Level
from future_war_agent.strategy.simulation.trigger import select_phase3_level
from tests.strategy_helpers import observation, robot, unit


class Phase3TriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = observation(
            round_no=71,
            our_units=(
                unit(1, 2, 2, 'worker', health=100),
                unit(2, 7, 7, 'station', health=1000),
                unit(3, 5, 5, 'wall', health=500),
                unit(4, 3, 3, 'rocket', health=500, cooldown=1),
            ),
        )

    def test_stable_night_skips_rollout(self) -> None:
        current = replace(
            self.previous,
            time=TurnTime.from_round(72),
        )

        trigger = select_phase3_level(
            current,
            self.previous,
            DEFAULT_STRATEGIC_INTENT,
        )

        self.assertIs(trigger.level, Phase3Level.NONE)

    def test_wall_pressure_and_ready_rocket_trigger_lite(self) -> None:
        pressured = replace(
            self.previous,
            time=TurnTime.from_round(72),
            robots=(robot(10, 5, 6, target_team='challenger'),),
        )
        ready_units = tuple(
            replace(item, cooldown=0) if item.unit_id == 4 else item
            for item in self.previous.our.units
        )
        ready = replace(
            self.previous,
            time=TurnTime.from_round(72),
            our=replace(self.previous.our, units=ready_units),
        )

        self.assertIs(
            select_phase3_level(
                pressured,
                self.previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.LITE,
        )

        stable_pressure = replace(
            pressured,
            time=TurnTime.from_round(73),
        )
        self.assertIs(
            select_phase3_level(
                stable_pressure,
                pressured,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.NONE,
        )
        self.assertIs(
            select_phase3_level(
                ready,
                self.previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.LITE,
        )

    def test_station_damage_and_asset_loss_trigger_full(self) -> None:
        damaged_units = tuple(
            replace(item, health=900) if item.unit_id == 2 else item
            for item in self.previous.our.units
        )
        damaged = replace(
            self.previous,
            time=TurnTime.from_round(72),
            our=replace(self.previous.our, units=damaged_units),
        )
        breached = replace(
            self.previous,
            time=TurnTime.from_round(72),
            our=replace(
                self.previous.our,
                units=tuple(
                    item for item in self.previous.our.units if item.unit_id != 3
                ),
            ),
        )

        self.assertIs(
            select_phase3_level(
                damaged,
                self.previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.FULL,
        )
        self.assertIs(
            select_phase3_level(
                breached,
                self.previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.FULL,
        )

    def test_other_team_robots_do_not_trigger_and_desperation_does(self) -> None:
        unrelated = replace(
            self.previous,
            time=TurnTime.from_round(72),
            robots=(robot(10, 5, 6, target_team='defender'),),
        )

        self.assertIs(
            select_phase3_level(
                unrelated,
                self.previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.NONE,
        )
        self.assertIs(
            select_phase3_level(
                unrelated,
                self.previous,
                intent_for_profile(StrategyProfile.DESPERATION, 'test'),
            ).level,
            Phase3Level.FULL,
        )

    def test_forecast_risk_drives_none_lite_and_full(self) -> None:
        current = replace(self.previous, time=TurnTime.from_round(72))
        expected = (
            (RiskLevel.SAFE, Phase3Level.NONE),
            (RiskLevel.WATCH, Phase3Level.LITE),
            (RiskLevel.CRITICAL, Phase3Level.FULL),
            (RiskLevel.LETHAL, Phase3Level.FULL),
        )
        for risk, level in expected:
            with self.subTest(risk=risk):
                self.assertIs(
                    select_phase3_level(
                        current,
                        self.previous,
                        DEFAULT_STRATEGIC_INTENT,
                        forecast=self._forecast(risk),
                    ).level,
                    level,
                )

    def test_unknown_robot_uses_conservative_visible_threat(self) -> None:
        low_station_units = tuple(
            replace(item, health=30) if item.unit_id == 2 else item
            for item in self.previous.our.units
        )
        previous = replace(
            self.previous,
            our=replace(self.previous.our, units=low_station_units),
        )
        current = replace(
            previous,
            time=TurnTime.from_round(72),
            robots=(
                robot(
                    10,
                    7,
                    7,
                    role_type='unknownRobot',
                    target_team='challenger',
                ),
            ),
        )

        self.assertIs(
            select_phase3_level(
                current,
                previous,
                DEFAULT_STRATEGIC_INTENT,
            ).level,
            Phase3Level.FULL,
        )

    @staticmethod
    def _forecast(risk: RiskLevel) -> NightForecast:
        ratio = {
            RiskLevel.SAFE: Fraction(1, 10),
            RiskLevel.WATCH: Fraction(3, 5),
            RiskLevel.CRITICAL: Fraction(9, 10),
            RiskLevel.LETHAL: Fraction(2),
        }[risk]
        return NightForecast(
            expected_station_hp_at_dawn=500,
            predicted_damage_before_dawn=int(ratio * 100),
            effective_defense_hp=100,
            future_firepower=0,
            survival_margin=100 - int(ratio * 100),
            risk_ratio=ratio,
            risk_level=risk,
            expected_wall_losses=0,
            expected_weapon_losses=0,
            expected_role_losses=0,
            lethal_round=72 if risk is RiskLevel.LETHAL else None,
            critical_robot_ids=(),
            critical_wall_ids=(),
            generated_round=71,
            updated_round=71,
            day_no=1,
            model_version='night-forecast-v1',
            update_kind=ForecastUpdateKind.FULL,
            complete=True,
            uncertainty_reasons=(),
            observed_station_hp=1000,
            expected_next_station_hp=1000,
        )


if __name__ == '__main__':
    unittest.main()
