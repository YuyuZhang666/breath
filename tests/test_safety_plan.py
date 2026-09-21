import unittest
from fractions import Fraction

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.director import StrategicDirector
from future_war_agent.strategy.forecast import (
    ForecastUpdateKind,
    NightForecast,
    RiskLevel,
)
from future_war_agent.strategy.joint import TacticalCandidate, is_valid_joint
from future_war_agent.strategy.policy import BuildPlan, StrategyProfile
from future_war_agent.strategy.safety import (
    SafetyPlanStatus,
    find_cheapest_safe_plan,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class SafetyPlanTests(unittest.TestCase):
    def test_verified_missing_weapon_can_form_cheapest_safe_plan(self) -> None:
        observed = observation(
            round_no=1,
            gold=25,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 10, 10, 'station', health=500, level=1),
            ),
        )
        forecast = self._forecast(
            risk_level=RiskLevel.WATCH,
            predicted_damage=20,
            effective_hp=30,
            margin=10,
        )

        plan = find_cheapest_safe_plan(
            observed,
            forecast,
            BuildPlan(weapon_loadout=('gatling',), build_walls=False),
        )

        self.assertIsNotNone(plan)
        self.assertIs(plan.status, SafetyPlanStatus.FOUND)
        self.assertEqual(plan.gold_cost, 25)
        self.assertEqual(plan.reserve_eligible_actions, {('build', 'gatling')})

    def test_unverified_or_incomplete_forecast_does_not_invent_plan(self) -> None:
        observed = observation(round_no=1, gold=25)
        forecast = self._forecast(
            risk_level=RiskLevel.UNKNOWN,
            predicted_damage=0,
            effective_hp=500,
            margin=500,
            complete=False,
        )

        plan = find_cheapest_safe_plan(
            observed,
            forecast,
            BuildPlan(weapon_loadout=('gatling',)),
        )

        self.assertIs(plan.status, SafetyPlanStatus.UNKNOWN)
        self.assertEqual(plan.actions, ())

    def test_incomplete_risk_forecast_is_unknown_not_unavailable(self) -> None:
        observed = observation(round_no=71, gold=25)

        for risk_level in (RiskLevel.WATCH, RiskLevel.CRITICAL):
            with self.subTest(risk_level=risk_level):
                forecast = self._forecast(
                    risk_level=risk_level,
                    predicted_damage=600,
                    effective_hp=500,
                    margin=-100,
                    complete=False,
                )

                plan = find_cheapest_safe_plan(
                    observed,
                    forecast,
                    BuildPlan(weapon_loadout=('gatling',)),
                )

                self.assertIs(plan.status, SafetyPlanStatus.UNKNOWN)
                self.assertEqual(plan.actions, ())

    def test_unverified_missing_railgun_keeps_plan_unknown(self) -> None:
        observed = observation(round_no=20, gold=25)
        forecast = self._forecast(
            risk_level=RiskLevel.WATCH,
            predicted_damage=20,
            effective_hp=30,
            margin=10,
        )

        plan = find_cheapest_safe_plan(
            observed,
            forecast,
            BuildPlan(weapon_loadout=('railgun',), build_walls=False),
        )

        self.assertIs(plan.status, SafetyPlanStatus.UNKNOWN)
        self.assertEqual(plan.actions, ())

    def test_safe_forecast_overrides_legacy_low_station_threshold(self) -> None:
        observed = observation(
            round_no=20,
            gold=200,
            our_units=(
                unit(1, 5, 5, 'station', health=30, level=1),
                unit(2, 2, 2, 'worker'),
                unit(3, 3, 3, 'gatling', level=1),
                unit(4, 4, 3, 'railgun', level=1),
                unit(5, 5, 3, 'rocket', level=1),
            ),
        )

        decision = StrategicDirector().select(
            observed,
            forecast=self._forecast(
                risk_level=RiskLevel.SAFE,
                predicted_damage=0,
                effective_hp=30,
                margin=30,
            ),
        )

        self.assertIs(decision.intent.profile, StrategyProfile.SCORE)
        self.assertEqual(decision.intent.gold_reserve, 0)

    def test_critical_unavailable_plan_reserves_all_gold(self) -> None:
        observed = observation(
            round_no=20,
            gold=80,
            our_units=(
                unit(1, 5, 5, 'station', health=500, level=1),
                unit(2, 2, 2, 'worker'),
                unit(3, 3, 3, 'gatling', level=1),
                unit(4, 4, 3, 'railgun', level=1),
                unit(5, 5, 3, 'rocket', level=1),
            ),
        )

        decision = StrategicDirector().select(
            observed,
            forecast=self._forecast(
                risk_level=RiskLevel.CRITICAL,
                predicted_damage=600,
                effective_hp=500,
                margin=-100,
            ),
        )

        self.assertIs(decision.intent.profile, StrategyProfile.SURVIVE)
        self.assertEqual(decision.intent.gold_reserve, 80)

    def test_only_matching_safety_action_can_consume_reserve(self) -> None:
        observed = observation(
            round_no=1,
            gold=25,
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 10, 10, 'station', health=500, level=1),
            ),
        )
        world = WorldGrid.from_observation(observed)
        planned = TacticalCandidate.build(
            1,
            Position(1, 1),
            Position(2, 1),
            'gatling',
            400,
            gold_cost=25,
            reserve_eligible=True,
        )
        discretionary = TacticalCandidate.build(
            1,
            Position(1, 1),
            Position(2, 1),
            'gatling',
            400,
            gold_cost=25,
        )

        self.assertTrue(
            is_valid_joint(
                observed,
                world,
                (planned,),
                gold_reserve=25,
            )
        )
        self.assertFalse(
            is_valid_joint(
                observed,
                world,
                (discretionary,),
                gold_reserve=25,
            )
        )

    @staticmethod
    def _forecast(
        *,
        risk_level: RiskLevel,
        predicted_damage: int,
        effective_hp: int,
        margin: int,
        complete: bool = True,
    ) -> NightForecast:
        return NightForecast(
            expected_station_hp_at_dawn=max(0, effective_hp - predicted_damage),
            predicted_damage_before_dawn=predicted_damage,
            effective_defense_hp=effective_hp,
            future_firepower=0,
            survival_margin=margin,
            risk_ratio=Fraction(predicted_damage, max(1, effective_hp)),
            risk_level=risk_level,
            expected_wall_losses=0,
            expected_weapon_losses=0,
            expected_role_losses=0,
            lethal_round=None,
            critical_robot_ids=(),
            critical_wall_ids=(),
            generated_round=71,
            updated_round=71,
            day_no=1,
            model_version='night-forecast-v1',
            update_kind=ForecastUpdateKind.FULL,
            complete=complete,
            uncertainty_reasons=(() if complete else ('test',)),
            observed_station_hp=effective_hp,
            expected_next_station_hp=effective_hp,
        )


if __name__ == '__main__':
    unittest.main()
