import unittest
from dataclasses import FrozenInstanceError, fields

from future_war_agent.strategy.policy import (
    DEFAULT_BUILD_PLAN,
    DEFAULT_STRATEGIC_INTENT,
    BuildPlan,
    DayPriorities,
    ItemPolicy,
    RuleFeatureFlags,
    StrategicIntent,
    StrategyProfile,
    intent_for_profile,
)
from future_war_agent.strategy.simulation.objective import NightObjective


class StrategyPolicyTests(unittest.TestCase):
    def test_defaults_prioritize_fast_parallel_construction(self) -> None:
        self.assertEqual(
            DEFAULT_BUILD_PLAN.weapon_loadout,
            ("gatling", "railgun", "rocket"),
        )
        self.assertEqual(DEFAULT_BUILD_PLAN.minimum_weapons_before_walls, 0)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_wall_force_round, 1)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_stone_batch_target, 5)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_gate_close_round, 1)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_critical_wall_count, 12)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_early_weapon_target, 2)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_early_weapon_deadline_round, 15)
        self.assertEqual(DEFAULT_BUILD_PLAN.opening_early_weapon_priority_boost, 50)
        self.assertEqual(DEFAULT_STRATEGIC_INTENT.profile, StrategyProfile.ECONOMY)
        self.assertEqual(DEFAULT_STRATEGIC_INTENT.gold_reserve, 0)
        self.assertEqual(DEFAULT_STRATEGIC_INTENT.item_policy.medicine_stock, 0)
        self.assertEqual(RuleFeatureFlags(), RuleFeatureFlags())
        flags = RuleFeatureFlags()
        self.assertFalse(any(getattr(flags, field.name) for field in fields(flags)))

    def test_policy_values_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_BUILD_PLAN.wall_site_limit = 3  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_STRATEGIC_INTENT.gold_reserve = 10  # type: ignore[misc]

    def test_invalid_policy_values_are_rejected(self) -> None:
        invalid_factories = (
            lambda: BuildPlan(weapon_loadout=("",)),
            lambda: BuildPlan(weapon_loadout=("gatling", "gatling")),
            lambda: BuildPlan(wall_site_limit=-1),
            lambda: BuildPlan(minimum_weapons_before_walls=-1),
            lambda: BuildPlan(opening_wall_force_round=0),
            lambda: BuildPlan(opening_stone_batch_target=0),
            lambda: BuildPlan(opening_gate_close_round=71),
            lambda: BuildPlan(opening_critical_wall_count=-1),
            lambda: BuildPlan(opening_early_weapon_target=-1),
            lambda: BuildPlan(opening_early_weapon_deadline_round=0),
            lambda: BuildPlan(opening_early_weapon_priority_boost=-1),
            lambda: BuildPlan(critical_wall_priority_boost=-1),
            lambda: BuildPlan(threat_wall_priority_boost=-1),
            lambda: BuildPlan(rebuild_wall_priority_boost=-1),
            lambda: BuildPlan(max_wall_job_candidates=0),
            lambda: BuildPlan(minimum_wall_stock=-1),
            lambda: DayPriorities(recall=0),
            lambda: ItemPolicy(medicine_health_threshold=-1),
            lambda: ItemPolicy(medicine_stock=-1),
            lambda: StrategicIntent(gold_reserve=-1),
            lambda: StrategicIntent(minimum_hold_rounds=-1),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()

    def test_profile_intents_express_distinct_risk_budgets(self) -> None:
        survive = intent_for_profile(StrategyProfile.SURVIVE, "station damage")
        economy = intent_for_profile(StrategyProfile.ECONOMY, "reserve incomplete")
        score = intent_for_profile(StrategyProfile.SCORE, "defense funded")
        desperation = intent_for_profile(StrategyProfile.DESPERATION, "fatal threat")

        self.assertGreater(survive.gold_reserve, economy.gold_reserve)
        self.assertTrue(survive.night_objective.protect_all_controllers)
        self.assertTrue(score.allow_tasks)
        self.assertTrue(economy.allow_tasks)
        self.assertTrue(DEFAULT_STRATEGIC_INTENT.allow_tasks)
        self.assertEqual(desperation.gold_reserve, 0)
        self.assertFalse(desperation.night_objective.protect_all_weapons)
        self.assertEqual(survive.transition_reason, "station damage")

    def test_custom_intent_keeps_declared_night_objective(self) -> None:
        objective = NightObjective(
            minimum_station_health=50,
            protect_all_controllers=False,
            protect_all_weapons=False,
            require_post_horizon_buffer=False,
            enable_score_band=True,
        )
        intent = StrategicIntent(night_objective=objective)

        self.assertIs(intent.night_objective, objective)


if __name__ == "__main__":
    unittest.main()
