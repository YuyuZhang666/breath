import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.validator import validate_decision
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.planner import plan_turn
from future_war_agent.strategy.policy import (
    ItemPolicy,
    RuleFeatureFlags,
    StrategicIntent,
    StrategyProfile,
)
from tests.strategy_helpers import observation, robot, unit


def phase4_observation(
    round_no: int,
    *,
    defended: bool,
    station_health: int = 1000,
    robots=(),
):
    units = [
        unit(10, 5, 5, "station", health=station_health, level=1),
        unit(14, 2, 2, "worker"),
    ]
    if defended:
        units.extend(
            (
                unit(11, 3, 3, "gatling", level=1),
                unit(12, 4, 3, "railgun", level=1),
                unit(13, 5, 3, "rocket", level=1),
            )
        )
    return observation(
        round_no=round_no,
        our_units=tuple(units),
        robots=robots,
        gold=200,
    )


class Phase4AcceptanceTests(unittest.TestCase):
    def test_engine_moves_from_economy_to_score_after_hold_period(self) -> None:
        engine = StrategyEngine()

        engine.plan(phase4_observation(1, defended=False))
        first = engine._sessions.get("team")
        for round_no in range(2, 6):
            engine.plan(phase4_observation(round_no, defended=True))
        settled = engine._sessions.get("team")

        self.assertEqual(first.intent.profile, StrategyProfile.ECONOMY)
        self.assertEqual(settled.intent.profile, StrategyProfile.SCORE)
        self.assertEqual(settled.director_state.since_round, 5)

    def test_station_damage_overrides_score_hold_in_real_engine(self) -> None:
        engine = StrategyEngine()
        for round_no in range(1, 6):
            engine.plan(phase4_observation(round_no, defended=True))
        engine.plan(
            phase4_observation(
                6,
                defended=True,
                station_health=900,
                robots=(robot(20, 7, 7, role_type="mediumRobot"),),
            )
        )

        session = engine._sessions.get("team")

        self.assertEqual(session.intent.profile, StrategyProfile.SURVIVE)
        self.assertEqual(session.director_state.since_round, 6)

    def test_confirmed_medicine_action_survives_final_validation(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(
                    14,
                    4,
                    5,
                    "worker",
                    health=50,
                    backpack=("Medicine",),
                ),
                unit(11, 5, 5, "gatling", level=1, attack_range=5),
            ),
        )
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            item_policy=ItemPolicy(
                medicine_health_threshold=100,
                medicine_stock=2,
            ),
        )

        validated = validate_decision(
            observed,
            plan_turn(observed, intent=intent),
        )

        self.assertEqual(validated.commands[14].kind, ActionKind.USE)
        self.assertEqual(validated.commands[14].name, "Medicine")

    def test_every_experimental_rule_is_disabled_by_default(self) -> None:
        flags = RuleFeatureFlags()

        self.assertFalse(flags.enable_pressure)
        self.assertFalse(flags.enable_bomb)
        self.assertFalse(flags.enable_stun)
        self.assertFalse(flags.enable_repairs)
        self.assertFalse(flags.enable_upgrades)
        self.assertFalse(flags.enable_enemy_wall_removal)
        self.assertFalse(flags.enable_blind_fire)
        self.assertFalse(flags.enable_rocket_structure_damage)
        self.assertFalse(flags.enable_treasure_retry)


if __name__ == "__main__":
    unittest.main()
