import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.planner import plan_turn
from tests.strategy_helpers import observation, robot, unit


class StrategyPlannerTests(unittest.TestCase):
    def test_day_worker_produces_a_legal_action(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
            ),
            zones=(Zone(Position(3, 2), "stone"),),
        )

        decision = plan_turn(observed)
        validated = validate_decision(observed, decision)

        self.assertTrue(validated.commands)

    def test_twilight_moves_remote_role_toward_defense(self) -> None:
        observed = observation(
            round_no=70,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10013, 9, 9, "station", level=1),
                unit(10020, 8, 8, "gatling", level=1, attack_range=4),
            ),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10010].kind, ActionKind.MOVE)

    def test_night_ready_weapon_attacks(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10013, 1, 1, "station", level=1),
                unit(10020, 5, 5, "gatling", level=1, attack_range=5),
            ),
            robots=(robot(30001, 3, 3),),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10020].kind, ActionKind.ATTACK)

    def test_identical_observations_serialize_identically(self) -> None:
        observed = observation(
            our_units=(unit(10010, 2, 2, "worker"),),
            zones=(Zone(Position(3, 2), "iron"),),
            vendor_shop=(ShopItem("iron", 4),),
        )
        self.assertEqual(
            decision_to_payload(plan_turn(observed)),
            decision_to_payload(plan_turn(observed)),
        )


if __name__ == "__main__":
    unittest.main()
