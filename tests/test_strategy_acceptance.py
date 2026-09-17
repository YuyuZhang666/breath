import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.planner import plan_turn
from tests.strategy_helpers import observation, unit


class StrategyAcceptanceTests(unittest.TestCase):
    def test_worker_approaches_a_reachable_mine(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, "worker"),),
            zones=(Zone(Position(4, 1), "iron"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10010].kind, ActionKind.MOVE)

    def test_adjacent_worker_collects(self) -> None:
        mine = Position(3, 2)
        observed = observation(
            our_units=(unit(10010, 2, 2, "worker"),),
            zones=(Zone(mine, "iron"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10010].kind, ActionKind.COLLECT)
        self.assertEqual(decision.commands[10010].target_positions, (mine,))

    def test_full_worker_sells_to_adjacent_vendor(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    "worker",
                    backpack_capacity=1,
                    backpack=("iron",),
                ),
            ),
            zones=(Zone(Position(1, 1), "vendor"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        decision = plan_turn(observed)

        action = decision.commands[10010]
        self.assertEqual(action.kind, ActionKind.SELL)
        self.assertEqual((action.name, action.quantity), ("iron", 1))

    def test_two_adjacent_workers_collect_the_same_mine(self) -> None:
        mine = Position(3, 3)
        observed = observation(
            our_units=(
                unit(10010, 2, 3, "worker"),
                unit(10012, 3, 2, "worker"),
            ),
            zones=(Zone(mine, "stone"),),
        )

        decision = plan_turn(observed)

        self.assertEqual(set(decision.commands), {10010, 10012})
        self.assertTrue(
            all(
                action.kind is ActionKind.COLLECT
                and action.target_positions == (mine,)
                for action in decision.commands.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
