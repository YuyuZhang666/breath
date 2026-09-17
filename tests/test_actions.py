import unittest

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.protocol.models import Position


class ActionTests(unittest.TestCase):
    def test_named_constructors_cover_protocol_actions(self) -> None:
        position = Position(3, 4)
        cases = {
            ActionKind.MOVE: Action.move(position),
            ActionKind.ATTACK: Action.attack(10010, (position,)),
            ActionKind.SELL: Action.sell("stone", 2),
            ActionKind.BUY: Action.buy("Medicine", 1),
            ActionKind.BUILD: Action.build("wall", position),
            ActionKind.REMOVE: Action.remove(position),
            ActionKind.ACCEPT_TASK: Action.accept_task(),
            ActionKind.SUBMIT_ANSWER: Action.submit_answer("answer"),
            ActionKind.SUMMON_TREASURE: Action.summon_treasure(
                position, ("StarSand",)
            ),
            ActionKind.USE: Action.use("Medicine"),
            ActionKind.DROP: Action.drop("stone"),
            ActionKind.COLLECT: Action.collect(position),
        }

        self.assertEqual(set(cases), set(ActionKind))
        for kind, action in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(action.kind, kind)

    def test_action_copies_iterables_to_tuples(self) -> None:
        targets = [Position(1, 2)]
        items = ["StarSand"]
        attack = Action.attack(10010, targets)
        treasure = Action.summon_treasure(targets[0], items)
        targets.append(Position(2, 3))
        items.append("FlameBreath")

        self.assertEqual(attack.target_positions, (Position(1, 2),))
        self.assertEqual(treasure.items, ("StarSand",))


if __name__ == "__main__":
    unittest.main()
