import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import action_to_payload, decision_to_payload
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Position


class SerializerTests(unittest.TestCase):
    def test_serializes_protocol_field_names(self) -> None:
        position = Position(3, 4)
        decision = Decision(
            commands={
                10020: Action.attack(10010, (position,)),
                10011: Action.submit_answer("answer"),
            },
            prompt="prompt text",
            execute_command="python solve.py",
        )

        self.assertEqual(
            decision_to_payload(decision),
            {
                "roleCommandMap": {
                    "10011": {
                        "action": "submitAnswer",
                        "taskAnswer": "answer",
                    },
                    "10020": {
                        "action": "attack",
                        "controllerId": "10010",
                        "targetPos": [{"x": 3, "y": 4}],
                    },
                },
                "prompt": "prompt text",
                "executeCmd": "python solve.py",
            },
        )

    def test_safe_payload_is_complete(self) -> None:
        self.assertEqual(
            safe_payload(),
            {
                "roleCommandMap": {},
                "prompt": "",
                "executeCmd": "",
            },
        )

    def test_serializes_every_action_shape(self) -> None:
        position = Position(3, 4)
        cases = (
            (
                Action.move(position),
                {"action": "move", "targetPos": [{"x": 3, "y": 4}]},
            ),
            (
                Action.attack(10010, (position,)),
                {
                    "action": "attack",
                    "controllerId": "10010",
                    "targetPos": [{"x": 3, "y": 4}],
                },
            ),
            (
                Action.sell("stone", 2),
                {"action": "sell", "name": "stone", "num": 2},
            ),
            (
                Action.buy("Medicine", 1),
                {"action": "buy", "name": "Medicine", "num": 1},
            ),
            (
                Action.build("wall", position),
                {
                    "action": "build",
                    "name": "wall",
                    "targetPos": [{"x": 3, "y": 4}],
                },
            ),
            (
                Action.remove(position),
                {"action": "remove", "targetPos": [{"x": 3, "y": 4}]},
            ),
            (Action.accept_task(), {"action": "acceptTask"}),
            (
                Action.submit_answer("answer"),
                {"action": "submitAnswer", "taskAnswer": "answer"},
            ),
            (
                Action.summon_treasure(position, ("StarSand",)),
                {
                    "action": "summonTreasure",
                    "targetPos": [{"x": 3, "y": 4}],
                    "item": ["StarSand"],
                },
            ),
            (
                Action.use("Medicine"),
                {"action": "use", "name": "Medicine"},
            ),
            (Action.drop("stone"), {"action": "drop", "name": "stone"}),
            (
                Action.collect(position),
                {"action": "collect", "targetPos": [{"x": 3, "y": 4}]},
            ),
        )

        for action, expected in cases:
            with self.subTest(action=action.kind):
                self.assertEqual(action_to_payload(action), expected)

    def test_decision_copies_command_mapping(self) -> None:
        commands = {10010: Action.move(Position(1, 1))}
        decision = Decision(commands=commands)
        commands.clear()

        self.assertIn(10010, decision.commands)
        with self.assertRaises(TypeError):
            decision.commands[10011] = Action.accept_task()


if __name__ == "__main__":
    unittest.main()
