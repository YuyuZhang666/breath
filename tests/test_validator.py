import json
import unittest
from dataclasses import replace
from pathlib import Path

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime


FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = parse_observation(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def test_keeps_valid_night_attack(self) -> None:
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(4, 4),)),
            }
        )

        validated = validate_decision(self.observed, decision)

        self.assertIn(10020, validated.commands)

    def test_drops_day_attack_but_keeps_valid_move(self) -> None:
        day = replace(self.observed, time=TurnTime.from_round(1))
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(4, 4),)),
                10010: Action.move(Position(5, 24)),
            }
        )

        validated = validate_decision(day, decision)

        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_drops_out_of_bounds_target(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(41, 0))})

        self.assertEqual(validate_decision(self.observed, decision).commands, {})

    def test_drops_attack_when_controller_has_personal_action(self) -> None:
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(4, 4),)),
                10010: Action.move(Position(5, 24)),
            }
        )

        validated = validate_decision(self.observed, decision)

        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_drops_structurally_incomplete_action(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(5, 24))})
        malformed = replace(decision.commands[10010], target_positions=())

        validated = validate_decision(
            self.observed,
            Decision(commands={10010: malformed}),
        )

        self.assertEqual(validated.commands, {})


if __name__ == "__main__":
    unittest.main()
