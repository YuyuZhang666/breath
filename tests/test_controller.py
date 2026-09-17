import json
import unittest
from pathlib import Path

from future_war_agent.controller import handle_payload
from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation, Position


STRATEGY_FIXTURE = Path(__file__).parent / "fixtures" / "strategy_request.json"
FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_default_controller_runs_phase_2_strategy(self) -> None:
        strategy_payload = json.loads(STRATEGY_FIXTURE.read_text(encoding="utf-8"))

        response = handle_payload(strategy_payload)

        self.assertTrue(response["roleCommandMap"])
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")

    def test_injected_planner_flows_through_validation(self) -> None:
        def planner(observed: Observation) -> Decision:
            self.assertEqual(observed.time.round_no, 85)
            return Decision(commands={10010: Action.move(Position(5, 24))})

        response = handle_payload(self.payload, planner=planner)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")

    def test_invalid_payload_returns_safe_payload(self) -> None:
        self.assertEqual(handle_payload({}), safe_payload())

    def test_planner_exception_returns_safe_payload(self) -> None:
        def broken(_: Observation) -> Decision:
            raise RuntimeError("private failure detail")

        self.assertEqual(handle_payload(self.payload, planner=broken), safe_payload())


if __name__ == "__main__":
    unittest.main()
