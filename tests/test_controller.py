import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from future_war_agent.controller import default_planner, handle_payload
from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.simulation.candidates import SimJointAction


STRATEGY_FIXTURE = Path(__file__).parent / "fixtures" / "strategy_request.json"
FIXTURE = Path(__file__).parent / "fixtures" / "request.json"
PHASE3_DAY_FIXTURE = Path(__file__).parent / "fixtures" / "phase3_day_request.json"
PHASE3_NIGHT_FIXTURE = Path(__file__).parent / "fixtures" / "phase3_night_request.json"


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_default_controller_runs_phase_2_strategy(self) -> None:
        strategy_payload = json.loads(STRATEGY_FIXTURE.read_text(encoding="utf-8"))

        response = handle_payload(strategy_payload)

        self.assertTrue(response["roleCommandMap"])
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")

    def test_default_planner_delegates_to_process_engine(self) -> None:
        observation = parse_observation(self.payload)
        expected = Decision(prompt="engine")

        with patch(
            "future_war_agent.controller.DEFAULT_STRATEGY_ENGINE"
        ) as process_engine:
            process_engine.plan.return_value = expected

            self.assertIs(default_planner(observation), expected)
            process_engine.plan.assert_called_once_with(observation)

    def test_injected_planner_flows_through_validation(self) -> None:
        def planner(observed: Observation) -> Decision:
            self.assertEqual(observed.time.round_no, 85)
            return Decision(commands={10010: Action.move(Position(5, 24))})

        response = handle_payload(self.payload, planner=planner)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")

    def test_request_budget_starts_before_parser_and_reaches_engine(self) -> None:
        starts: list[float] = []

        def planner(
            observed: Observation,
            *,
            request_started_at: float,
        ) -> Decision:
            self.assertEqual(observed.time.round_no, 85)
            starts.append(request_started_at)
            return Decision()

        handle_payload(self.payload, planner=planner)

        self.assertEqual(len(starts), 1)
        self.assertGreater(starts[0], 0)

    def test_invalid_payload_returns_safe_payload(self) -> None:
        self.assertEqual(handle_payload({}), safe_payload())

    def test_planner_exception_returns_safe_payload(self) -> None:
        def broken(_: Observation) -> Decision:
            raise RuntimeError("private failure detail")

        self.assertEqual(handle_payload(self.payload, planner=broken), safe_payload())


class Phase3ControllerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day_payload = json.loads(
            PHASE3_DAY_FIXTURE.read_text(encoding="utf-8")
        )
        self.night_payload = json.loads(
            PHASE3_NIGHT_FIXTURE.read_text(encoding="utf-8")
        )

    def test_day_then_night_returns_nonempty_three_field_schema(self) -> None:
        engine = StrategyEngine()

        day_response = handle_payload(self.day_payload, planner=engine.plan)
        night_response = handle_payload(self.night_payload, planner=engine.plan)

        expected_keys = {"roleCommandMap", "prompt", "executeCmd"}
        self.assertEqual(set(day_response), expected_keys)
        self.assertEqual(set(night_response), expected_keys)
        self.assertTrue(night_response["roleCommandMap"])

    def test_duplicate_night_responses_are_byte_equivalent(self) -> None:
        engine = StrategyEngine()
        handle_payload(self.day_payload, planner=engine.plan)

        first = handle_payload(self.night_payload, planner=engine.plan)
        second = handle_payload(self.night_payload, planner=engine.plan)

        encoded_first = json.dumps(
            first,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        encoded_second = json.dumps(
            second,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(encoded_first, encoded_second)
        self.assertTrue(first["roleCommandMap"])

    def test_missing_explicit_cooldown_forces_phase_2_response(self) -> None:
        fallback = Decision(
            commands={101: Action.move(Position(2, 5))},
            prompt="phase2-marker",
        )
        engine = StrategyEngine(phase2_planner=lambda _: fallback)
        incomplete = copy.deepcopy(self.night_payload)
        weapon = next(
            unit
            for unit in incomplete["teamOur"]["roles"]
            if unit["roleType"] == "gatling"
        )
        del weapon["cooldown"]
        handle_payload(self.day_payload, planner=engine.plan)

        response = handle_payload(incomplete, planner=engine.plan)

        self.assertEqual(response["prompt"], "phase2-marker")
        self.assertEqual(
            response["roleCommandMap"],
            {
                "101": {
                    "action": "move",
                    "targetPos": [{"x": 2, "y": 5}],
                }
            },
        )

    def test_multi_target_phase_3_attack_survives_validation(self) -> None:
        phase3_decision = Decision(
            commands={
                300: Action.attack(
                    101,
                    (Position(5, 4), Position(5, 5)),
                )
            }
        )
        result = SimpleNamespace(
            decision=phase3_decision,
            simulation_action=SimJointAction(),
            certificate=None,
        )
        engine = StrategyEngine(
            night_searcher=lambda *args, **kwargs: result
        )
        handle_payload(self.day_payload, planner=engine.plan)

        response = handle_payload(self.night_payload, planner=engine.plan)

        self.assertEqual(
            response["roleCommandMap"]["300"]["targetPos"],
            [{"x": 5, "y": 4}, {"x": 5, "y": 5}],
        )


if __name__ == "__main__":
    unittest.main()
