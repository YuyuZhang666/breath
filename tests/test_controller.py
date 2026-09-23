import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from future_war_agent.controller import (
    DEFAULT_STRATEGY_ENGINE,
    _SOP_PATH,
    default_planner,
    handle_payload,
)
from future_war_agent.deadline import RequestBudget, RequestDeadlineExceeded
from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.simulation.candidates import SimJointAction
from future_war_agent.telemetry import TelemetryRecorder


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

    def test_default_controller_enables_bounded_task_runtime(self) -> None:
        task_agent = DEFAULT_STRATEGY_ENGINE._task_agent

        self.assertTrue(task_agent._commands_enabled)
        self.assertEqual(task_agent._sop_store.path, _SOP_PATH)

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

    def test_action_counts_cover_validation_and_serialization_boundaries(self) -> None:
        telemetry = TelemetryRecorder()

        def planner(_: Observation) -> Decision:
            return Decision(
                commands={
                    10010: Action.move(Position(5, 24)),
                    99999: Action.move(Position(1, 1)),
                }
            )

        response = handle_payload(
            self.payload,
            planner=planner,
            telemetry=telemetry,
        )

        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.pre_validation_action_count, 2)
        self.assertEqual(sample.post_validation_action_count, 1)
        self.assertEqual(sample.serialized_action_count, 1)
        self.assertEqual(sample.response_action_count, 1)
        self.assertEqual(len(sample.planned_action_log), 2)
        self.assertEqual(len(sample.validated_action_log), 1)
        self.assertEqual(len(sample.validation_drop_log), 1)
        self.assertIn('id=99999', sample.validation_drop_log[0])
        self.assertIn('validated=dropped', sample.validation_drop_log[0])
        lifecycle = {
            int(entry.split(':', 1)[0].split('=', 1)[1]): entry
            for entry in sample.action_lifecycle_log
        }
        self.assertIn('planned=move', lifecycle[10010])
        self.assertIn('validated=move', lifecycle[10010])
        self.assertIn('response=move', lifecycle[10010])
        self.assertIn('type=unknown', lifecycle[99999])
        self.assertIn('validated=none', lifecycle[99999])
        self.assertIn('response=none', lifecycle[99999])
        self.assertEqual(len(sample.action_override_log), 1)
        self.assertIn('ACTION_OVERRIDE:id=99999', sample.action_override_log[0])
        self.assertIn('module=validator', sample.action_override_log[0])
        self.assertTrue(sample.validated_weapon_action_log)
        self.assertEqual(set(response['roleCommandMap']), {'10010'})

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

    def test_external_request_start_reaches_budget_aware_planner(self) -> None:
        budgets: list[RequestBudget] = []

        def planner(
            observed: Observation,
            *,
            request_budget: RequestBudget,
        ) -> Decision:
            self.assertEqual(observed.time.round_no, 85)
            budgets.append(request_budget)
            return Decision()

        response = handle_payload(
            self.payload,
            planner=planner,
            request_started_at=12.5,
            clock=lambda: 12.75,
        )

        self.assertEqual(response, safe_payload())
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0].started_at, 12.5)
        self.assertEqual(budgets[0].compute_deadline, 15.5)
        self.assertEqual(budgets[0].response_deadline, 16.0)

    def test_expired_external_request_skips_parser_and_planner(self) -> None:
        called = False

        def planner(_: Observation) -> Decision:
            nonlocal called
            called = True
            return Decision(prompt="too-late")

        with patch(
            "future_war_agent.controller.parse_observation"
        ) as parser:
            response = handle_payload(
                self.payload,
                planner=planner,
                request_started_at=1.0,
                clock=lambda: 4.1,
            )

        self.assertEqual(response, safe_payload())
        self.assertFalse(called)
        parser.assert_not_called()

    def test_validation_is_skipped_when_planner_uses_compute_budget(self) -> None:
        times = iter((10.0, 10.0, 13.1, 13.1))

        def planner(_: Observation) -> Decision:
            return Decision(prompt="late")

        with patch(
            "future_war_agent.controller.validate_decision"
        ) as validator:
            response = handle_payload(
                self.payload,
                planner=planner,
                request_started_at=10.0,
                clock=lambda: next(times),
            )

        self.assertEqual(response, safe_payload())
        validator.assert_not_called()

    def test_request_budget_helpers_use_absolute_deadlines(self) -> None:
        now = [2.0]
        budget = RequestBudget.start(
            started_at=1.0,
            response_budget_seconds=4.0,
            compute_budget_seconds=3.0,
            clock=lambda: now[0],
        )

        self.assertEqual(budget.remaining_compute(), 2.0)
        self.assertEqual(budget.remaining_response(), 3.0)
        self.assertEqual(budget.child_deadline(0.5), 2.5)
        now[0] = 4.0
        self.assertTrue(budget.compute_exhausted())
        self.assertFalse(budget.response_exhausted())
        with self.assertRaises(RequestDeadlineExceeded):
            budget.checkpoint("planner")

    def test_invalid_payload_returns_safe_payload(self) -> None:
        self.assertEqual(handle_payload({}), safe_payload())

    def test_planner_exception_returns_safe_payload(self) -> None:
        def broken(_: Observation) -> Decision:
            raise RuntimeError("private failure detail")

        self.assertEqual(handle_payload(self.payload, planner=broken), safe_payload())


    def test_planner_deadline_returns_safe_payload_with_timeout_reason(self) -> None:
        telemetry = TelemetryRecorder()

        def expired(_: Observation) -> Decision:
            raise RequestDeadlineExceeded('synthetic planner deadline')

        response = handle_payload(
            self.payload,
            planner=expired,
            telemetry=telemetry,
        )

        self.assertEqual(response, safe_payload())
        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.fallback_reason, 'deadline_low')
        self.assertTrue(sample.timeout_prevented)
        self.assertEqual(sample.decision_source, 'safe')


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
