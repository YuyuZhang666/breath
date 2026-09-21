import copy
import json
import unittest
from dataclasses import replace
from math import ceil
from pathlib import Path
from time import perf_counter

from future_war_agent.controller import handle_payload
from future_war_agent.deadline import RequestBudget, RequestDeadlineExceeded
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.forecast import (
    ForecastUpdateKind,
    refresh_night_forecast,
)
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.strategy.task_agent import TaskAgent
from future_war_agent.telemetry import TelemetryRecorder


FIXTURES = Path(__file__).parent / 'fixtures'
DAY_FIXTURE = FIXTURES / 'phase3_day_request.json'
NIGHT_FIXTURE = FIXTURES / 'phase3_night_request.json'


def load_observation(path: Path) -> Observation:
    return parse_observation(json.loads(path.read_text(encoding='utf-8')))


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Phase2Spy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: Observation, **kwargs: object) -> Decision:
        del observation, kwargs
        self.calls += 1
        return Decision(prompt=f'phase2-{self.calls}')


class TaskApplySpy(TaskAgent):
    def __init__(self) -> None:
        super().__init__()
        self.apply_calls = 0

    def apply(self, *args: object, **kwargs: object):
        self.apply_calls += 1
        return super().apply(*args, **kwargs)


class RequestBudgetTests(unittest.TestCase):
    def test_absolute_deadlines_share_one_fake_clock(self) -> None:
        clock = FakeClock(10.0)
        budget = RequestBudget.start(
            started_at=10.0,
            response_budget_seconds=3.5,
            compute_budget_seconds=3.0,
            clock=clock,
        )

        self.assertEqual(budget.response_deadline, 13.5)
        self.assertEqual(budget.compute_deadline, 13.0)
        self.assertEqual(budget.remaining_compute(), 3.0)
        self.assertEqual(budget.child_deadline(5.0), 13.0)

        clock.advance(2.75)
        self.assertAlmostEqual(budget.remaining_compute(), 0.25)
        self.assertAlmostEqual(budget.remaining_response(), 0.75)
        self.assertEqual(budget.child_deadline(0.1), 12.85)
        budget.checkpoint('still within compute budget')

        clock.advance(0.25)
        self.assertTrue(budget.compute_exhausted())
        self.assertFalse(budget.response_exhausted())
        with self.assertRaises(RequestDeadlineExceeded):
            budget.checkpoint('forecast')
        budget.checkpoint('serialize', response=True)

        clock.advance(0.5)
        with self.assertRaises(RequestDeadlineExceeded):
            budget.checkpoint('write response', response=True)


class DeadlineNightReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = load_observation(DAY_FIXTURE)
        self.night = load_observation(NIGHT_FIXTURE)

    def test_default_engine_explicitly_disables_full_night_forecast(self) -> None:
        allow_full_values: list[bool] = []
        update_kinds: list[ForecastUpdateKind] = []

        def forecast_spy(
            observation: Observation,
            *,
            allow_full: bool = True,
            **kwargs: object,
        ):
            allow_full_values.append(allow_full)
            refreshed = refresh_night_forecast(
                observation,
                allow_full=allow_full,
                **kwargs,
            )
            update_kinds.append(refreshed.forecast.update_kind)
            return refreshed

        engine = StrategyEngine(
            phase2_planner=Phase2Spy(),
            night_forecaster=forecast_spy,
            clock=lambda: 100.0,
        )

        decision = engine.plan(self.night)

        self.assertIsInstance(decision, Decision)
        self.assertEqual(allow_full_values, [False])
        self.assertEqual(update_kinds, [ForecastUpdateKind.LIGHTWEIGHT])

    def test_forecast_deadline_returns_legal_phase2_decision_and_reason(self) -> None:
        telemetry = TelemetryRecorder()
        phase2 = Phase2Spy()
        search_calls = 0

        def timeout_forecaster(
            observation: Observation,
            **kwargs: object,
        ):
            del observation, kwargs
            raise DeadlineExceeded('synthetic forecast watchdog')

        def forbidden_search(*args: object, **kwargs: object):
            nonlocal search_calls
            del args, kwargs
            search_calls += 1
            raise AssertionError('search must not run after forecast timeout')

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_forecaster=timeout_forecaster,
            night_searcher=forbidden_search,
            clock=lambda: 100.0,
            telemetry=telemetry,
        )

        engine.plan(self.day)
        decision = engine.plan(self.night)
        payload = decision_to_payload(decision)

        self.assertEqual(
            set(payload),
            {'roleCommandMap', 'prompt', 'executeCmd'},
        )
        self.assertIsInstance(payload['roleCommandMap'], dict)
        self.assertEqual(decision.prompt, 'phase2-2')
        self.assertEqual(search_calls, 0)
        sample = telemetry.snapshot()[-1]
        self.assertTrue(sample.fallback_used)
        self.assertTrue(sample.timeout_prevented)
        self.assertTrue(sample.watchdog_hit)
        self.assertEqual(sample.fallback_reason, 'forecast_timeout')
        self.assertEqual(sample.forecast_reason, 'forecast_timeout')

    def test_emergency_reserve_skips_phase3_and_task_apply(self) -> None:
        clock = FakeClock(0.0)
        phase2 = Phase2Spy()
        task_agent = TaskApplySpy()
        search_calls = 0
        telemetry = TelemetryRecorder()

        def forbidden_search(*args: object, **kwargs: object):
            nonlocal search_calls
            del args, kwargs
            search_calls += 1
            raise AssertionError('phase 3 must not run inside emergency reserve')

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_searcher=forbidden_search,
            task_agent=task_agent,
            clock=clock,
            telemetry=telemetry,
        )
        engine.plan(self.day)
        normal_phase2_calls = phase2.calls
        normal_task_calls = task_agent.apply_calls
        clock.now = 2.6
        budget = RequestBudget.start(
            started_at=0.0,
            response_budget_seconds=3.5,
            compute_budget_seconds=3.0,
            clock=clock,
        )

        decision = engine.plan(self.night, request_budget=budget)
        payload = decision_to_payload(decision)

        self.assertEqual(search_calls, 0)
        self.assertEqual(phase2.calls, normal_phase2_calls)
        self.assertEqual(task_agent.apply_calls, normal_task_calls)
        self.assertEqual(payload['roleCommandMap'], {})
        self.assertEqual(set(payload), {'roleCommandMap', 'prompt', 'executeCmd'})
        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.compute_governor_action, 'emergency_reserve')
        self.assertEqual(sample.fallback_reason, 'deadline_low')
        self.assertEqual(sample.decision_source, 'safe')
        self.assertTrue(sample.timeout_prevented)

    def test_round_71_to_130_replay_never_requests_synchronous_full(self) -> None:
        allow_full_values: list[bool] = []
        update_kinds: list[ForecastUpdateKind] = []

        def forecast_spy(
            observation: Observation,
            *,
            allow_full: bool = True,
            **kwargs: object,
        ):
            if allow_full:
                raise AssertionError('night replay requested synchronous full forecast')
            allow_full_values.append(allow_full)
            refreshed = refresh_night_forecast(
                observation,
                allow_full=allow_full,
                **kwargs,
            )
            update_kinds.append(refreshed.forecast.update_kind)
            return refreshed

        engine = StrategyEngine(
            phase2_planner=Phase2Spy(),
            night_forecaster=forecast_spy,
            clock=lambda: 100.0,
        )
        decisions: list[Decision] = []

        for round_no in range(71, 131):
            observed = replace(
                self.night,
                time=TurnTime.from_round(round_no),
            )
            decisions.append(engine.plan(observed))

        self.assertEqual(len(allow_full_values), 60)
        self.assertTrue(all(value is False for value in allow_full_values))
        self.assertNotIn(ForecastUpdateKind.FULL, update_kinds)
        self.assertEqual(update_kinds[0], ForecastUpdateKind.LIGHTWEIGHT)
        self.assertTrue(
            all(
                set(decision_to_payload(decision))
                == {'roleCommandMap', 'prompt', 'executeCmd'}
                for decision in decisions
            )
        )

    def test_round_1_to_130_controller_path_meets_deadline_baseline(self) -> None:
        day_payload = json.loads(DAY_FIXTURE.read_text(encoding='utf-8'))
        night_payload = json.loads(NIGHT_FIXTURE.read_text(encoding='utf-8'))
        telemetry = TelemetryRecorder()
        engine = StrategyEngine(telemetry=telemetry)
        elapsed_seconds: list[float] = []
        night_forecast_modes: list[str] = []

        for round_no in range(1, 131):
            payload = copy.deepcopy(
                day_payload if round_no <= 70 else night_payload
            )
            payload['roundNo'] = round_no
            started = perf_counter()
            response = handle_payload(
                payload,
                planner=engine.plan,
                telemetry=telemetry,
            )
            elapsed_seconds.append(perf_counter() - started)
            sample = telemetry.snapshot()[-1]

            self.assertEqual(
                set(response),
                {'roleCommandMap', 'prompt', 'executeCmd'},
            )
            friendly_ids = {
                str(item['id']) for item in payload['teamOur']['roles']
            }
            self.assertTrue(set(response['roleCommandMap']) <= friendly_ids)
            self.assertLess(sample.request_total_ms, 5_000)
            if round_no >= 71:
                night_forecast_modes.append(sample.forecast_mode)

        samples = telemetry.snapshot()
        ordered = sorted(elapsed_seconds)
        p99 = ordered[ceil(0.99 * len(ordered)) - 1]
        self.assertEqual(len(samples), 130)
        self.assertEqual(
            tuple(sample.round_no for sample in samples),
            tuple(range(1, 131)),
        )
        self.assertTrue(
            all(sample.fallback_reason == 'normal' for sample in samples)
        )
        self.assertFalse(any(sample.watchdog_hit for sample in samples))
        self.assertFalse(any(sample.timeout_prevented for sample in samples))
        self.assertTrue(
            all(sample.forecast_mode == 'none' for sample in samples[:70])
        )
        self.assertLess(max(elapsed_seconds), 5.0)
        self.assertLess(p99, 3.0)
        self.assertNotIn(ForecastUpdateKind.FULL.value, night_forecast_modes)


if __name__ == '__main__':
    unittest.main()
