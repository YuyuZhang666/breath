import json
import unittest
from pathlib import Path

from future_war_agent.controller import handle_payload
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.telemetry import TelemetryRecorder


FIXTURES = Path(__file__).parent / 'fixtures'


class TelemetryTests(unittest.TestCase):
    def test_nested_context_produces_one_bounded_sample(self) -> None:
        ticks = iter((0, 1_000_000, 3_000_000, 5_000_000))
        recorder = TelemetryRecorder(clock=lambda: next(ticks), max_samples=1)
        token = recorder.begin()
        nested = recorder.begin()
        recorder.identify(team_id='alpha', round_no=7, phase='day')
        with recorder.measure('parser_ms'):
            pass
        recorder.increment('phase3_fallback_count')
        recorder.set(fallback_used=True)

        self.assertIsNone(recorder.finish(nested))
        sample = recorder.finish(token)

        self.assertIsNotNone(sample)
        self.assertEqual(sample.team_id, 'alpha')
        self.assertEqual(sample.round_no, 7)
        self.assertEqual(sample.parser_ms, 2.0)
        self.assertEqual(sample.total_ms, 5.0)
        self.assertEqual(sample.phase3_fallback_count, 1)
        self.assertTrue(sample.fallback_used)
        self.assertEqual(recorder.snapshot(), (sample,))

    def test_controller_and_engine_share_one_complete_turn_record(self) -> None:
        payload = json.loads(
            (FIXTURES / 'strategy_request.json').read_text(encoding='utf-8')
        )
        recorder = TelemetryRecorder()
        engine = StrategyEngine(telemetry=recorder)

        response = handle_payload(
            payload,
            planner=engine.plan,
            telemetry=recorder,
        )

        self.assertEqual(
            set(response),
            {'roleCommandMap', 'prompt', 'executeCmd'},
        )
        samples = recorder.snapshot()
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.team_id, 'team')
        self.assertEqual(sample.round_no, payload['roundNo'])
        self.assertGreater(sample.total_ms, 0)
        self.assertGreaterEqual(sample.parser_ms, 0)
        self.assertGreaterEqual(sample.director_ms, 0)
        self.assertGreaterEqual(sample.phase2_ms, 0)
        self.assertGreaterEqual(sample.task_ms, 0)
        self.assertGreaterEqual(sample.validation_ms, 0)

    def test_invalid_payload_records_fallback_without_breaking_schema(self) -> None:
        recorder = TelemetryRecorder()

        response = handle_payload({}, telemetry=recorder)

        self.assertEqual(
            response,
            {'roleCommandMap': {}, 'prompt': '', 'executeCmd': ''},
        )
        self.assertTrue(recorder.snapshot()[0].fallback_used)

    def test_phase3_records_work_counts_and_segment_times(self) -> None:
        day = parse_observation(
            json.loads(
                (FIXTURES / 'phase3_day_request.json').read_text(
                    encoding='utf-8'
                )
            )
        )
        night = parse_observation(
            json.loads(
                (FIXTURES / 'phase3_night_request.json').read_text(
                    encoding='utf-8'
                )
            )
        )
        recorder = TelemetryRecorder()
        engine = StrategyEngine(telemetry=recorder)

        engine.plan(day)
        engine.plan(night)

        sample = recorder.snapshot()[-1]
        self.assertEqual(sample.phase3_level, 'legacy_full')
        self.assertGreater(sample.root_candidate_count, 0)
        self.assertEqual(sample.scenario_count, 4)
        self.assertGreater(sample.simulation_count, 0)
        self.assertGreaterEqual(sample.candidate_generation_ms, 0)
        self.assertGreater(sample.simulation_ms, 0)

    def test_phase2_5_records_joint_fire_counts_and_time(self) -> None:
        night = parse_observation(
            json.loads(
                (FIXTURES / 'phase3_night_request.json').read_text(
                    encoding='utf-8'
                )
            )
        )
        recorder = TelemetryRecorder()
        engine = StrategyEngine(telemetry=recorder)

        engine.plan(night)

        sample = recorder.snapshot()[-1]
        self.assertGreater(sample.phase2_5_ms, 0)
        self.assertEqual(sample.phase2_5_fallback_count, 0)
        self.assertGreater(sample.phase2_5_combination_count, 0)
        self.assertGreater(sample.phase2_5_active_weapon_count, 0)

    def test_phase3_deadline_records_watchdog_and_fallback(self) -> None:
        day = parse_observation(
            json.loads(
                (FIXTURES / 'phase3_day_request.json').read_text(
                    encoding='utf-8'
                )
            )
        )
        night = parse_observation(
            json.loads(
                (FIXTURES / 'phase3_night_request.json').read_text(
                    encoding='utf-8'
                )
            )
        )

        def expired(*args, **kwargs):
            raise DeadlineExceeded('late')

        recorder = TelemetryRecorder()
        engine = StrategyEngine(night_searcher=expired, telemetry=recorder)
        engine.plan(day)
        engine.plan(night)

        sample = recorder.snapshot()[-1]
        self.assertTrue(sample.watchdog_hit)
        self.assertTrue(sample.fallback_used)
        self.assertEqual(sample.phase3_fallback_count, 1)


if __name__ == '__main__':
    unittest.main()
