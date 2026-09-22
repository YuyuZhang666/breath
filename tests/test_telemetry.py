import json
import unittest
from pathlib import Path

from future_war_agent.controller import handle_payload
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.telemetry import TelemetryRecorder, TurnTelemetry


FIXTURES = Path(__file__).parent / 'fixtures'


class TelemetryTests(unittest.TestCase):
    def test_observability_fields_have_backward_compatible_defaults(self) -> None:
        sample = TurnTelemetry()

        self.assertEqual(sample.request_total_ms, 0.0)
        self.assertEqual(sample.remaining_deadline_ms, 0.0)
        self.assertEqual(sample.forecast_mode, 'none')
        self.assertEqual(sample.forecast_reason, '')
        self.assertFalse(sample.safe_action_generated)
        self.assertEqual(sample.fallback_reason, 'normal')
        self.assertFalse(sample.timeout_prevented)
        self.assertEqual(sample.decision_source, 'safe')
        self.assertEqual(sample.pre_validation_action_count, 0)
        self.assertEqual(sample.post_validation_action_count, 0)
        self.assertEqual(sample.serialized_action_count, 0)
        self.assertEqual(sample.response_action_count, 0)
        self.assertEqual(sample.phase3_executed_level, 'none')
        self.assertEqual(sample.phase3_skip_reason, 'none')
        self.assertEqual(sample.phase3_completed_root_count, 0)
        self.assertEqual(sample.forecast_generated_round, 0)
        self.assertEqual(sample.forecast_updated_round, 0)
        self.assertEqual(sample.forecast_age_rounds, 0)
        self.assertEqual(sample.forecast_margin_source, 'none')
        self.assertFalse(sample.forecast_conservative_bound)
        self.assertEqual(sample.wall_plan_stage, 'unknown')
        self.assertEqual(sample.wall_blocker, 'unknown')
        self.assertEqual(sample.wall_job_count, 0)
        self.assertEqual(sample.wall_failed_build_log, ())
        self.assertEqual(sample.new_wall_gap_count, 0)
        self.assertEqual(sample.rebuild_wall_gap_count, 0)
        self.assertEqual(sample.historically_built_wall_count, 0)
        self.assertEqual(sample.fortification_anchor_day, 0)
        self.assertEqual(sample.build_failure_count, 0)
        self.assertEqual(sample.build_cooldown_count, 0)
        self.assertEqual(sample.build_reroute_count, 0)
        self.assertEqual(sample.build_failure_log, ())
        self.assertEqual(sample.weapon_build_reroute_log, ())
        self.assertTrue(sample.own_station_alive)
        self.assertEqual(sample.engine_lock_wait_ms, 0.0)
        self.assertFalse(sample.engine_lock_timed_out)
        self.assertEqual(sample.deadline_stage, 'none')
        self.assertEqual(sample.emergency_fire_ms, 0.0)
        self.assertEqual(sample.emergency_fire_action_count, 0)
        self.assertFalse(sample.emergency_fire_deadline_hit)

    def test_observability_fields_are_finished_and_logged(self) -> None:
        ticks = iter((0, 4_000_000))
        recorder = TelemetryRecorder(clock=lambda: next(ticks))
        token = recorder.begin()
        recorder.set(
            request_total_ms=3.5,
            remaining_deadline_ms=496.5,
            forecast_mode='lightweight',
            forecast_reason='critical_risk',
            safe_action_generated=True,
            fallback_reason='deadline_low',
            timeout_prevented=True,
            decision_source='phase2',
            pre_validation_action_count=3,
            post_validation_action_count=2,
            serialized_action_count=1,
            response_action_count=2,
            phase3_executed_level='lite',
            phase3_skip_reason='none',
            phase3_completed_root_count=4,
            forecast_generated_round=71,
            forecast_updated_round=72,
            forecast_age_rounds=1,
            forecast_margin_source='conservative_bound',
            forecast_conservative_bound=True,
            build_failure_count=2,
            build_cooldown_count=1,
            build_reroute_count=1,
            build_failure_log=('rocket@(9,6):failures=2:cooldown=2',),
            weapon_build_reroute_log=(
                'rocket@(9,6)->(9,9):repeated_failure',
            ),
            own_station_alive=False,
            engine_lock_wait_ms=3.25,
            engine_lock_timed_out=True,
            deadline_stage='engine_lock',
            emergency_fire_ms=1.25,
            emergency_fire_action_count=2,
            emergency_fire_deadline_hit=True,
        )

        with self.assertLogs('future_war_agent.telemetry', level='INFO') as logs:
            sample = recorder.finish(token)

        self.assertIsNotNone(sample)
        self.assertEqual(sample.request_total_ms, 3.5)
        self.assertEqual(sample.remaining_deadline_ms, 496.5)
        self.assertEqual(sample.forecast_mode, 'lightweight')
        self.assertEqual(sample.forecast_reason, 'critical_risk')
        self.assertTrue(sample.safe_action_generated)
        self.assertEqual(sample.fallback_reason, 'deadline_low')
        self.assertTrue(sample.timeout_prevented)
        self.assertEqual(sample.decision_source, 'phase2')
        self.assertEqual(sample.pre_validation_action_count, 3)
        self.assertEqual(sample.post_validation_action_count, 2)
        self.assertEqual(sample.serialized_action_count, 1)
        self.assertEqual(sample.response_action_count, 2)
        self.assertEqual(sample.phase3_executed_level, 'lite')
        self.assertEqual(sample.phase3_completed_root_count, 4)
        self.assertEqual(sample.forecast_generated_round, 71)
        self.assertEqual(sample.forecast_updated_round, 72)
        self.assertEqual(sample.forecast_age_rounds, 1)
        self.assertEqual(sample.forecast_margin_source, 'conservative_bound')
        self.assertTrue(sample.forecast_conservative_bound)
        self.assertEqual(sample.build_failure_count, 2)
        self.assertEqual(sample.build_cooldown_count, 1)
        self.assertEqual(sample.build_reroute_count, 1)
        self.assertTrue(sample.build_failure_log)
        self.assertTrue(sample.weapon_build_reroute_log)
        self.assertFalse(sample.own_station_alive)
        self.assertEqual(sample.engine_lock_wait_ms, 3.25)
        self.assertTrue(sample.engine_lock_timed_out)
        self.assertEqual(sample.deadline_stage, 'engine_lock')
        self.assertEqual(sample.emergency_fire_ms, 1.25)
        self.assertEqual(sample.emergency_fire_action_count, 2)
        self.assertTrue(sample.emergency_fire_deadline_hit)
        payload = json.loads(logs.output[0].split('turn_telemetry ', 1)[1])
        self.assertEqual(payload['forecast_mode'], 'lightweight')
        self.assertEqual(payload['fallback_reason'], 'deadline_low')
        self.assertEqual(payload['response_action_count'], 2)
        self.assertEqual(payload['pre_validation_action_count'], 3)
        self.assertEqual(payload['phase3_executed_level'], 'lite')
        self.assertTrue(payload['forecast_conservative_bound'])
        self.assertEqual(payload['build_failure_count'], 2)
        self.assertEqual(payload['build_reroute_count'], 1)
        self.assertTrue(payload['weapon_build_reroute_log'])
        self.assertTrue(payload['engine_lock_timed_out'])
        self.assertEqual(payload['deadline_stage'], 'engine_lock')
        self.assertEqual(payload['emergency_fire_action_count'], 2)

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
        self.assertEqual(sample.decision_source, 'phase2')

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
        self.assertEqual(sample.phase3_level, 'lite')
        self.assertEqual(sample.phase3_effective_level, 'lite')
        self.assertEqual(sample.phase3_executed_level, 'lite')
        self.assertEqual(sample.phase3_skip_reason, 'none')
        self.assertGreater(sample.phase3_completed_root_count, 0)
        self.assertEqual(sample.compute_governor_action, 'normal')
        self.assertGreater(sample.governor_root_limit, 0)
        self.assertEqual(sample.governor_scenario_limit, 1)
        self.assertGreater(sample.root_candidate_count, 0)
        self.assertEqual(sample.scenario_count, 1)
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
        self.assertTrue(sample.phase2_5_weapon_log)

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
        self.assertEqual(sample.phase3_executed_level, 'none')
        self.assertEqual(sample.phase3_skip_reason, 'phase3_timeout')


if __name__ == '__main__':
    unittest.main()
