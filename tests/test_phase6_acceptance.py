import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityStatus,
)
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.memory import MatchMemoryStore
from future_war_agent.strategy.policy import StrategyProfile
from future_war_agent.telemetry import TelemetryRecorder
from tests.strategy_helpers import observation, unit
from tests.test_strategy_engine import load_observation, DAY_FIXTURE, NIGHT_FIXTURE


class BrokenMemoryStore:
    def observe(self, *args, **kwargs):
        raise RuntimeError('memory failure')


class Phase6AcceptanceTests(unittest.TestCase):
    def test_manual_capabilities_are_logged_without_changing_strategy(self) -> None:
        matrix = CapabilityMatrix.from_manual_config(
            {Capability.ATTACK_ENEMY_STATION: CapabilityStatus.SUPPORTED}
        )
        telemetry = TelemetryRecorder()
        configured = StrategyEngine(
            memory_store=MatchMemoryStore(capability_matrix=matrix),
            telemetry=telemetry,
        )
        baseline = StrategyEngine()
        observed = observation(
            our_units=(
                unit(10, 5, 5, 'station', health=1000, level=1),
                unit(11, 3, 3, 'gatling', level=1),
                unit(12, 4, 3, 'railgun', level=1),
                unit(13, 5, 3, 'rocket', level=1),
                unit(14, 1, 1, 'pioneer'),
            ),
            enemy_units=(unit(90, 12, 12, 'station', health=100, level=1),),
            gold=200,
        )

        expected = baseline.plan(observed)
        actual = configured.plan(observed)
        sample = telemetry.snapshot()[-1]

        self.assertEqual(actual, expected)
        self.assertIsNot(configured.current_profile('team'), StrategyProfile.PRESSURE)
        self.assertEqual(sample.capability_supported_count, 1)
        self.assertEqual(sample.capability_unknown_count, len(Capability) - 1)
        self.assertEqual(len(sample.capability_log), len(Capability))
        self.assertTrue(sample.opponent_structure_log)
        self.assertTrue(sample.opponent_evidence_log)

    def test_observations_never_auto_promote_unknown_capability(self) -> None:
        store = MatchMemoryStore()
        store.observe(
            observation(
                enemy_units=(unit(90, 4, 4, 'station', health=100),),
            )
        )

        matrix = store.get('team').capability_matrix

        self.assertEqual(matrix.status_counts(), (len(Capability), 0, 0))

    def test_memory_survives_session_discontinuity_and_side_swap(self) -> None:
        engine = StrategyEngine()
        first = observation(
            round_no=1,
            our_units=(unit(10, 2, 2, 'station', level=1),),
            enemy_units=(unit(90, 12, 12, 'worker'),),
        )
        second = observation(
            round_no=200,
            our_units=(unit(10, 12, 12, 'station', level=1),),
            enemy_units=(unit(90, 2, 2, 'worker'),),
        )

        engine.plan(first)
        engine.plan(second)

        memory = engine._memory.get('team')
        self.assertEqual(memory.belief.tracks[0].position, Position(2, 2))
        self.assertEqual(memory.belief.tracks[0].last_seen_round, 200)

    def test_engine_memory_survives_true_round_rollback(self) -> None:
        engine = StrategyEngine()
        engine.plan(
            observation(
                round_no=100,
                our_units=(unit(10, 2, 2, 'station', level=1),),
                enemy_units=(unit(90, 12, 12, 'worker'),),
            )
        )

        engine.plan(
            observation(
                round_no=1,
                our_units=(unit(10, 12, 12, 'station', level=1),),
            )
        )

        memory = engine._memory.get('team')
        self.assertEqual(memory.epoch, 1)
        self.assertEqual(memory.last_round, 1)
        self.assertEqual(memory.belief.tracks[0].confidence, 90)

    def test_fresh_phase3_certificate_is_recorded_once(self) -> None:
        engine = StrategyEngine()
        day = load_observation(DAY_FIXTURE)
        night = load_observation(NIGHT_FIXTURE)

        engine.plan(day)
        engine.plan(night)
        memory = engine._memory.get(day.our.team_id)
        first_count = len(memory.wave_summaries)
        engine.plan(night)
        duplicate = engine._memory.get(day.our.team_id)

        self.assertEqual(first_count, 1)
        self.assertIs(memory, duplicate)

    def test_memory_failure_does_not_change_planning_result(self) -> None:
        base = StrategyEngine()
        broken = StrategyEngine(memory_store=BrokenMemoryStore())
        observed = observation(
            our_units=(unit(10, 5, 5, 'station', level=1),),
        )

        expected = base.plan(observed)
        actual = broken.plan(observed)

        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
