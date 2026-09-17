import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.engine import StrategyEngine
from tests.strategy_helpers import observation, unit
from tests.test_strategy_engine import load_observation, DAY_FIXTURE, NIGHT_FIXTURE


class BrokenMemoryStore:
    def observe(self, *args, **kwargs):
        raise RuntimeError('memory failure')


class Phase6AcceptanceTests(unittest.TestCase):
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
