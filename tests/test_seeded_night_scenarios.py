import unittest

from future_war_agent.decision.validator import validate_decision
from future_war_agent.strategy.planner import plan_turn
from tests.seeded_night_scenarios import seeded_night_observation


class SeededNightScenarioTests(unittest.TestCase):
    def test_generated_waves_are_deterministic_and_emit_legal_plans(
        self,
    ) -> None:
        for seed in range(24):
            with self.subTest(seed=seed):
                observed = seeded_night_observation(seed)
                first = plan_turn(observed)
                second = plan_turn(observed)

                self.assertEqual(first, second)
                self.assertEqual(validate_decision(observed, first), first)
                self.assertTrue(first.commands)


if __name__ == '__main__':
    unittest.main()
