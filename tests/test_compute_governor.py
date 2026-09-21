import unittest

from future_war_agent.strategy.compute import ComputeGovernor, ComputeGovernorConfig
from future_war_agent.strategy.simulation.config import (
    Phase3Config,
    Phase3Level,
)


class ComputeGovernorTests(unittest.TestCase):
    def test_emergency_micro_budgets_must_fit_inside_reserve(self) -> None:
        with self.assertRaises(ValueError):
            ComputeGovernorConfig(emergency_fire_budget_seconds=0.0)
        with self.assertRaises(ValueError):
            ComputeGovernorConfig(emergency_postprocess_guard_seconds=0.0)
        with self.assertRaises(ValueError):
            ComputeGovernorConfig(
                emergency_fire_budget_seconds=0.45,
                emergency_postprocess_guard_seconds=0.05,
            )

    def setUp(self) -> None:
        self.governor = ComputeGovernor()
        self.phase3 = Phase3Config()

    def observe(
        self,
        *,
        team: str = 'alpha',
        round_no: int,
        day_no: int = 1,
        phase2_5_ms: float = 1.0,
        phase3_ms: float = 0.0,
        roots: int = 0,
        scenarios: int = 0,
        horizon: int = 0,
        watchdog: bool = False,
    ):
        return self.governor.observe_turn(
            team,
            round_no=round_no,
            day_no=day_no,
            phase2_5_ms=phase2_5_ms,
            phase3_ms=phase3_ms,
            roots_evaluated=roots,
            scenarios_per_root=scenarios,
            exact_horizon=horizon,
            watchdog_hit=watchdog,
        )

    def plan(
        self,
        *,
        team: str = 'alpha',
        round_no: int,
        day_no: int = 1,
        level: Phase3Level = Phase3Level.FULL,
        now: float = 10.0,
        remaining: float = 3.0,
    ):
        return self.governor.plan(
            team,
            requested_level=level,
            phase3_config=self.phase3,
            round_no=round_no,
            day_no=day_no,
            now=now,
            request_deadline=now + remaining,
        )

    def test_ewma_records_phase2_5_and_complete_root_cost(self) -> None:
        first = self.observe(
            round_no=70,
            phase2_5_ms=2.0,
            phase3_ms=100.0,
            roots=4,
            scenarios=1,
            horizon=2,
        )
        second = self.observe(
            round_no=71,
            phase2_5_ms=6.0,
            phase3_ms=200.0,
            roots=4,
            scenarios=1,
            horizon=2,
        )

        self.assertEqual(first.ewma_phase2_5_ms, 2.0)
        self.assertEqual(first.ewma_root_rollout_ms, 25.0)
        self.assertEqual(second.ewma_phase2_5_ms, 3.0)
        self.assertEqual(second.ewma_root_rollout_ms, 31.25)

    def test_over_250_ms_halves_roots_only_on_next_round(self) -> None:
        self.observe(
            round_no=70,
            phase3_ms=251.0,
            roots=100,
            scenarios=1,
            horizon=2,
        )

        next_round = self.plan(round_no=71, level=Phase3Level.LITE)
        later_round = self.plan(round_no=72, level=Phase3Level.LITE)

        self.assertEqual(next_round.attempts[0].budget.root_candidates, 2)
        self.assertIn('slow_previous_turn', next_round.reasons)
        self.assertEqual(later_round.attempts[0].budget.root_candidates, 4)

    def test_exact_250_ms_does_not_halve_roots(self) -> None:
        self.observe(
            round_no=70,
            phase3_ms=250.0,
            roots=100,
            scenarios=1,
            horizon=2,
        )

        plan = self.plan(round_no=71, level=Phase3Level.LITE)

        self.assertEqual(plan.attempts[0].budget.root_candidates, 4)
        self.assertNotIn('slow_previous_turn', plan.reasons)

    def test_two_over_300_ms_disable_full_for_current_night(self) -> None:
        for round_no in (70, 71):
            self.observe(
                round_no=round_no,
                phase3_ms=301.0,
                roots=8,
                scenarios=2,
                horizon=4,
            )

        same_night = self.plan(round_no=72, day_no=1)
        next_night = self.plan(round_no=132, day_no=2)

        self.assertIs(same_night.effective_level, Phase3Level.LITE)
        self.assertEqual(
            tuple(item.level for item in same_night.attempts),
            (Phase3Level.LITE,),
        )
        self.assertIs(next_night.effective_level, Phase3Level.FULL)

    def test_exact_300_ms_does_not_disable_full(self) -> None:
        for round_no in (70, 71):
            self.observe(
                round_no=round_no,
                phase3_ms=300.0,
                roots=100,
                scenarios=2,
                horizon=4,
            )

        plan = self.plan(round_no=72, day_no=1)

        self.assertIs(plan.effective_level, Phase3Level.FULL)

    def test_same_round_revision_does_not_count_as_two_slow_turns(self) -> None:
        for _ in range(2):
            self.observe(
                round_no=70,
                phase3_ms=301.0,
                roots=8,
                scenarios=2,
                horizon=4,
            )

        plan = self.plan(round_no=71, day_no=1)

        self.assertIs(plan.effective_level, Phase3Level.FULL)

    def test_watchdog_disables_exactly_ten_following_rounds(self) -> None:
        self.observe(round_no=70, watchdog=True)

        for round_no in (71, 75, 80):
            with self.subTest(round_no=round_no):
                self.assertIs(
                    self.plan(round_no=round_no).effective_level,
                    Phase3Level.NONE,
                )
        self.assertIs(
            self.plan(round_no=81).effective_level,
            Phase3Level.FULL,
        )

    def test_remaining_emergency_reserve_returns_no_attempt(self) -> None:
        plan = self.plan(round_no=70, remaining=0.499)

        self.assertIs(plan.effective_level, Phase3Level.NONE)
        self.assertTrue(plan.emergency_return)
        self.assertEqual(plan.attempts, ())

    def test_none_request_never_creates_a_lite_attempt(self) -> None:
        plan = self.plan(round_no=70, level=Phase3Level.NONE)

        self.assertIs(plan.effective_level, Phase3Level.NONE)
        self.assertEqual(plan.attempts, ())

    def test_horizon_and_scenario_scaling_reduce_full_work(self) -> None:
        self.observe(
            round_no=70,
            phase3_ms=80.0,
            roots=4,
            scenarios=1,
            horizon=2,
        )

        plan = self.plan(round_no=71, level=Phase3Level.FULL)

        self.assertIs(plan.effective_level, Phase3Level.FULL)
        self.assertLess(
            plan.attempts[0].budget.root_candidates,
            self.phase3.max_root_actions,
        )

    def test_unaffordable_lite_turns_phase3_off(self) -> None:
        self.observe(
            round_no=70,
            phase3_ms=1000.0,
            roots=2,
            scenarios=1,
            horizon=2,
        )

        plan = self.plan(round_no=71, level=Phase3Level.LITE)

        self.assertIs(plan.effective_level, Phase3Level.NONE)
        self.assertEqual(plan.attempts, ())
        self.assertIn('compute_exhausted', plan.reasons)

    def test_team_states_are_isolated(self) -> None:
        self.observe(team='alpha', round_no=70, watchdog=True)

        self.assertIs(
            self.plan(team='alpha', round_no=71).effective_level,
            Phase3Level.NONE,
        )
        self.assertIs(
            self.plan(team='bravo', round_no=71).effective_level,
            Phase3Level.FULL,
        )


if __name__ == '__main__':
    unittest.main()
