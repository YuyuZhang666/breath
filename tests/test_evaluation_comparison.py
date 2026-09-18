import unittest

from future_war_agent.decision.decision import Decision
from future_war_agent.evaluation.comparison import (
    evaluate_variants,
    rank_aggregates,
)
from future_war_agent.evaluation.loader import load_replay_data
from future_war_agent.evaluation.models import ReplayVariant, VariantAggregate

from tests.test_evaluation_runner import FakeClock, raw_turn


class EmptyPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, observation) -> Decision:
        self.calls += 1
        return Decision()


class ReplayComparisonTests(unittest.TestCase):
    def make_corpus(self):
        return load_replay_data(
            {
                "cases": [
                    {
                        "name": "won",
                        "outcome": "win",
                        "turns": [
                            raw_turn(1, score=4, station_health=100),
                            raw_turn(2, score=9, station_health=80),
                        ],
                    },
                    {
                        "name": "unrated",
                        "outcome": "unknown",
                        "turns": [
                            raw_turn(1, score=7, station_health=0),
                        ],
                    },
                ]
            }
        )

    def test_evaluates_every_pair_with_fresh_planners_and_aggregates(self) -> None:
        planners: list[EmptyPlanner] = []

        def factory() -> EmptyPlanner:
            planner = EmptyPlanner()
            planners.append(planner)
            return planner

        report = evaluate_variants(
            self.make_corpus(),
            (ReplayVariant("baseline", factory),),
            clock_factory=lambda: FakeClock([0, 1, 1, 2]),
        )

        self.assertEqual([planner.calls for planner in planners], [2, 1])
        self.assertEqual([result.case_name for result in report.results], ["won", "unrated"])
        aggregate = report.aggregates[0]
        self.assertEqual(aggregate.variant_name, "baseline")
        self.assertEqual(aggregate.case_count, 2)
        self.assertEqual(aggregate.league_points, 3)
        self.assertEqual(aggregate.surviving_cases, 1)
        self.assertEqual(aggregate.score_gain, 5)
        self.assertEqual(aggregate.answer_submit_count, 0)
        self.assertEqual(aggregate.max_p99_latency_ns, 1)
        self.assertEqual(report.ranking, ("baseline",))

    def test_ranking_applies_every_tie_break_in_documented_order(self) -> None:
        values = (
            VariantAggregate("points", 1, 4, 0, 0, 0, 99),
            VariantAggregate("survival", 1, 3, 2, 0, 0, 99),
            VariantAggregate("score", 1, 3, 1, 8, 0, 99),
            VariantAggregate("fast-z", 1, 3, 1, 7, 0, 10),
            VariantAggregate("fast-a", 1, 3, 1, 7, 0, 10),
            VariantAggregate("slow", 1, 3, 1, 7, 0, 20),
        )

        ranked = rank_aggregates(reversed(values))

        self.assertEqual(
            tuple(item.variant_name for item in ranked),
            ("points", "survival", "score", "fast-a", "fast-z", "slow"),
        )

    def test_variant_iteration_order_does_not_change_report_order(self) -> None:
        variants = (
            ReplayVariant("zeta", EmptyPlanner),
            ReplayVariant("alpha", EmptyPlanner),
        )

        clock_factory = lambda: FakeClock([0, 1, 1, 2])
        first = evaluate_variants(
            self.make_corpus(), variants, clock_factory=clock_factory
        )
        second = evaluate_variants(
            self.make_corpus(),
            tuple(reversed(variants)),
            clock_factory=clock_factory,
        )

        self.assertEqual(first.ranking, ("alpha", "zeta"))
        self.assertEqual(first.ranking, second.ranking)
        self.assertEqual(
            tuple((item.variant_name, item.case_name) for item in first.results),
            tuple((item.variant_name, item.case_name) for item in second.results),
        )

    def test_rejects_duplicate_variant_names(self) -> None:
        variants = (
            ReplayVariant("same", EmptyPlanner),
            ReplayVariant("same", EmptyPlanner),
        )

        with self.assertRaisesRegex(ValueError, "duplicate variant name"):
            evaluate_variants(self.make_corpus(), variants)


if __name__ == "__main__":
    unittest.main()
