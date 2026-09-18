import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.evaluation.errors import ReplayEvaluationError
from future_war_agent.evaluation.loader import load_replay_data
from future_war_agent.evaluation.models import ReplayVariant
from future_war_agent.evaluation.runner import run_case
from future_war_agent.protocol.models import Position
from future_war_agent.protocol.parser import ProtocolError


def raw_turn(round_no: int, *, score: int, station_health: int) -> dict[str, object]:
    return {
        "roundNo": round_no,
        "mapInfo": {"width": 12, "height": 12, "zones": []},
        "teamOur": {
            "type": "challenger",
            "teamId": "replay-team",
            "totalScore": score,
            "roles": [
                {
                    "id": 100,
                    "pos": {"x": 5, "y": 5},
                    "roleType": "station",
                    "health": station_health,
                },
                {
                    "id": 1,
                    "pos": {"x": 4, "y": 5},
                    "roleType": "pioneer",
                    "health": 100,
                },
            ],
        },
    }


class ScriptedPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.profile = "economy"

    def plan(self, observation) -> Decision:
        self.calls += 1
        if self.calls == 1:
            return Decision(
                commands={
                    1: Action.accept_task(),
                    999: Action.move(Position(0, 0)),
                },
                prompt="solve this",
            )
        if self.calls == 2:
            return Decision(
                commands={1: Action.submit_answer("secret")},
                execute_command="status",
            )
        self.profile = "score"
        if self.calls == 3:
            return Decision(commands={1: Action.move(Position(4, 6))})
        return Decision()


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class ReplayRunnerTests(unittest.TestCase):
    def make_case(self):
        turns = [
            raw_turn(1, score=10, station_health=100),
            raw_turn(2, score=12, station_health=80),
            raw_turn(3, score=15, station_health=70),
            raw_turn(4, score=18, station_health=75),
        ]
        return load_replay_data(
            {"cases": [{"name": "match-a", "outcome": "win", "turns": turns}]}
        ).cases[0]

    def test_runs_production_boundaries_and_computes_metrics(self) -> None:
        variant = ReplayVariant(
            "scripted",
            ScriptedPlanner,
            profile_reader=lambda planner, observation: planner.profile,
        )
        clock = FakeClock([0, 10, 10, 30, 30, 60, 60, 100])

        result = run_case(variant, self.make_case(), clock=clock)

        self.assertEqual(result.elapsed_ns, (10, 20, 30, 40))
        self.assertEqual(len(result.response_digests), 4)
        self.assertTrue(all(len(value) == 64 for value in result.response_digests))
        self.assertEqual(result.profile_labels, ("economy", "economy", "score", "score"))
        self.assertEqual(result.metrics.league_points, 3)
        self.assertEqual(result.metrics.initial_score, 10)
        self.assertEqual(result.metrics.final_score, 18)
        self.assertEqual(result.metrics.score_gain, 8)
        self.assertTrue(result.metrics.final_station_survived)
        self.assertEqual(result.metrics.minimum_station_health, 70)
        self.assertEqual(result.metrics.command_count, 3)
        self.assertEqual(result.metrics.prompt_count, 1)
        self.assertEqual(result.metrics.execute_command_count, 1)
        self.assertEqual(result.metrics.task_accept_count, 1)
        self.assertEqual(result.metrics.answer_submit_count, 1)
        self.assertEqual(
            result.metrics.action_kind_counts,
            (("acceptTask", 1), ("move", 1), ("submitAnswer", 1)),
        )
        self.assertEqual(result.metrics.profile_switch_count, 1)
        self.assertEqual(result.metrics.median_latency_ns, 25)
        self.assertEqual(result.metrics.p99_latency_ns, 40)

    def test_same_replay_and_planner_have_identical_digests_and_fresh_state(self) -> None:
        planners: list[ScriptedPlanner] = []

        def factory() -> ScriptedPlanner:
            planner = ScriptedPlanner()
            planners.append(planner)
            return planner

        variant = ReplayVariant("scripted", factory)
        first = run_case(variant, self.make_case(), clock=FakeClock(list(range(8))))
        second = run_case(variant, self.make_case(), clock=FakeClock(list(range(8))))

        self.assertEqual(first.response_digests, second.response_digests)
        self.assertEqual(len(planners), 2)
        self.assertEqual([planner.calls for planner in planners], [4, 4])

    def test_wraps_turn_failures_with_variant_case_and_one_based_turn(self) -> None:
        case = load_replay_data(
            {
                "cases": [
                    {
                        "name": "broken-match",
                        "outcome": "unknown",
                        "turns": [{"roundNo": 1}],
                    }
                ]
            }
        ).cases[0]

        with self.assertRaisesRegex(
            ReplayEvaluationError,
            "variant 'scripted'.*case 'broken-match'.*turn 1",
        ) as raised:
            run_case(
                ReplayVariant("scripted", ScriptedPlanner),
                case,
                clock=FakeClock([0]),
            )

        self.assertIsInstance(raised.exception.__cause__, ProtocolError)

    def test_wraps_factory_failure_with_variant_and_case_context(self) -> None:
        def broken_factory():
            raise RuntimeError("factory failed")

        with self.assertRaisesRegex(
            ReplayEvaluationError,
            "variant 'broken'.*case 'match-a'.*planner factory",
        ) as raised:
            run_case(ReplayVariant("broken", broken_factory), self.make_case())

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_wraps_post_validation_metric_failure_with_turn_context(self) -> None:
        class InvalidDecisionPlanner:
            def plan(self, observation):
                return Decision(prompt=None)

        with self.assertRaisesRegex(
            ReplayEvaluationError,
            "variant 'invalid'.*case 'match-a'.*turn 1",
        ) as raised:
            run_case(
                ReplayVariant("invalid", InvalidDecisionPlanner),
                self.make_case(),
            )

        self.assertIsInstance(raised.exception.__cause__, AttributeError)


if __name__ == "__main__":
    unittest.main()
