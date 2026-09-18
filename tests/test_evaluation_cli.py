import io
import json
import tempfile
import unittest
from pathlib import Path

from future_war_agent.decision.decision import Decision
from future_war_agent.evaluation.__main__ import main
from future_war_agent.evaluation.comparison import evaluate_variants
from future_war_agent.evaluation.loader import load_replay_data
from future_war_agent.evaluation.models import ReplayVariant
from future_war_agent.evaluation.report import report_to_data, report_to_json
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.policy import StrategyProfile

from tests.test_evaluation_runner import raw_turn


class SecretPlanner:
    def plan(self, observation) -> Decision:
        return Decision(prompt="private prompt", execute_command="private command")


class EvaluationCliTests(unittest.TestCase):
    def replay_data(self) -> dict[str, object]:
        return {
            "cases": [
                {
                    "name": "cli-match",
                    "outcome": "draw",
                    "turns": [raw_turn(1, score=5, station_health=100)],
                }
            ]
        }

    def test_strategy_engine_exposes_only_current_profile(self) -> None:
        engine = StrategyEngine()
        observation = parse_observation(raw_turn(1, score=5, station_health=100))

        self.assertIsNone(engine.current_profile("replay-team"))
        engine.plan(observation)

        self.assertIs(engine.current_profile("replay-team"), StrategyProfile.ECONOMY)
        self.assertIsNone(engine.current_profile("missing-team"))

    def test_report_is_plain_stable_json_without_raw_response_content(self) -> None:
        corpus = load_replay_data(self.replay_data())
        report = evaluate_variants(
            corpus,
            (ReplayVariant("secret", SecretPlanner),),
        )

        data = report_to_data(report)
        encoded = report_to_json(report)

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["ranking"], ["secret"])
        self.assertEqual(json.loads(encoded), data)
        self.assertEqual(encoded, report_to_json(report))
        self.assertNotIn("private prompt", encoded)
        self.assertNotIn("private command", encoded)
        self.assertNotIn("roleCommandMap", encoded)

    def test_cli_runs_production_variant_and_writes_only_json_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(self.replay_data()), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = main([str(path)], stdout=stdout, stderr=stderr)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["ranking"], ["production"])
        self.assertEqual(payload["results"][0]["outcome"], "draw")

    def test_cli_reports_invalid_replay_on_stderr_with_nonzero_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("not-json", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = main([str(path)], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("broken.json", stderr.getvalue())
        self.assertIn("valid JSON", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
