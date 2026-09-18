import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from future_war_agent.evaluation.comparison import evaluate_variants
from future_war_agent.evaluation.loader import load_replay_data
from future_war_agent.evaluation.models import ReplayVariant
from future_war_agent.strategy.engine import StrategyEngine


FIXTURES = Path(__file__).parent / "fixtures"
DAY_FIXTURE = FIXTURES / "phase3_day_request.json"
NIGHT_FIXTURE = FIXTURES / "phase3_night_request.json"


def production_profile(planner, observation):
    return planner.current_profile(observation.our.team_id)


class Phase7AAcceptanceTests(unittest.TestCase):
    def replay_data(self) -> dict[str, object]:
        day = json.loads(DAY_FIXTURE.read_text(encoding="utf-8"))
        night = json.loads(NIGHT_FIXTURE.read_text(encoding="utf-8"))
        return {
            "cases": [
                {
                    "name": "phase3-day-night-duplicate",
                    "outcome": "unknown",
                    "turns": [day, night, night],
                }
            ]
        }

    def test_real_production_replay_is_deterministic_and_duplicate_safe(self) -> None:
        corpus = load_replay_data(self.replay_data())
        variant = ReplayVariant(
            "production",
            StrategyEngine,
            profile_reader=production_profile,
        )

        first = evaluate_variants(corpus, (variant,))
        second = evaluate_variants(corpus, (variant,))

        first_result = first.results[0]
        second_result = second.results[0]
        self.assertEqual(first_result.response_digests, second_result.response_digests)
        self.assertEqual(first_result.response_digests[1], first_result.response_digests[2])
        self.assertEqual(len(first_result.profile_labels), 3)
        self.assertTrue(all(first_result.profile_labels))
        self.assertEqual(first_result.metrics.league_points, 0)
        self.assertTrue(first_result.metrics.final_station_survived)

    def test_python_module_cli_accepts_real_production_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "production-replay.json"
            replay_path.write_text(
                json.dumps(self.replay_data()),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "future_war_agent.evaluation",
                    str(replay_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["ranking"], ["production"])
        self.assertEqual(payload["results"][0]["case_name"], "phase3-day-night-duplicate")


if __name__ == "__main__":
    unittest.main()
