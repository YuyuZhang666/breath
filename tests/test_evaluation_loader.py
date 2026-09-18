import json
import re
import tempfile
import unittest
from pathlib import Path

from future_war_agent.evaluation.errors import ReplayFormatError
from future_war_agent.evaluation.loader import (
    MAX_CASES,
    MAX_TURNS_PER_CASE,
    load_replay_data,
    load_replay_file,
)
from future_war_agent.evaluation.models import MatchOutcome


def replay_case(
    name: str = "match-a",
    *,
    outcome: str = "win",
    turns: list[object] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "outcome": outcome,
        "turns": [{"roundNo": 1}] if turns is None else turns,
    }


class ReplayLoaderTests(unittest.TestCase):
    def test_loads_valid_cases_and_freezes_nested_turn_data(self) -> None:
        raw = {
            "cases": [
                replay_case(
                    turns=[{"roundNo": 1, "nested": {"items": [1, 2]}}]
                )
            ]
        }

        corpus = load_replay_data(raw)

        self.assertEqual(corpus.cases[0].name, "match-a")
        self.assertIs(corpus.cases[0].outcome, MatchOutcome.WIN)
        self.assertEqual(corpus.cases[0].outcome.league_points, 3)
        self.assertEqual(corpus.cases[0].turns[0]["nested"]["items"], (1, 2))
        with self.assertRaises(TypeError):
            corpus.cases[0].turns[0]["roundNo"] = 2

    def test_rejects_duplicate_case_names_after_trimming(self) -> None:
        raw = {"cases": [replay_case("match-a"), replay_case(" match-a ")]}

        with self.assertRaisesRegex(ReplayFormatError, "duplicate.*match-a"):
            load_replay_data(raw)

    def test_rejects_invalid_case_contracts_with_context(self) -> None:
        cases = {
            "blank name": replay_case("  "),
            "unknown outcome": replay_case(outcome="victory"),
            "empty turns": replay_case(turns=[]),
            "non-object turn": replay_case(turns=["bad"]),
        }
        expected = {
            "blank name": "cases[0].name",
            "unknown outcome": "cases[0].outcome",
            "empty turns": "cases[0].turns",
            "non-object turn": "cases[0].turns[0]",
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ReplayFormatError, re.escape(expected[label])
                ):
                    load_replay_data({"cases": [value]})

    def test_rejects_empty_and_oversized_corpora_before_evaluation(self) -> None:
        invalid = (
            {"cases": []},
            {"cases": [replay_case(str(index)) for index in range(MAX_CASES + 1)]},
            {
                "cases": [
                    replay_case(
                        turns=[{"roundNo": index + 1} for index in range(MAX_TURNS_PER_CASE + 1)]
                    )
                ]
            },
        )
        for raw in invalid:
            with self.subTest(size=len(raw["cases"])):
                with self.assertRaises(ReplayFormatError):
                    load_replay_data(raw)

    def test_file_errors_are_reported_as_replay_format_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{not json", encoding="utf-8")

            with self.assertRaisesRegex(
                ReplayFormatError, "broken.json.*valid JSON"
            ):
                load_replay_file(path)

            path.write_text(json.dumps({"cases": []}), encoding="utf-8")
            with self.assertRaisesRegex(ReplayFormatError, "cases"):
                load_replay_file(path)

    def test_rejects_non_standard_non_finite_json_numbers(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "non-finite.json"
                    path.write_text(
                        '{"cases":[{"name":"match","outcome":"win",'
                        f'"turns":[{{"value":{constant}}}]}}]}}',
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        ReplayFormatError, "non-finite.json.*non-standard"
                    ):
                        load_replay_file(path)


if __name__ == "__main__":
    unittest.main()
