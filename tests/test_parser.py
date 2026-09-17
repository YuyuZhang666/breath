import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from future_war_agent.protocol.parser import ProtocolError, parse_observation


FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ObservationParserTests(unittest.TestCase):
    def test_parses_representative_request(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        observed = parse_observation(raw)

        self.assertEqual(observed.time.round_no, 85)
        self.assertEqual(observed.width, 41)
        self.assertEqual(observed.our.gold, 20)
        self.assertEqual(observed.our.units[0].unit_id, 10010)
        self.assertIsNone(observed.our.tasks[0].timeout_rounds)
        self.assertIsNone(observed.robots[0].target_team)
        self.assertIs(observed.last_action_results[10010], True)

    def test_optional_sections_default_to_empty_values(self) -> None:
        observed = parse_observation(
            {
                "roundNo": 1,
                "mapInfo": {"width": 41, "height": 32},
                "teamOur": {},
            }
        )

        self.assertEqual(observed.zones, ())
        self.assertEqual(observed.enemy.units, ())
        self.assertEqual(observed.robots, ())
        self.assertEqual(dict(observed.last_action_results), {})
        self.assertEqual(observed.world_news.official_news, "")

    def test_required_top_level_data_is_enforced(self) -> None:
        cases = ({}, {"roundNo": 1}, {"roundNo": 1, "mapInfo": {}})
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ProtocolError):
                    parse_observation(raw)

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_observation(
                {
                    "roundNo": True,
                    "mapInfo": {"width": 41, "height": 32},
                    "teamOur": {},
                }
            )

    def test_models_and_collections_are_immutable(self) -> None:
        observed = parse_observation(
            {
                "roundNo": 1,
                "mapInfo": {"width": 41, "height": 32},
                "teamOur": {},
            }
        )

        with self.assertRaises(FrozenInstanceError):
            observed.width = 99
        with self.assertRaises(TypeError):
            observed.last_action_results[10010] = True


if __name__ == "__main__":
    unittest.main()
