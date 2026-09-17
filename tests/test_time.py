import unittest

from future_war_agent.protocol.time import Phase, TurnTime


class TurnTimeTests(unittest.TestCase):
    def test_round_boundaries(self) -> None:
        cases = {
            1: (1, Phase.DAY, 1, 0),
            70: (1, Phase.DAY, 70, 69),
            71: (1, Phase.NIGHT, 1, 70),
            130: (1, Phase.NIGHT, 60, 129),
            131: (2, Phase.DAY, 1, 0),
        }

        for round_no, expected in cases.items():
            with self.subTest(round_no=round_no):
                turn = TurnTime.from_round(round_no)
                self.assertEqual(
                    (
                        turn.day_no,
                        turn.phase,
                        turn.round_in_phase,
                        turn.offset_in_day,
                    ),
                    expected,
                )

    def test_round_number_must_be_positive_integer(self) -> None:
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    TurnTime.from_round(invalid)


if __name__ == "__main__":
    unittest.main()
