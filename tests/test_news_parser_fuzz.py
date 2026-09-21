import random
import unittest

from future_war_agent.strategy.market import parse_market_signals
from future_war_agent.strategy.treasure import parse_treasure_clues


class NewsParserFuzzTests(unittest.TestCase):
    def test_seeded_unicode_and_control_input_is_bounded_and_safe(self) -> None:
        randomizer = random.Random(20260921)
        alphabet = (
            'abcXYZ0123,，:：;；()[]{}\x00\n'
            '石铁铜矿停工塌方短缺新矿复产增产'
            '祭品物品坐标位置第天置信度'
            '🔥🧭🌙'
        )

        for _ in range(250):
            length = randomizer.randint(0, 2048)
            text = ''.join(randomizer.choice(alphabet) for _ in range(length))

            market = parse_market_signals(text, current_day=1)
            treasure = parse_treasure_clues(text, width=41, height=32)

            self.assertLessEqual(len(market), 16)
            self.assertLessEqual(len(treasure), 16)
            self.assertTrue(
                all(
                    0 <= clue.position.x < 41
                    and 0 <= clue.position.y < 32
                    and clue.items
                    and clue.opening_days
                    for clue in treasure
                )
            )

    def test_oversized_input_fails_closed(self) -> None:
        oversized = 'x' * 32_769

        self.assertEqual(
            parse_market_signals(oversized, current_day=1),
            (),
        )
        self.assertEqual(
            parse_treasure_clues(oversized, width=41, height=32),
            (),
        )


if __name__ == '__main__':
    unittest.main()
