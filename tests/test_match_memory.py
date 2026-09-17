import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.memory import (
    MatchMemoryStore,
    canonical_position,
)
from tests.strategy_helpers import observation, unit
from tests.test_strategy_engine import search_result


class MatchMemoryTests(unittest.TestCase):
    def test_mirrored_sides_share_canonical_coordinates(self) -> None:
        upper_left = observation(
            width=15,
            height=15,
            our_units=(unit(10, 2, 2, 'station', level=1),),
        )
        lower_right = observation(
            width=15,
            height=15,
            our_units=(unit(10, 12, 12, 'station', level=1),),
        )

        first = canonical_position(upper_left, Position(12, 12))
        second = canonical_position(lower_right, Position(2, 2))

        self.assertEqual(first, Position(2, 2))
        self.assertEqual(second, Position(2, 2))

    def test_missing_station_uses_identity_orientation(self) -> None:
        observed = observation(width=10, height=10)

        self.assertEqual(
            canonical_position(observed, Position(2, 3)),
            Position(2, 3),
        )

    def test_store_reuses_belief_across_side_swap_and_round_gap(self) -> None:
        store = MatchMemoryStore()
        first = observation(
            round_no=1,
            our_units=(unit(10, 2, 2, 'station', level=1),),
            enemy_units=(unit(90, 12, 12, 'worker'),),
            zones=(Zone(Position(10, 10), 'stone'),),
        )
        second = observation(
            round_no=200,
            our_units=(unit(10, 12, 12, 'station', level=1),),
            enemy_units=(unit(90, 2, 2, 'worker'),),
            zones=(Zone(Position(4, 4), 'stone'),),
        )

        store.observe(first)
        memory = store.observe(second)

        self.assertEqual(len(memory.belief.tracks), 1)
        self.assertEqual(memory.belief.tracks[0].position, Position(2, 2))
        self.assertEqual(memory.belief.tracks[0].last_seen_round, 200)
        self.assertEqual(memory.last_round, 200)

    def test_duplicate_observation_is_mutation_free(self) -> None:
        store = MatchMemoryStore()
        observed = observation(
            enemy_units=(unit(90, 4, 4, 'worker'),),
        )

        first = store.observe(observed)
        duplicate = store.observe(observed)

        self.assertIs(first, duplicate)

    def test_wave_summaries_are_bounded(self) -> None:
        store = MatchMemoryStore()
        certificate = search_result().certificate

        for round_no in range(1, 21):
            store.observe(
                observation(round_no=round_no),
                certificate=certificate,
            )

        memory = store.get('team')
        self.assertEqual(len(memory.wave_summaries), 16)
        self.assertEqual(memory.wave_summaries[0].round_no, 5)
        self.assertEqual(memory.wave_summaries[-1].round_no, 20)


if __name__ == '__main__':
    unittest.main()
