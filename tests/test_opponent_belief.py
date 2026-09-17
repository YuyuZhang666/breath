import unittest

from future_war_agent.strategy.belief import (
    BeliefConfig,
    OpponentBelief,
    update_opponent_belief,
)
from tests.strategy_helpers import observation, unit


class OpponentBeliefTests(unittest.TestCase):
    def test_visible_track_decays_and_contradiction_replaces_facts(self) -> None:
        first = observation(
            round_no=1,
            enemy_units=(unit(90, 8, 8, 'worker', health=100),),
        )
        belief = update_opponent_belief(OpponentBelief(), first)

        self.assertEqual(belief.tracks[0].confidence, 100)
        self.assertEqual(belief.tracks[0].role_type, 'worker')

        decayed = update_opponent_belief(
            belief,
            observation(round_no=2),
        )
        self.assertEqual(decayed.tracks[0].confidence, 90)

        decayed_again = update_opponent_belief(
            decayed,
            observation(round_no=3),
        )
        self.assertEqual(decayed_again.tracks[0].confidence, 80)

        contradicted = update_opponent_belief(
            decayed_again,
            observation(
                round_no=4,
                enemy_units=(unit(90, 4, 4, 'pioneer', health=50),),
            ),
        )
        track = contradicted.tracks[0]
        self.assertEqual(track.role_type, 'pioneer')
        self.assertEqual((track.position.x, track.position.y), (4, 4))
        self.assertEqual(track.health, 50)
        self.assertEqual(track.confidence, 100)

    def test_zero_confidence_tracks_are_removed(self) -> None:
        config = BeliefConfig(decay_per_round=50)
        first = update_opponent_belief(
            OpponentBelief(),
            observation(
                round_no=1,
                enemy_units=(unit(90, 4, 4, 'unknown-role'),),
            ),
            config=config,
        )

        expired = update_opponent_belief(
            first,
            observation(round_no=3),
            config=config,
        )

        self.assertEqual(expired.tracks, ())

    def test_track_count_is_bounded_and_stably_ordered(self) -> None:
        config = BeliefConfig(max_tracks=64)
        observed = observation(
            enemy_units=tuple(
                unit(1000 + index, index % 15, index // 15, 'worker')
                for index in range(70)
            ),
        )

        belief = update_opponent_belief(
            OpponentBelief(),
            observed,
            config=config,
        )

        self.assertEqual(len(belief.tracks), 64)
        self.assertEqual(
            [track.unit_id for track in belief.tracks],
            list(range(1000, 1064)),
        )


if __name__ == '__main__':
    unittest.main()
