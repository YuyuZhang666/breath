import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.build_recovery import (
    EMPTY_BUILD_RECOVERY_STATE,
    update_build_recovery,
)
from tests.strategy_helpers import observation, unit


class BuildRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Position(8, 22)
        self.build = Decision(
            commands={10010: Action.build('rocket', self.target)}
        )

    def test_repeated_failure_grows_cooldown_and_enables_reroute(self) -> None:
        first = update_build_recovery(
            EMPTY_BUILD_RECOVERY_STATE,
            observation(round_no=2, last_action_results={10010: False}),
            self.build,
        )
        second = update_build_recovery(
            first,
            observation(round_no=3, last_action_results={10010: False}),
            self.build,
        )

        self.assertEqual(first.failures[0].consecutive_failures, 1)
        self.assertEqual(first.failures[0].cooldown_until_round, 2)
        self.assertEqual(second.failures[0].consecutive_failures, 2)
        self.assertEqual(second.failures[0].cooldown_until_round, 4)
        self.assertIn(
            self.target,
            second.reroute_targets('rocket', round_no=3),
        )

    def test_success_clears_matching_failure(self) -> None:
        failed = update_build_recovery(
            EMPTY_BUILD_RECOVERY_STATE,
            observation(round_no=2, last_action_results={10010: False}),
            self.build,
        )

        recovered = update_build_recovery(
            failed,
            observation(
                round_no=3,
                our_units=(
                    unit(10020, self.target.x, self.target.y, 'rocket'),
                ),
                last_action_results={10010: True},
            ),
            self.build,
        )

        self.assertEqual(recovered.failures, ())

    def test_reported_success_without_structure_is_failure(self) -> None:
        recovered = update_build_recovery(
            EMPTY_BUILD_RECOVERY_STATE,
            observation(round_no=3, last_action_results={10010: True}),
            self.build,
        )

        self.assertEqual(recovered.failures[0].consecutive_failures, 1)

    def test_stale_failure_expires(self) -> None:
        failed = update_build_recovery(
            EMPTY_BUILD_RECOVERY_STATE,
            observation(round_no=2, last_action_results={10010: False}),
            self.build,
        )

        expired = update_build_recovery(
            failed,
            observation(round_no=23),
            None,
        )

        self.assertEqual(expired.failures, ())


if __name__ == '__main__':
    unittest.main()
