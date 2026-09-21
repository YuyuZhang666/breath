import unittest

from future_war_agent.strategy.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityRecord,
    CapabilityStatus,
)


class CapabilityMatrixTests(unittest.TestCase):
    def test_default_matrix_is_complete_unknown_and_closed(self) -> None:
        matrix = CapabilityMatrix()

        self.assertEqual(len(matrix.records), len(Capability))
        self.assertEqual(
            {record.status for record in matrix.records},
            {CapabilityStatus.UNKNOWN},
        )
        self.assertEqual(matrix.status_counts(), (len(Capability), 0, 0))
        self.assertFalse(matrix.supports(Capability.ATTACK_ENEMY_STATION))
        self.assertFalse(
            matrix.supports_all(
                Capability.ATTACK_ENEMY_STATION,
                Capability.CROSS_TEAM_KILL_SCORE,
            )
        )

    def test_manual_config_changes_only_named_capabilities(self) -> None:
        matrix = CapabilityMatrix.from_manual_config(
            {
                Capability.ATTACK_ENEMY_STATION: CapabilityStatus.SUPPORTED,
                'collect_at_night': 'unsupported',
                Capability.STUN_REFRESHES: CapabilityRecord(
                    Capability.STUN_REFRESHES,
                    CapabilityStatus.SUPPORTED,
                    source='manual_test',
                    note='verified offline',
                ),
            }
        )

        self.assertTrue(matrix.supports('attack_enemy_station'))
        self.assertTrue(matrix.supports(Capability.STUN_REFRESHES))
        self.assertEqual(
            matrix.status(Capability.COLLECT_AT_NIGHT),
            CapabilityStatus.UNSUPPORTED,
        )
        self.assertEqual(
            matrix.status(Capability.ATTACK_ENEMY_ROLE),
            CapabilityStatus.UNKNOWN,
        )
        self.assertEqual(matrix.status_counts(), (8, 2, 1))
        self.assertEqual(len(matrix.log_entries()), len(Capability))

    def test_invalid_manual_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapabilityMatrix.from_manual_config({'invented_rule': 'supported'})
        with self.assertRaises(ValueError):
            CapabilityMatrix.from_manual_config(
                {Capability.ATTACK_ENEMY_ROLE: 'maybe'}
            )
        with self.assertRaises(ValueError):
            CapabilityMatrix.from_manual_config(
                {
                    Capability.ATTACK_ENEMY_ROLE: CapabilityRecord(
                        Capability.ATTACK_ENEMY_STATION,
                        CapabilityStatus.SUPPORTED,
                    )
                }
            )


if __name__ == '__main__':
    unittest.main()
