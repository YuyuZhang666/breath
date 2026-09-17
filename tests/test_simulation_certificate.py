import unittest
from dataclasses import replace
from fractions import Fraction

from future_war_agent.strategy.simulation.certificate import (
    ScenarioOutcome,
    WaveClassification,
    build_certificate,
    score_rank_key,
    survival_rank_key,
)
from future_war_agent.strategy.simulation.objective import NightObjective


class SimulationCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.objective = NightObjective()
        self.safe_outcomes = tuple(
            ScenarioOutcome(
                weight=Fraction(1, 4),
                station_health=health,
                controller_losses=0,
                key_weapon_losses=0,
                minimum_role_health=50,
                surviving_asset_health=200,
                owned_kill_score=score,
                remaining_threat=threat,
                remaining_one_turn_damage=10,
                ended_with_night=False,
            )
            for health, score, threat in (
                (100, 1, 4),
                (80, 2, 3),
                (60, 3, 2),
                (40, 4, 1),
            )
        )

    def test_certificate_name_and_classification_are_wave_scoped(self) -> None:
        certificate = build_certificate(self.safe_outcomes, self.objective)

        self.assertIs(certificate.classification, WaveClassification.WAVE_SAFE)
        self.assertTrue(certificate.secured)

    def test_exact_weighted_metrics_and_lower_tail(self) -> None:
        certificate = build_certificate(self.safe_outcomes, self.objective)

        self.assertEqual(certificate.station_survival_probability, Fraction(1))
        self.assertEqual(certificate.expected_station_health, Fraction(70))
        self.assertEqual(certificate.p10_station_health, 40)
        self.assertEqual(certificate.worst_station_health, 40)
        self.assertEqual(certificate.worst_controller_losses, 0)
        self.assertEqual(certificate.worst_key_weapon_losses, 0)
        self.assertEqual(certificate.worst_minimum_role_health, 50)
        self.assertEqual(certificate.expected_owned_kill_score, Fraction(5, 2))
        self.assertEqual(certificate.expected_remaining_threat, Fraction(5, 2))

    def test_weighted_p10_accumulates_exact_fraction_mass(self) -> None:
        outcomes = (
            replace(self.safe_outcomes[0], weight=Fraction(1, 20), station_health=5),
            replace(self.safe_outcomes[1], weight=Fraction(1, 20), station_health=10),
            replace(self.safe_outcomes[2], weight=Fraction(2, 5), station_health=50),
            replace(self.safe_outcomes[3], weight=Fraction(1, 2), station_health=100),
        )

        certificate = build_certificate(outcomes, self.objective)

        self.assertEqual(certificate.p10_station_health, 10)

    def test_score_band_is_unavailable_when_one_controller_dies(self) -> None:
        outcomes = (
            replace(self.safe_outcomes[0], controller_losses=1),
            *self.safe_outcomes[1:],
        )

        certificate = build_certificate(outcomes, self.objective)

        self.assertFalse(certificate.secured)
        self.assertIs(
            certificate.classification,
            WaveClassification.WAVE_MARGINAL,
        )

    def test_station_loss_classifies_wave_unsafe(self) -> None:
        outcomes = (
            replace(self.safe_outcomes[0], station_health=0),
            *self.safe_outcomes[1:],
        )

        certificate = build_certificate(outcomes, self.objective)

        self.assertFalse(certificate.secured)
        self.assertIs(
            certificate.classification,
            WaveClassification.WAVE_UNSAFE,
        )
        self.assertEqual(certificate.station_survival_probability, Fraction(3, 4))

    def test_night_end_does_not_require_post_horizon_buffer(self) -> None:
        outcomes = tuple(
            replace(
                outcome,
                station_health=1,
                remaining_one_turn_damage=999,
                ended_with_night=True,
            )
            for outcome in self.safe_outcomes
        )

        certificate = build_certificate(outcomes, self.objective)

        self.assertTrue(certificate.secured)

    def test_invalid_weight_sets_are_rejected(self) -> None:
        cases = (
            (),
            (replace(self.safe_outcomes[0], weight=Fraction(0)),),
            self.safe_outcomes[:-1],
        )
        for outcomes in cases:
            with self.subTest(outcomes=outcomes):
                with self.assertRaises(ValueError):
                    build_certificate(outcomes, self.objective)

    def test_survival_rank_beats_kill_score_until_secured(self) -> None:
        safer = build_certificate(
            (
                replace(self.safe_outcomes[0], station_health=20, owned_kill_score=0),
                replace(self.safe_outcomes[1], station_health=20, owned_kill_score=0),
                replace(self.safe_outcomes[2], station_health=0, owned_kill_score=0),
                replace(self.safe_outcomes[3], station_health=0, owned_kill_score=0),
            ),
            self.objective,
        )
        greedier = build_certificate(
            tuple(
                replace(outcome, station_health=0, owned_kill_score=99)
                for outcome in self.safe_outcomes
            ),
            self.objective,
        )

        self.assertLess(
            survival_rank_key(safer, ("safe",)),
            survival_rank_key(greedier, ("greedy",)),
        )

    def test_score_rank_prefers_kills_only_between_secured_roots(self) -> None:
        low_score = build_certificate(self.safe_outcomes, self.objective)
        high_score = build_certificate(
            tuple(
                replace(outcome, owned_kill_score=outcome.owned_kill_score + 5)
                for outcome in self.safe_outcomes
            ),
            self.objective,
        )

        self.assertTrue(low_score.secured)
        self.assertTrue(high_score.secured)
        self.assertLess(
            score_rank_key(high_score, ("high",)),
            score_rank_key(low_score, ("low",)),
        )


if __name__ == "__main__":
    unittest.main()
