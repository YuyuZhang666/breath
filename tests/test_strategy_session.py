import unittest
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position, Zone
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.session import (
    SessionContinuity,
    SessionStore,
    StrategySession,
    classify_continuity,
    observation_fingerprint,
    static_signature,
)
from future_war_agent.strategy.simulation.candidates import SimJointAction
from future_war_agent.strategy.simulation.certificate import (
    ScenarioOutcome,
    build_certificate,
)
from future_war_agent.strategy.simulation.objective import NightObjective
from tests.strategy_helpers import observation, unit


WEIGHTS = (Fraction(1, 4),) * 4


class StrategySessionTests(unittest.TestCase):
    def test_canonical_fingerprint_ignores_mapping_order(self) -> None:
        first = replace(
            self._observation(),
            last_action_results={2: False, 1: True},
        )
        second = replace(
            self._observation(),
            last_action_results={1: True, 2: False},
        )

        self.assertEqual(
            observation_fingerprint(first),
            observation_fingerprint(second),
        )

    def test_fingerprint_changes_on_planning_relevant_value(self) -> None:
        observed = self._observation()
        richer = replace(observed, our=replace(observed.our, gold=observed.our.gold + 1))

        self.assertNotEqual(
            observation_fingerprint(observed),
            observation_fingerprint(richer),
        )

    def test_static_signature_tracks_map_and_station_identity_not_health(self) -> None:
        observed = self._observation()
        damaged_station = replace(
            observed,
            our=replace(
                observed.our,
                units=tuple(
                    replace(value, health=value.health - 1)
                    if value.role_type == "station"
                    else value
                    for value in observed.our.units
                ),
            ),
        )
        moved_zone = replace(
            observed,
            zones=(Zone(Position(2, 2), "mine"),),
        )
        changed_station = replace(
            observed,
            our=replace(
                observed.our,
                units=tuple(
                    replace(value, unit_id=999)
                    if value.role_type == "station"
                    else value
                    for value in observed.our.units
                ),
            ),
        )

        self.assertEqual(static_signature(observed), static_signature(damaged_station))
        self.assertNotEqual(static_signature(observed), static_signature(moved_zone))
        self.assertNotEqual(static_signature(observed), static_signature(changed_station))

    def test_duplicate_returns_same_cached_decision_object(self) -> None:
        store = SessionStore()
        session = self._session(self._observation(), team_id="alpha")
        store.put(session)

        cached = store.get("alpha")

        self.assertIs(cached.decision, session.decision)
        self.assertIs(
            classify_continuity(cached, session.observation),
            SessionContinuity.DUPLICATE,
        )

    def test_revision_consecutive_and_discontinuity_are_distinct(self) -> None:
        previous = self._session(self._observation(round_no=71), team_id="alpha")
        revision = replace(
            previous.observation,
            our=replace(previous.observation.our, gold=99),
        )
        consecutive = replace(previous.observation, time=TurnTime.from_round(72))
        gap = replace(previous.observation, time=TurnTime.from_round(73))
        rollback = replace(previous.observation, time=TurnTime.from_round(70))
        signature_change = replace(previous.observation, width=16)

        self.assertIs(
            classify_continuity(previous, revision),
            SessionContinuity.REVISION,
        )
        self.assertIs(
            classify_continuity(previous, consecutive),
            SessionContinuity.CONSECUTIVE,
        )
        for observed in (gap, rollback, signature_change):
            with self.subTest(observed=observed):
                self.assertIs(
                    classify_continuity(previous, observed),
                    SessionContinuity.DISCONTINUITY,
                )

    def test_different_team_ids_are_isolated_and_blank_ids_are_ignored(self) -> None:
        store = SessionStore()
        alpha = self._session(self._observation(), team_id="alpha")
        beta = self._session(self._observation(), team_id="beta")
        blank = self._session(self._observation(), team_id="   ")

        store.put(alpha)
        store.put(beta)
        store.put(blank)

        self.assertIs(store.get("alpha"), alpha)
        self.assertIs(store.get("beta"), beta)
        self.assertIsNone(store.get(""))
        self.assertIsNone(store.get("   "))

    def test_session_is_frozen_and_retains_internal_certificate(self) -> None:
        outcome = ScenarioOutcome(
            weight=Fraction(1),
            station_health=100,
            surviving_controlled_role_count=1,
            surviving_controller_count=1,
            controller_losses=0,
            surviving_key_weapon_count=1,
            key_weapon_losses=0,
            wall_losses=0,
            weapon_losses=0,
            minimum_controlled_role_health=100,
            surviving_wall_non_key_weapon_value=50,
            owned_kill_score=1,
            remaining_threat=0,
            remaining_one_turn_damage=0,
            ended_with_night=True,
        )
        certificate = build_certificate((outcome,), NightObjective())
        session = replace(
            self._session(self._observation(), team_id="alpha"),
            simulation_action=SimJointAction(),
            certificate=certificate,
        )

        self.assertIs(session.certificate, certificate)
        with self.assertRaises(FrozenInstanceError):
            session.last_round = 99

    @staticmethod
    def _observation(*, round_no: int = 71):
        return observation(
            round_no=round_no,
            zones=(Zone(Position(1, 1), "mine"),),
            our_units=(
                unit(1, 2, 2, "worker"),
                unit(2, 5, 5, "station", health=500),
            ),
        )

    @staticmethod
    def _session(observed, *, team_id: str) -> StrategySession:
        observed = replace(
            observed,
            our=replace(observed.our, team_id=team_id),
        )
        return StrategySession(
            team_id=team_id,
            last_round=observed.time.round_no,
            fingerprint=observation_fingerprint(observed),
            signature=static_signature(observed),
            observation=observed,
            decision=Decision(),
            simulation_action=None,
            certificate=None,
            scenario_weights=WEIGHTS,
        )


if __name__ == "__main__":
    unittest.main()
