import unittest
from fractions import Fraction

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.night import assign_controllers
from future_war_agent.strategy.simulation.candidates import generate_root_actions
from future_war_agent.strategy.simulation.certificate import WaveClassification
from future_war_agent.strategy.simulation.config import Phase3Config, Phase3Level
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.strategy.simulation.search import _scenario_outcome, search_night
from future_war_agent.strategy.simulation.state import (
    SimRole,
    SimState,
    SimStructure,
    SimWeapon,
    build_sim_state,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


WEIGHTS = (Fraction(1, 4),) * 4
EXPLICIT_WEAPON_FIELDS = frozenset(
    {"attackPower", "attackRange", "level", "cooldown"}
)


class SimulationSearchTests(unittest.TestCase):
    def test_search_obeys_root_scenario_and_horizon_caps(self) -> None:
        observed = self._gatling_scenario()

        result = search_night(
            observed,
            WEIGHTS,
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertLessEqual(result.stats.roots_generated, 8)
        self.assertEqual(result.stats.roots_evaluated, result.stats.roots_generated)
        self.assertEqual(result.stats.scenarios_per_root, 2)
        self.assertLessEqual(result.stats.maximum_steps, 4)

    def test_lite_obeys_four_one_two_budget(self) -> None:
        result = search_night(
            self._gatling_scenario(),
            WEIGHTS,
            level=Phase3Level.LITE,
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertLessEqual(result.stats.roots_generated, 4)
        self.assertEqual(result.stats.scenarios_per_root, 1)
        self.assertLessEqual(result.stats.maximum_steps, 2)
        self.assertEqual(len(result.certificate.outcomes), 1)
        self.assertEqual(result.certificate.outcomes[0].weight, Fraction(1))
        self.assertIs(
            result.certificate.classification,
            WaveClassification.UNKNOWN,
        )
        self.assertFalse(result.certificate.secured)

    def test_complete_tail_replaces_short_horizon_unknown(self) -> None:
        result = search_night(
            self._gatling_scenario(),
            WEIGHTS,
            level=Phase3Level.LITE,
            config=Phase3Config(tail_visible_roster_complete=True),
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertTrue(result.certificate.outcomes[0].tail_estimated)
        self.assertTrue(result.certificate.projection_complete)
        self.assertIsNot(
            result.certificate.classification,
            WaveClassification.UNKNOWN,
        )

    def test_tail_detects_lethal_damage_beyond_four_rounds(self) -> None:
        observed = observation(
            round_no=71,
            width=16,
            height=16,
            our_units=(
                unit(1, 1, 1, 'worker', health=100),
                unit(2, 14, 14, 'station', health=100, level=1),
            ),
            robots=(
                robot(
                    9,
                    0,
                    0,
                    role_type='bossRobot',
                    health=800,
                ),
            ),
        )

        result = search_night(
            observed,
            WEIGHTS,
            config=Phase3Config(tail_visible_roster_complete=True),
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertIs(
            result.certificate.classification,
            WaveClassification.WAVE_UNSAFE,
        )
        self.assertLess(result.certificate.worst_survival_margin, 0)

    def test_stable_ties_return_byte_equivalent_decisions(self) -> None:
        observed = self._gatling_scenario()

        first = search_night(observed, WEIGHTS, clock=lambda: 0.0, deadline=1.0)
        second = search_night(observed, WEIGHTS, clock=lambda: 0.0, deadline=1.0)

        self.assertEqual(
            decision_to_payload(first.decision),
            decision_to_payload(second.decision),
        )
        self.assertEqual(first.simulation_action, second.simulation_action)
        self.assertEqual(first.certificate, second.certificate)

    def test_all_unsafe_state_returns_least_bad_fully_evaluated_root(self) -> None:
        observed = observation(
            round_no=71,
            width=12,
            height=10,
            our_units=(
                unit(1, 0, 1, "worker", health=100),
                unit(2, 5, 5, "station", health=5, level=1),
                unit(
                    3,
                    0,
                    0,
                    "gatling",
                    health=100,
                    attack_power=10,
                    attack_range=1,
                    level=1,
                    cooldown=0,
                    provided_fields=EXPLICIT_WEAPON_FIELDS,
                ),
            ),
            robots=(robot(9, 3, 5),),
        )

        result = search_night(
            observed,
            WEIGHTS,
            config=Phase3Config(max_horizon=2, watchdog_seconds=10),
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertIs(
            result.certificate.classification,
            WaveClassification.WAVE_UNSAFE,
        )
        self.assertEqual(result.stats.roots_evaluated, result.stats.roots_generated)
        self.assertGreater(result.stats.roots_evaluated, 0)

    def test_station_destruction_stops_rollout_before_later_kill_score(self) -> None:
        observed = observation(
            round_no=71,
            width=12,
            height=10,
            our_units=(
                unit(1, 0, 4, "worker", health=100),
                unit(2, 5, 5, "station", health=5, level=1),
                unit(
                    3,
                    1,
                    4,
                    "gatling",
                    health=100,
                    attack_power=10,
                    attack_range=8,
                    level=1,
                    cooldown=1,
                    provided_fields=EXPLICIT_WEAPON_FIELDS,
                ),
            ),
            robots=(robot(9, 2, 5, health=10),),
        )

        result = search_night(
            observed,
            WEIGHTS,
            config=Phase3Config(max_horizon=3, watchdog_seconds=10),
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertTrue(
            all(
                outcome.station_health == 0
                for outcome in result.certificate.outcomes
            )
        )
        self.assertTrue(
            all(
                outcome.owned_kill_score == 0
                for outcome in result.certificate.outcomes
            )
        )

    def test_outcome_tracks_required_and_non_key_asset_scopes(self) -> None:
        station_cells = frozenset(
            {
                Position(8, 8),
                Position(9, 8),
                Position(8, 9),
                Position(9, 9),
            }
        )
        wall_cells = frozenset({Position(7, 7)})
        state = SimState(
            round_no=73,
            width=12,
            height=12,
            remaining_night_turns=58,
            team_type="challenger",
            static_blocked=station_cells | wall_cells | frozenset(
                {Position(2, 1), Position(5, 5)}
            ),
            roles=(
                SimRole(1, "worker", Position(1, 1), 50, 10, Position(1, 1)),
                SimRole(2, "pioneer", Position(3, 3), 5, None, None),
            ),
            station=SimStructure(
                100,
                "station",
                Position(8, 8),
                station_cells,
                500,
                1,
            ),
            walls=(
                SimStructure(
                    20,
                    "wall",
                    Position(7, 7),
                    wall_cells,
                    30,
                    1,
                ),
            ),
            weapons=(
                SimWeapon(10, "gatling", Position(2, 1), 40, 10, 6, 1, 0),
                SimWeapon(11, "rocket", Position(5, 5), 25, 20, 6, 1, 0),
            ),
            robots=(),
        )

        outcome = _scenario_outcome(
            state,
            Fraction(1),
            initial_controlled_role_ids=frozenset({1, 2, 3}),
            initial_controller_ids=frozenset({1, 3}),
            initial_key_weapon_ids=frozenset({10, 12}),
            initial_wall_ids=frozenset({20, 21}),
            initial_weapon_ids=frozenset({10, 11, 12}),
            non_key_weapon_ids=frozenset({11}),
        )

        self.assertEqual(outcome.surviving_controlled_role_count, 2)
        self.assertEqual(outcome.surviving_controller_count, 1)
        self.assertEqual(outcome.controller_losses, 1)
        self.assertEqual(outcome.surviving_key_weapon_count, 1)
        self.assertEqual(outcome.key_weapon_losses, 1)
        self.assertEqual(outcome.wall_losses, 1)
        self.assertEqual(outcome.weapon_losses, 1)
        self.assertEqual(outcome.minimum_controlled_role_health, 5)
        self.assertEqual(outcome.surviving_wall_non_key_weapon_value, 55)

    def test_rocket_cooldown_changes_available_root_action(self) -> None:
        ready = self._rocket_scenario(cooldown=0)
        cooling = self._rocket_scenario(cooldown=1)
        ready_world = WorldGrid.from_observation(ready)
        cooling_world = WorldGrid.from_observation(cooling)
        ready_assignments = assign_controllers(ready, ready_world)
        cooling_assignments = assign_controllers(cooling, cooling_world)
        ready_roots = generate_root_actions(
            ready,
            ready_world,
            build_sim_state(ready, ready_assignments),
            controller_assignments=ready_assignments,
        )
        cooling_roots = generate_root_actions(
            cooling,
            cooling_world,
            build_sim_state(cooling, cooling_assignments),
            controller_assignments=cooling_assignments,
        )

        self.assertTrue(
            any(
                action.kind is ActionKind.ATTACK
                for root in ready_roots
                for action in root.decision.commands.values()
            )
        )
        self.assertFalse(
            any(
                action.kind is ActionKind.ATTACK
                for root in cooling_roots
                for action in root.decision.commands.values()
            )
        )

    def test_deadline_before_search_raises_without_result(self) -> None:
        with self.assertRaises(DeadlineExceeded):
            search_night(
                self._gatling_scenario(),
                WEIGHTS,
                clock=lambda: 1.0,
                deadline=1.0,
            )

    def test_deadline_during_evaluation_discards_completed_prefix(self) -> None:
        class AdvancingClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                self.value += 0.1
                return self.value

        with self.assertRaises(DeadlineExceeded):
            search_night(
                self._gatling_scenario(),
                WEIGHTS,
                clock=AdvancingClock(),
                deadline=0.55,
            )

    def test_deadline_after_complete_root_returns_anytime_best(self) -> None:
        class PrefixClock:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> float:
                self.calls += 1
                # Cooperative checks now cover candidates, joint-fire, BFS,
                # exact steps and the TailEstimator. This boundary expires
                # after one complete root instead of counting only outer loops.
                return 0.0 if self.calls <= 200 else 1.0

        result = search_night(
            self._gatling_scenario(),
            WEIGHTS,
            clock=PrefixClock(),
            deadline=0.5,
        )

        self.assertTrue(result.stats.deadline_hit)
        self.assertGreaterEqual(result.stats.roots_evaluated, 1)
        self.assertLess(result.stats.roots_evaluated, result.stats.roots_generated)

    def test_invalid_scenario_weights_are_rejected(self) -> None:
        invalid = (
            (Fraction(1, 3),) * 3,
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), Fraction(0)),
        )
        for weights in invalid:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    search_night(
                        self._gatling_scenario(),
                        weights,
                        clock=lambda: 0.0,
                        deadline=1.0,
                    )

    @staticmethod
    def _gatling_scenario():
        return observation(
            round_no=71,
            width=14,
            height=12,
            our_units=(
                unit(1, 3, 4, "worker", health=100),
                unit(2, 8, 8, "station", health=500, level=1),
                unit(
                    3,
                    4,
                    4,
                    "gatling",
                    health=100,
                    attack_power=10,
                    attack_range=8,
                    level=1,
                    cooldown=0,
                    provided_fields=EXPLICIT_WEAPON_FIELDS,
                ),
            ),
            robots=(robot(9, 6, 4, health=20),),
        )

    @staticmethod
    def _rocket_scenario(*, cooldown: int):
        return observation(
            round_no=71,
            width=14,
            height=12,
            our_units=(
                unit(1, 3, 4, "worker", health=100),
                unit(2, 8, 8, "station", health=500, level=1),
                unit(
                    3,
                    4,
                    4,
                    "rocket",
                    health=100,
                    attack_power=20,
                    attack_range=8,
                    level=1,
                    cooldown=cooldown,
                    provided_fields=EXPLICIT_WEAPON_FIELDS,
                ),
            ),
            robots=(robot(9, 6, 4, health=40),),
        )


if __name__ == "__main__":
    unittest.main()
