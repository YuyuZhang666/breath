import unittest
from fractions import Fraction

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.certificate import WaveClassification
from future_war_agent.strategy.simulation.config import Phase3Config
from future_war_agent.strategy.simulation.errors import DeadlineExceeded
from future_war_agent.strategy.simulation.search import search_night
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

        self.assertLessEqual(result.stats.roots_generated, 64)
        self.assertEqual(result.stats.roots_evaluated, result.stats.roots_generated)
        self.assertEqual(result.stats.scenarios_per_root, 4)
        self.assertLessEqual(result.stats.maximum_steps, 6)

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

    def test_rocket_cooldown_changes_available_root_action(self) -> None:
        ready = self._rocket_scenario(cooldown=0)
        cooling = self._rocket_scenario(cooldown=1)
        config = Phase3Config(max_horizon=5, watchdog_seconds=10)

        ready_result = search_night(
            ready,
            WEIGHTS,
            config=config,
            clock=lambda: 0.0,
            deadline=1.0,
        )
        cooling_result = search_night(
            cooling,
            WEIGHTS,
            config=config,
            clock=lambda: 0.0,
            deadline=1.0,
        )

        self.assertTrue(
            any(
                action.kind is ActionKind.ATTACK
                for action in ready_result.decision.commands.values()
            )
        )
        self.assertFalse(
            any(
                action.kind is ActionKind.ATTACK
                for action in cooling_result.decision.commands.values()
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
