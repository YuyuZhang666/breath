import unittest
from dataclasses import replace
from fractions import Fraction

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.reconcile import (
    _reconciliation_loss,
    _weights_from_losses,
    reconcile_scenario_weights,
    uniform_scenario_weights,
)
from future_war_agent.strategy.session import (
    StrategySession,
    observation_fingerprint,
    static_signature,
)
from future_war_agent.strategy.simulation.candidates import SimJointAction
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimRole,
    SimState,
    SimStructure,
    SimWeapon,
)
from tests.strategy_helpers import observation, unit


class StrategyReconcileTests(unittest.TestCase):
    def test_loss_combines_robot_position_and_controlled_health_errors(self) -> None:
        predicted = self._state()
        current = replace(
            predicted,
            robots=(replace(predicted.robots[0], position=Position(3, 2)),),
            roles=(replace(predicted.roles[0], health=80),),
            station=replace(predicted.station, health=490),
            walls=(replace(predicted.walls[0], health=90),),
            weapons=(replace(predicted.weapons[0], health=70),),
        )

        loss, comparable = _reconciliation_loss(predicted, current)

        self.assertTrue(comparable)
        self.assertEqual(loss, 42)

    def test_alive_dead_robot_mismatch_costs_twenty(self) -> None:
        predicted = self._state()
        current = replace(predicted, robots=())

        loss, comparable = _reconciliation_loss(predicted, current)

        self.assertTrue(comparable)
        self.assertEqual(loss, 20)

    def test_no_comparable_entities_reports_missing_evidence(self) -> None:
        predicted = self._state(empty_assets=True)
        current = replace(
            predicted,
            station=replace(predicted.station, unit_id=999),
        )

        loss, comparable = _reconciliation_loss(predicted, current)

        self.assertEqual(loss, 0)
        self.assertFalse(comparable)

    def test_exact_fraction_normalization_reserves_five_percent_floor(self) -> None:
        weights = _weights_from_losses(
            uniform_scenario_weights(),
            (0, 1, 3, 19),
            Fraction(1, 20),
        )

        self.assertEqual(
            weights,
            (
                Fraction(89, 180),
                Fraction(49, 180),
                Fraction(29, 180),
                Fraction(13, 180),
            ),
        )
        self.assertEqual(sum(weights, Fraction()), 1)
        self.assertTrue(all(weight >= Fraction(1, 20) for weight in weights))

    def test_missing_simulation_action_keeps_old_weights(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 1, 1, "worker"),
                unit(2, 5, 5, "station", health=500),
            ),
        )
        old = (
            Fraction(1, 2),
            Fraction(1, 6),
            Fraction(1, 6),
            Fraction(1, 6),
        )
        previous = StrategySession(
            team_id="team",
            last_round=71,
            fingerprint=observation_fingerprint(observed),
            signature=static_signature(observed),
            observation=observed,
            decision=Decision(),
            simulation_action=None,
            certificate=None,
            scenario_weights=old,
        )

        self.assertEqual(
            reconcile_scenario_weights(previous, observed),
            old,
        )

    def test_unsupported_conversion_keeps_old_weights(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(unit(1, 1, 1, "worker"),),
        )
        previous = StrategySession(
            team_id="team",
            last_round=71,
            fingerprint=observation_fingerprint(observed),
            signature=static_signature(observed),
            observation=observed,
            decision=Decision(),
            simulation_action=SimJointAction(),
            certificate=None,
            scenario_weights=uniform_scenario_weights(),
        )

        self.assertEqual(
            reconcile_scenario_weights(previous, observed),
            uniform_scenario_weights(),
        )

    @staticmethod
    def _state(*, empty_assets: bool = False) -> SimState:
        station_cells = frozenset(
            {Position(8, 8), Position(9, 8), Position(8, 9), Position(9, 9)}
        )
        station = SimStructure(100, "station", Position(8, 8), station_cells, 500, 1)
        if empty_assets:
            roles = ()
            walls = ()
            weapons = ()
            robots = ()
        else:
            roles = (SimRole(1, "worker", Position(2, 2), 90, 300, Position(3, 4)),)
            wall_cell = Position(6, 6)
            walls = (SimStructure(200, "wall", wall_cell, frozenset({wall_cell}), 100, 1),)
            weapons = (SimWeapon(300, "gatling", Position(4, 4), 80, 10, 6, 1, 0),)
            robots = (SimRobot(9, "smallRobot", Position(1, 1), 40, 5, 3, 1, False),)
        return SimState(
            72,
            12,
            12,
            59,
            "challenger",
            station_cells,
            roles,
            station,
            walls,
            weapons,
            robots,
        )


if __name__ == "__main__":
    unittest.main()
