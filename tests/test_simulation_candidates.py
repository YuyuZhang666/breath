import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.night import assign_controllers
from future_war_agent.strategy.simulation.candidates import generate_root_actions
from future_war_agent.strategy.simulation.state import build_sim_state
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


EXPLICIT_WEAPON_FIELDS = frozenset(
    {"attackPower", "attackRange", "level", "cooldown"}
)


class SimulationCandidateTests(unittest.TestCase):
    def test_adjacent_controller_generates_deterministic_multi_target_roots(self) -> None:
        observed, world, state = self._scenario(controller=Position(4, 5))

        first = generate_root_actions(observed, world, state)
        second = generate_root_actions(observed, world, state)

        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertLessEqual(len(first), 64)
        attacks = tuple(
            attack
            for root in first
            for attack in root.simulation_action.weapon_attacks
        )
        self.assertTrue(attacks)
        self.assertTrue(all(len(attack.targets) == 2 for attack in attacks))
        self.assertTrue(
            any(
                action.kind is ActionKind.ATTACK
                and len(action.target_positions) == 2
                for root in first
                for action in root.decision.commands.values()
            )
        )

    def test_remote_controller_moves_before_weapon_can_fire(self) -> None:
        observed, world, state = self._scenario(controller=Position(1, 1))

        roots = generate_root_actions(observed, world, state)

        self.assertTrue(roots)
        self.assertTrue(
            any(root.simulation_action.role_moves for root in roots)
        )
        self.assertTrue(
            all(not root.simulation_action.weapon_attacks for root in roots)
        )

    def test_root_stable_keys_and_simulation_actions_match_decisions(self) -> None:
        observed, world, state = self._scenario(controller=Position(4, 5))

        roots = generate_root_actions(observed, world, state)

        self.assertEqual(len({root.stable_key for root in roots}), len(roots))
        for root in roots:
            for move in root.simulation_action.role_moves:
                action = root.decision.commands[move.role_id]
                self.assertIs(action.kind, ActionKind.MOVE)
                self.assertEqual(action.target_positions, (move.target,))
            for attack in root.simulation_action.weapon_attacks:
                action = root.decision.commands[attack.weapon_id]
                self.assertIs(action.kind, ActionKind.ATTACK)
                self.assertEqual(action.controller_id, attack.controller_id)
                self.assertEqual(action.target_positions, attack.targets)

    @staticmethod
    def _scenario(controller: Position):
        observed = observation(
            round_no=71,
            width=15,
            height=15,
            our_units=(
                unit(10, controller.x, controller.y, "worker", health=200),
                unit(20, 5, 5, "station", health=800, level=1),
                unit(
                    30,
                    5,
                    4,
                    "gatling",
                    health=150,
                    attack_power=10,
                    attack_range=8,
                    level=2,
                    cooldown=0,
                    provided_fields=EXPLICIT_WEAPON_FIELDS,
                ),
            ),
            robots=(robot(1, 8, 4), robot(2, 5, 8)),
        )
        world = WorldGrid.from_observation(observed)
        assignments = assign_controllers(observed, world)
        state = build_sim_state(observed, assignments)
        return observed, world, state


if __name__ == "__main__":
    unittest.main()
