import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.joint import (
    TacticalCandidate,
    is_valid_joint,
    solve_joint,
)
from future_war_agent.strategy.jobs import JobKind
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


def candidate(
    role_id: int,
    start: Position,
    target: Position | None,
    *,
    priority: int = 100,
) -> TacticalCandidate:
    action = None if target is None else Action.move(target)
    return TacticalCandidate(
        role_id=role_id,
        command_actor_id=role_id,
        action=action,
        job_kind=JobKind.PREPOSITION,
        start=start,
        move_target=target,
        priority=priority,
        completes_job=False,
        progress=1 if target is not None else 0,
    )


class JointSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = observation(
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 2, 1, "pioneer"),
                unit(10012, 3, 1, "worker", backpack=("stone",)),
            )
        )
        self.world = WorldGrid.from_observation(self.observed)

    def test_rejects_duplicate_move_destinations(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 2)),
            candidate(10011, Position(2, 1), Position(2, 2)),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_position_swap(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), Position(1, 1)),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_move_into_stationary_role(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), None),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_accepts_move_into_vacated_role_cell(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), Position(2, 2)),
        )
        self.assertTrue(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_duplicate_build_and_gold_overspend(self) -> None:
        build_one = TacticalCandidate.build(
            10010,
            Position(1, 1),
            Position(2, 2),
            "gatling",
            priority=400,
            gold_cost=25,
        )
        build_two = TacticalCandidate.build(
            10012,
            Position(3, 1),
            Position(2, 2),
            "railgun",
            priority=400,
            gold_cost=25,
        )
        self.assertFalse(
            is_valid_joint(self.observed, self.world, (build_one, build_two))
        )
        poor = observation(our_units=self.observed.our.units, gold=25)
        world = WorldGrid.from_observation(poor)
        other = TacticalCandidate.build(
            10012,
            Position(3, 1),
            Position(3, 2),
            "railgun",
            priority=400,
            gold_cost=25,
        )
        self.assertFalse(is_valid_joint(poor, world, (build_one, other)))

    def test_controller_cannot_also_take_personal_action(self) -> None:
        attack = TacticalCandidate.attack(
            role_id=10011,
            start=Position(2, 1),
            weapon_id=10020,
            target=Position(6, 6),
            priority=500,
        )
        move = candidate(10011, Position(2, 1), Position(2, 2))
        self.assertFalse(is_valid_joint(self.observed, self.world, (attack, move)))

    def test_solver_is_deterministic(self) -> None:
        choices = {
            10010: (
                candidate(10010, Position(1, 1), Position(1, 2)),
                candidate(10010, Position(1, 1), None),
            ),
            10011: (
                candidate(10011, Position(2, 1), Position(2, 2)),
                candidate(10011, Position(2, 1), None),
            ),
        }
        self.assertEqual(
            solve_joint(self.observed, self.world, choices),
            solve_joint(self.observed, self.world, choices),
        )


if __name__ == "__main__":
    unittest.main()
