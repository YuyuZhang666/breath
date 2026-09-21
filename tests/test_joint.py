import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.joint import (
    TacticalCandidate,
    candidates_for_jobs,
    decision_for_joint,
    enumerate_legal_joints,
    is_valid_joint,
    solve_joint,
)
from future_war_agent.strategy.jobs import Job, JobKind
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


def candidate(
    role_id: int,
    start: Position,
    target: Position | None,
    *,
    priority: int = 100,
    utility: float = 0.0,
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
        utility=utility,
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
            targets=(Position(6, 6),),
            priority=500,
        )
        move = candidate(10011, Position(2, 1), Position(2, 2))
        self.assertFalse(is_valid_joint(self.observed, self.world, (attack, move)))

    def test_one_target_attack_preserves_phase_2_payload(self) -> None:
        target = Position(6, 6)
        attack = TacticalCandidate.attack(
            role_id=10011,
            start=Position(2, 1),
            weapon_id=10020,
            targets=(target,),
            priority=500,
        )

        payload = decision_to_payload(decision_for_joint((attack,)))

        self.assertEqual(
            payload["roleCommandMap"],
            {
                "10020": {
                    "action": "attack",
                    "controllerId": "10011",
                    "targetPos": [{"x": 6, "y": 6}],
                }
            },
        )

    def test_multi_target_attack_preserves_every_target(self) -> None:
        targets = (Position(6, 6), Position(7, 6))

        attack = TacticalCandidate.attack(
            role_id=10011,
            start=Position(2, 1),
            weapon_id=10020,
            targets=targets,
            priority=500,
        )

        self.assertEqual(attack.action.target_positions, targets)

    def test_multi_target_attack_requires_injected_validator(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 1, 2, "worker"),
                unit(2, 2, 2, "gatling", attack_range=8, level=2),
            ),
        )
        world = WorldGrid.from_observation(observed)
        attack = TacticalCandidate.attack(
            role_id=1,
            start=Position(1, 2),
            weapon_id=2,
            targets=(Position(5, 2), Position(5, 3)),
            priority=500,
        )
        choices = {1: (attack,)}

        self.assertFalse(is_valid_joint(observed, world, (attack,)))
        self.assertEqual(enumerate_legal_joints(observed, world, choices), ())
        accepted = enumerate_legal_joints(
            observed,
            world,
            choices,
            attack_validator=lambda role, weapon, action: (
                role.unit_id == action.controller_id
                and weapon.unit_id == 2
                and len(action.target_positions) == 2
            ),
        )
        self.assertEqual(accepted, ((attack,),))

    def test_legal_enumeration_is_deterministic_and_honors_limit(self) -> None:
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

        first = enumerate_legal_joints(
            self.observed,
            self.world,
            choices,
            limit=2,
        )
        second = enumerate_legal_joints(
            self.observed,
            self.world,
            choices,
            limit=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

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

    def test_solver_uses_job_utility_before_repr_tiebreak(self) -> None:
        choices = {
            10010: (
                candidate(
                    10010,
                    Position(1, 1),
                    Position(1, 2),
                    utility=1.0,
                ),
                candidate(
                    10010,
                    Position(1, 1),
                    Position(2, 2),
                    utility=50.0,
                ),
            ),
        }

        decision = solve_joint(self.observed, self.world, choices)

        self.assertEqual(
            decision.commands[10010].target_positions,
            (Position(2, 2),),
        )

    def test_two_workers_cannot_move_toward_same_wall_job(self) -> None:
        observed = observation(
            our_units=(
                unit(1, 1, 1, 'worker', backpack=('stone',)),
                unit(2, 3, 1, 'worker', backpack=('stone',)),
            ),
        )
        world = WorldGrid.from_observation(observed)
        target = Position(2, 4)
        jobs = {
            role_id: (
                Job(
                    role_id=role_id,
                    kind=JobKind.BUILD_WALL,
                    target=target,
                    priority=350,
                    value=10,
                    name='wall',
                    quantity=1,
                ),
            )
            for role_id in (1, 2)
        }
        first = candidates_for_jobs(
            observed,
            world,
            world.unit_by_id(1),
            jobs[1],
        )[0]
        second = candidates_for_jobs(
            observed,
            world,
            world.unit_by_id(2),
            jobs[2],
        )[0]

        self.assertNotEqual(first.move_target, second.move_target)
        self.assertEqual(first.exclusive_job_key, second.exclusive_job_key)
        self.assertFalse(is_valid_joint(observed, world, (first, second)))

    def test_two_workers_may_collect_same_tail_mine(self) -> None:
        mine = Position(2, 2)
        observed = observation(
            our_units=(
                unit(1, 1, 2, 'worker'),
                unit(2, 2, 1, 'worker'),
            ),
            zones=(Zone(mine, 'stone'),),
        )
        world = WorldGrid.from_observation(observed)
        first = TacticalCandidate(
            role_id=1,
            command_actor_id=1,
            action=Action.collect(mine),
            job_kind=JobKind.COLLECT,
            start=Position(1, 2),
        )
        second = TacticalCandidate(
            role_id=2,
            command_actor_id=2,
            action=Action.collect(mine),
            job_kind=JobKind.COLLECT,
            start=Position(2, 1),
        )

        self.assertTrue(is_valid_joint(observed, world, (first, second)))

    def test_gold_reserve_applies_to_whole_joint(self) -> None:
        observed = observation(our_units=self.observed.our.units, gold=50)
        world = WorldGrid.from_observation(observed)
        joint = (
            TacticalCandidate.build(
                10010,
                Position(1, 1),
                Position(1, 2),
                'gatling',
                priority=400,
                gold_cost=25,
            ),
            TacticalCandidate.build(
                10012,
                Position(3, 1),
                Position(3, 2),
                'railgun',
                priority=400,
                gold_cost=25,
            ),
        )

        self.assertTrue(is_valid_joint(observed, world, joint))
        self.assertFalse(is_valid_joint(observed, world, joint, gold_reserve=25))

    def test_completed_preposition_does_not_move_away(self) -> None:
        role = self.world.unit_by_id(10010)
        jobs = (
            Job(
                role_id=10010,
                kind=JobKind.PREPOSITION,
                target=Position(1, 1),
                priority=100,
                value=0,
            ),
        )

        choices = candidates_for_jobs(self.observed, self.world, role, jobs)

        self.assertEqual(len(choices), 1)
        self.assertIsNone(choices[0].action)

    def test_use_item_matches_inventory_name_case_insensitively(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 1, 1, "worker", backpack=("medicine",)),
            ),
        )
        world = WorldGrid.from_observation(observed)
        use = TacticalCandidate.personal_action(
            10010,
            Position(1, 1),
            Action.use("Medicine"),
            JobKind.USE_ITEM,
            priority=475,
        )

        self.assertTrue(is_valid_joint(observed, world, (use,)))


if __name__ == "__main__":
    unittest.main()
