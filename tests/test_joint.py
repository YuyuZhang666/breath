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

    def test_recall_wait_at_goal_beats_move_to_other_goal(self) -> None:
        wait_at_goal = TacticalCandidate(
            role_id=10010,
            command_actor_id=10010,
            action=None,
            job_kind=JobKind.RECALL,
            start=Position(1, 1),
            priority=501,
            completes_job=True,
            progress=1,
            utility=0.0,
        )
        move_to_other_goal = candidate(
            10010,
            Position(1, 1),
            Position(2, 2),
            priority=501,
            utility=-1.0,
        )

        decision = solve_joint(
            self.observed,
            self.world,
            {10010: (wait_at_goal, move_to_other_goal)},
        )

        self.assertNotIn(10010, decision.commands)

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

    def test_two_workers_cannot_move_toward_same_weapon_job(self) -> None:
        observed = observation(
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 3, 1, 'worker'),
            ),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        target = Position(2, 4)
        jobs = {
            role_id: (
                Job(
                    role_id=role_id,
                    kind=JobKind.BUILD_WEAPON,
                    target=target,
                    priority=450,
                    value=10,
                    name='gatling',
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
        observed = observation(
            our_units=self.observed.our.units + (
                unit(10013, 1, 3, 'station', level=1),
            ),
            gold=50,
        )
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

    def test_candidate_limit_preserves_distinct_work_families(self) -> None:
        observed = observation(
            our_units=(unit(1, 1, 1, 'worker'),),
            zones=(Zone(Position(2, 2), 'stone'),),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        role = world.unit_by_id(1)
        jobs = (
            Job(1, JobKind.BUILD_WEAPON, Position(5, 5), 450, 0, 'gatling'),
            Job(1, JobKind.BUILD_WEAPON, Position(5, 6), 450, 0, 'railgun'),
            Job(1, JobKind.BUILD_WEAPON, Position(6, 5), 450, 0, 'rocket'),
            Job(1, JobKind.COLLECT, Position(2, 2), 426, 0, 'stone'),
        )

        choices = candidates_for_jobs(observed, world, role, jobs, limit=4)

        self.assertIn(JobKind.BUILD_WEAPON, {item.job_kind for item in choices})
        self.assertIn(JobKind.COLLECT, {item.job_kind for item in choices})
        self.assertTrue(any(item.action is None for item in choices))
        build_keys = {
            item.exclusive_job_key
            for item in choices
            if item.job_kind is JobKind.BUILD_WEAPON
        }
        self.assertEqual(len(build_keys), 2)

    def test_candidate_limit_keeps_all_three_opening_work_families(
        self,
    ) -> None:
        observed = observation(
            our_units=(
                unit(1, 1, 1, 'worker', backpack=('stone',)),
            ),
            zones=(Zone(Position(2, 2), 'stone'),),
            gold=75,
        )
        world = WorldGrid.from_observation(observed)
        role = world.unit_by_id(1)
        jobs = (
            Job(1, JobKind.BUILD_WEAPON, Position(5, 5), 450, 0, 'gatling'),
            Job(1, JobKind.COLLECT, Position(2, 2), 426, 0, 'stone'),
            Job(1, JobKind.BUILD_WALL, Position(5, 6), 425, 0, 'wall'),
        )

        choices = candidates_for_jobs(observed, world, role, jobs, limit=4)

        self.assertTrue(
            {
                JobKind.BUILD_WEAPON,
                JobKind.COLLECT,
                JobKind.BUILD_WALL,
            }.issubset({item.job_kind for item in choices})
        )
        self.assertTrue(any(item.action is None for item in choices))

    def test_solver_chooses_fully_active_joint_before_waiting(self) -> None:
        observed = observation(
            our_units=(
                unit(1, 1, 1, 'worker'),
                unit(2, 3, 1, 'worker'),
            ),
        )
        world = WorldGrid.from_observation(observed)
        choices = {
            1: (
                TacticalCandidate(
                    role_id=1,
                    command_actor_id=1,
                    action=Action.move(Position(2, 1)),
                    job_kind=JobKind.RECALL,
                    start=Position(1, 1),
                    move_target=Position(2, 1),
                    priority=500,
                ),
                TacticalCandidate.wait(1, Position(1, 1)),
            ),
            2: (
                TacticalCandidate(
                    role_id=2,
                    command_actor_id=2,
                    action=Action.move(Position(2, 1)),
                    job_kind=JobKind.BUILD_WEAPON,
                    start=Position(3, 1),
                    move_target=Position(2, 1),
                    priority=500,
                ),
                TacticalCandidate(
                    role_id=2,
                    command_actor_id=2,
                    action=Action.move(Position(3, 2)),
                    job_kind=JobKind.COLLECT,
                    start=Position(3, 1),
                    move_target=Position(3, 2),
                    priority=-10,
                ),
                TacticalCandidate.wait(2, Position(3, 1)),
            ),
        }

        decision = solve_joint(observed, world, choices)

        self.assertEqual(set(decision.commands), {1, 2})
        self.assertEqual(
            decision.commands[2].target_positions,
            (Position(3, 2),),
        )

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

    def test_move_job_offers_multiple_first_steps(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, "worker"),),
        )
        world = WorldGrid.from_observation(observed)
        jobs = (
            Job(10010, JobKind.BUILD_WALL, Position(8, 8), 350, 0, "wall"),
        )

        choices = candidates_for_jobs(
            observed,
            world,
            world.unit_by_id(10010),
            jobs,
        )

        moves = [item for item in choices if item.move_target is not None]
        self.assertGreaterEqual(len(moves), 3)

    def test_idle_day_worker_gets_sidestep_candidates(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 8, 8, "worker"),
                unit(10013, 10, 10, "station", level=1),
            ),
        )
        world = WorldGrid.from_observation(observed)

        choices = candidates_for_jobs(
            observed,
            world,
            world.unit_by_id(10010),
            (),
        )

        sidesteps = [item for item in choices if item.action is not None]
        self.assertTrue(sidesteps)
        base_distance = world.station_distance(Position(8, 8))
        for item in sidesteps:
            self.assertIsNotNone(item.move_target)
            self.assertLessEqual(
                world.station_distance(item.move_target),
                base_distance,
            )

    def test_night_idle_worker_waits_without_sidestep(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(unit(10010, 8, 8, "worker"),),
        )
        world = WorldGrid.from_observation(observed)

        choices = candidates_for_jobs(
            observed,
            world,
            world.unit_by_id(10010),
            (),
        )

        self.assertFalse(any(item.action is not None for item in choices))


if __name__ == "__main__":
    unittest.main()
