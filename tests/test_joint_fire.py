import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.joint_fire import plan_joint_fire
from future_war_agent.strategy.night import ControllerAssignment
from future_war_agent.strategy.planner import plan_turn
from future_war_agent.strategy.policy import StrategicIntent, StrategyProfile
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


WEAPON_FIELDS = ('attackPower', 'attackRange', 'level', 'cooldown')


def weapon(
    unit_id: int,
    x: int,
    y: int,
    role_type: str,
    *,
    attack_power: int,
    attack_range: int = 8,
    level: int = 1,
    cooldown: int = 0,
):
    return unit(
        unit_id,
        x,
        y,
        role_type,
        health=1000,
        attack_power=attack_power,
        attack_range=attack_range,
        level=level,
        cooldown=cooldown,
        provided_fields=WEAPON_FIELDS,
    )


class JointFireTests(unittest.TestCase):
    def test_joint_enumeration_splits_fire_instead_of_overkilling(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 4, 'worker'),
                unit(102, 2, 8, 'worker'),
                unit(200, 8, 6, 'station', health=1000, level=1),
                weapon(301, 3, 4, 'gatling', attack_power=10),
                weapon(302, 3, 8, 'gatling', attack_power=10),
            ),
            robots=(
                robot(501, 6, 4, health=10),
                robot(502, 6, 8, health=10),
            ),
        )
        assignments = (
            ControllerAssignment(101, 301, Position(2, 4), 0),
            ControllerAssignment(102, 302, Position(2, 8), 0),
        )

        plan = plan_joint_fire(
            observed,
            WorldGrid.from_observation(observed),
            controller_assignments=assignments,
        )

        attacks = tuple(
            action
            for action in plan.decision.commands.values()
            if action.kind is ActionKind.ATTACK
        )
        self.assertEqual(len(attacks), 2)
        self.assertEqual(
            {action.target_positions[0] for action in attacks},
            {Position(6, 4), Position(6, 8)},
        )
        self.assertLessEqual(plan.combinations_evaluated, 125)
        self.assertEqual(validate_decision(observed, plan.decision), plan.decision)

    def test_profile_changes_target_priority(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 5, 'worker'),
                unit(200, 8, 8, 'station', health=1000, level=1),
                weapon(301, 3, 5, 'gatling', attack_power=10),
            ),
            robots=(
                robot(501, 6, 3, health=10, role_type='smallRobot'),
                robot(502, 6, 7, health=800, role_type='bossRobot'),
            ),
        )
        world = WorldGrid.from_observation(observed)
        assignments = (ControllerAssignment(101, 301, Position(2, 5), 0),)

        survive = plan_joint_fire(
            observed,
            world,
            StrategicIntent(profile=StrategyProfile.SURVIVE),
            controller_assignments=assignments,
        )
        score = plan_joint_fire(
            observed,
            world,
            StrategicIntent(profile=StrategyProfile.SCORE),
            controller_assignments=assignments,
        )

        self.assertEqual(
            survive.decision.commands[301].target_positions,
            (Position(6, 7),),
        )
        self.assertEqual(
            score.decision.commands[301].target_positions,
            (Position(6, 3),),
        )

    def test_upgraded_rocket_options_are_bounded_and_legal(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 5, 'worker'),
                unit(200, 8, 8, 'station', health=1000, level=1),
                weapon(
                    301,
                    3,
                    5,
                    'rocket',
                    attack_power=20,
                    level=3,
                ),
            ),
            robots=(
                robot(501, 6, 3),
                robot(502, 6, 5),
                robot(503, 6, 7),
            ),
        )
        assignments = (ControllerAssignment(101, 301, Position(2, 5), 0),)

        plan = plan_joint_fire(
            observed,
            WorldGrid.from_observation(observed),
            controller_assignments=assignments,
        )

        action = plan.decision.commands[301]
        self.assertEqual(len(action.target_positions), 3)
        self.assertEqual(len(set(action.target_positions)), 3)
        self.assertLessEqual(plan.attack_option_count, 4)
        self.assertLessEqual(plan.combinations_evaluated, 5)
        self.assertEqual(validate_decision(observed, plan.decision), plan.decision)

    def test_output_is_deterministic(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 5, 'worker'),
                unit(200, 8, 8, 'station', health=1000, level=1),
                weapon(301, 3, 5, 'railgun', attack_power=40),
            ),
            robots=(robot(501, 6, 4), robot(502, 6, 6)),
        )
        world = WorldGrid.from_observation(observed)
        assignments = (ControllerAssignment(101, 301, Position(2, 5), 0),)

        first = plan_joint_fire(
            observed,
            world,
            controller_assignments=assignments,
        ).decision
        second = plan_joint_fire(
            observed,
            world,
            controller_assignments=assignments,
        ).decision

        self.assertEqual(first, second)

    def test_weapon_diagnostics_explain_selected_and_blocked_fire(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 5, 'worker'),
                unit(102, 2, 8, 'worker'),
                unit(200, 8, 8, 'station', health=1000, level=1),
                weapon(301, 3, 5, 'gatling', attack_power=10),
                weapon(302, 3, 8, 'rocket', attack_power=20, cooldown=2),
            ),
            robots=(robot(501, 6, 5),),
        )
        assignments = (
            ControllerAssignment(101, 301, Position(2, 5), 0),
            ControllerAssignment(102, 302, Position(2, 8), 0),
        )

        plan = plan_joint_fire(
            observed,
            WorldGrid.from_observation(observed),
            controller_assignments=assignments,
        )

        self.assertEqual(len(plan.weapon_log), 2)
        self.assertIn('id=301', plan.weapon_log[0])
        self.assertIn('selected=attack', plan.weapon_log[0])
        self.assertIn('id=302', plan.weapon_log[1])
        self.assertIn('reason=cooldown', plan.weapon_log[1])

    def test_default_fallback_does_not_attack_other_team_robot(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(101, 2, 5, 'worker'),
                unit(200, 8, 8, 'station', health=1000, level=1),
                unit(301, 3, 5, 'gatling', attack_range=8, level=1),
            ),
            robots=(robot(501, 6, 5, target_team='defender'),),
        )

        decision = plan_turn(observed)

        self.assertFalse(
            any(
                action.kind is ActionKind.ATTACK
                for action in decision.commands.values()
            )
        )


if __name__ == '__main__':
    unittest.main()
