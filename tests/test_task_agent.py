import unittest
from dataclasses import replace

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position, TaskPointState, WorldNews
from future_war_agent.strategy.policy import StrategyProfile, intent_for_profile
from future_war_agent.strategy.task_agent import TaskAgent, TaskAgentState
from tests.strategy_helpers import observation, unit


def task(
    task_type: str,
    x: int,
    y: int,
    *,
    score: int = 10,
    gold: int = 0,
    timeout: int | None = 20,
    cooldown: int = 0,
    valid: bool = True,
) -> TaskPointState:
    return TaskPointState(
        task_type=task_type,
        position=Position(x, y),
        cooldown_rounds=cooldown,
        score_reward=score,
        gold_reward=gold,
        is_valid=valid,
        timeout_rounds=timeout,
    )


SCORE_INTENT = intent_for_profile(StrategyProfile.SCORE, 'test')
SURVIVE_INTENT = intent_for_profile(StrategyProfile.SURVIVE, 'test')


class TaskAgentTests(unittest.TestCase):
    def test_selects_nearest_equal_value_task_and_moves_pioneer(self) -> None:
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('far', 8, 8), task('near', 4, 1)),
        )

        result = TaskAgent().apply(observed, Decision(), SCORE_INTENT)

        self.assertEqual(result.state.active_task_type, 'near')
        self.assertEqual(result.decision.commands[2].kind, ActionKind.MOVE)
        self.assertEqual(result.decision.commands[2].target_positions, (Position(2, 0),))

    def test_adjacent_pioneer_accepts_selected_task(self) -> None:
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('math', 2, 2),),
        )

        result = TaskAgent().apply(observed, Decision(), SCORE_INTENT)

        self.assertEqual(result.decision.commands[2].kind, ActionKind.ACCEPT_TASK)
        self.assertEqual(result.state.active_task_type, 'math')
        self.assertEqual(result.state.accepted_round, 1)

    def test_survival_intent_interrupts_task_overlay(self) -> None:
        base = Decision(commands={2: Action.move(Position(1, 2))})
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('math', 2, 2),),
        )

        result = TaskAgent().apply(observed, base, SURVIVE_INTENT)

        self.assertEqual(result.decision, base)

    def test_active_task_prompts_without_executing_commands(self) -> None:
        base = Decision(
            commands={
                1: Action.move(Position(3, 3)),
                2: Action.move(Position(2, 2)),
            },
            execute_command='unsafe command',
        )
        observed = observation(
            our_units=(
                unit(1, 2, 3, 'worker'),
                unit(2, 1, 1, 'pioneer'),
            ),
            phase_task='Return the sum of 20 and 22.',
            world_news=WorldNews('official', 'legend'),
        )

        result = TaskAgent().apply(
            observed,
            base,
            SCORE_INTENT,
            previous_state=TaskAgentState(active_task_type='math'),
        )

        self.assertIn('20 and 22', result.decision.prompt)
        self.assertLessEqual(len(result.decision.prompt), 2048)
        self.assertEqual(set(result.decision.commands), {1})
        self.assertEqual(result.decision.execute_command, '')
        self.assertEqual(result.state.official_news, 'official')

    def test_successful_submission_becomes_exact_type_sop(self) -> None:
        agent = TaskAgent()
        active = TaskAgentState(active_task_type='math')
        answer_observation = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the sum of 20 and 22.',
            llm_response='  42  ',
        )

        submitted = agent.apply(
            answer_observation,
            Decision(),
            SCORE_INTENT,
            previous_state=active,
        )
        self.assertEqual(
            submitted.decision.commands[2].task_answer,
            '42',
        )

        feedback = observation(
            round_no=3,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            last_action_results={2: True},
        )
        learned = agent.apply(
            feedback,
            Decision(),
            SCORE_INTENT,
            previous_state=submitted.state,
        )
        self.assertEqual(learned.state.sops[0].task_type, 'math')
        self.assertEqual(learned.state.sops[0].answer, '42')

        repeated = agent.apply(
            replace(answer_observation, llm_response='', time=feedback.time),
            Decision(),
            SCORE_INTENT,
            previous_state=replace(
                learned.state,
                active_task_type='math',
            ),
        )
        self.assertEqual(repeated.decision.prompt, '')
        self.assertEqual(repeated.decision.commands[2].task_answer, '42')

    def test_failed_submission_is_not_learned_and_treasure_is_never_emitted(self) -> None:
        state = TaskAgentState(
            active_task_type='math',
            pending_task_type='math',
            pending_answer='bad',
            pending_pioneer_id=2,
        )
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            last_action_results={2: False},
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertEqual(result.state.sops, ())
        self.assertTrue(
            all(
                action.kind is not ActionKind.SUMMON_TREASURE
                for action in result.decision.commands.values()
            )
        )


if __name__ == '__main__':
    unittest.main()
