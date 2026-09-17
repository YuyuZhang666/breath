import unittest
from dataclasses import replace

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position, TaskPointState, WorldNews
from future_war_agent.strategy.policy import StrategyProfile, intent_for_profile
from future_war_agent.strategy.task_agent import (
    MAX_TASK_TYPE_LENGTH,
    TaskAgent,
    TaskAgentState,
    TaskSop,
)
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

    def test_task_never_overrides_emergency_item_use(self) -> None:
        base = Decision(commands={2: Action.use('Medicine')})
        observed = observation(
            our_units=(
                unit(2, 1, 1, 'pioneer', health=20, backpack=('Medicine',)),
            ),
            tasks=(task('math', 2, 2),),
        )

        result = TaskAgent().apply(observed, base, SCORE_INTENT)

        self.assertEqual(result.decision, base)
        self.assertEqual(result.state.active_task_type, '')

    def test_task_never_overrides_twilight_recall_move(self) -> None:
        base = Decision(commands={2: Action.move(Position(2, 2))})
        observed = observation(
            round_no=70,
            our_units=(
                unit(2, 1, 1, 'pioneer'),
                unit(11, 5, 5, 'gatling', level=1),
            ),
            tasks=(task('math', 2, 1),),
        )

        result = TaskAgent().apply(observed, base, SCORE_INTENT)

        self.assertEqual(result.decision, base)
        self.assertEqual(result.state.active_task_type, '')

    def test_task_move_never_conflicts_with_base_build_target(self) -> None:
        base = Decision(commands={1: Action.build('wall', Position(2, 0))})
        observed = observation(
            our_units=(
                unit(1, 3, 0, 'worker', backpack=('stone',)),
                unit(2, 1, 1, 'pioneer'),
            ),
            tasks=(task('math', 4, 1),),
        )

        result = TaskAgent().apply(observed, base, SCORE_INTENT)

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

    def test_missing_immediate_feedback_expires_pending_submission(self) -> None:
        pending = TaskAgentState(
            active_task_type='math',
            pending_task_type='math',
            pending_answer='42',
            pending_pioneer_id=2,
            pending_round=1,
        )
        missing = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
        )
        late_success = observation(
            round_no=3,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            last_action_results={2: True},
        )

        expired = TaskAgent().apply(
            missing,
            Decision(),
            SCORE_INTENT,
            previous_state=pending,
        )
        result = TaskAgent().apply(
            late_success,
            Decision(),
            SCORE_INTENT,
            previous_state=expired.state,
        )

        self.assertEqual(result.state.sops, ())
        self.assertIsNone(result.state.pending_pioneer_id)

    def test_selected_task_is_sticky_until_acceptance(self) -> None:
        agent = TaskAgent()
        first = observation(
            round_no=1,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('chosen', 6, 1),),
        )
        selected = agent.apply(first, Decision(), SCORE_INTENT)
        moved_pioneer = unit(2, 2, 0, 'pioneer')
        second = observation(
            round_no=2,
            our_units=(moved_pioneer,),
            tasks=(
                task('chosen', 6, 1),
                task('new-high-value', 3, 1, score=1000),
            ),
        )

        continued = agent.apply(
            second,
            Decision(),
            SCORE_INTENT,
            previous_state=selected.state,
        )

        self.assertEqual(continued.state.active_task_type, 'chosen')
        self.assertEqual(continued.state.selected_task_position, Position(6, 1))
        self.assertIsNone(continued.state.accepted_round)

    def test_deadline_submits_bounded_best_answer_and_tracks_prompt(self) -> None:
        agent = TaskAgent()
        state = TaskAgentState(
            active_task_type='math',
            deadline_round=2,
            best_answer='partial-42',
        )
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the sum of 20 and 22.',
        )

        result = agent.apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertEqual(result.decision.commands[2].task_answer, 'partial-42')
        self.assertTrue(result.state.last_prompt_fingerprint)

    def test_missing_feedback_keeps_best_answer_for_deadline_retry(self) -> None:
        state = TaskAgentState(
            active_task_type='math',
            accepted_round=1,
            deadline_round=3,
            best_answer='42',
            pending_task_type='math',
            pending_answer='42',
            pending_pioneer_id=2,
            pending_round=2,
        )
        observed = observation(
            round_no=3,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the sum of 20 and 22.',
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertEqual(result.decision.commands[2].task_answer, '42')

    def test_missing_phase_task_releases_failed_acceptance_next_round(self) -> None:
        state = TaskAgentState(
            active_task_type='old',
            selected_task_position=Position(2, 2),
            accepted_round=1,
            deadline_round=10,
        )
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('new', 2, 2),),
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertEqual(result.state.active_task_type, 'new')
        self.assertEqual(result.decision.commands[2].kind, ActionKind.ACCEPT_TASK)

    def test_task_type_and_sop_fields_are_bounded(self) -> None:
        long_type = 'x' * (MAX_TASK_TYPE_LENGTH + 50)
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task(long_type, 2, 2),),
        )

        result = TaskAgent().apply(observed, Decision(), SCORE_INTENT)

        self.assertEqual(len(result.state.active_task_type), MAX_TASK_TYPE_LENGTH)
        with self.assertRaises(ValueError):
            TaskSop(long_type, 'answer')

    def test_long_task_types_with_same_prefix_keep_distinct_identity(self) -> None:
        prefix = 'x' * (MAX_TASK_TYPE_LENGTH + 20)
        first = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task(prefix + 'A', 2, 2),),
        )
        second = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task(prefix + 'B', 2, 2),),
        )

        first_state = TaskAgent().apply(first, Decision(), SCORE_INTENT).state
        second_state = TaskAgent().apply(second, Decision(), SCORE_INTENT).state

        self.assertNotEqual(
            first_state.active_task_type,
            second_state.active_task_type,
        )
        self.assertEqual(len(first_state.active_task_type), MAX_TASK_TYPE_LENGTH)

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
