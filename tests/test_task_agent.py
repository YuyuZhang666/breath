import unittest
from dataclasses import replace
from hashlib import sha256

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import (
    GameError,
    Position,
    TaskPointState,
    WorldNews,
)
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.policy import (
    RuleFeatureFlags,
    StrategyProfile,
    intent_for_profile,
)
from future_war_agent.strategy.task_agent import (
    CommandResultKind,
    MAX_TASK_TYPE_LENGTH,
    TaskAbandonPolicy,
    TaskAgent,
    TaskAgentState,
    TaskSop,
    parse_command_result,
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
ECONOMY_INTENT = intent_for_profile(StrategyProfile.ECONOMY, 'test')
SURVIVE_INTENT = intent_for_profile(StrategyProfile.SURVIVE, 'test')


def prompted_state(task_type: str, task_text: str) -> TaskAgentState:
    fingerprint = sha256(task_text.encode('utf-8')).hexdigest()
    return TaskAgentState(
        active_task_type=task_type,
        last_prompt_fingerprint=fingerprint,
        pending_prompt_fingerprint=fingerprint,
        pending_prompt_round=1,
    )


class TaskAgentTests(unittest.TestCase):
    def test_command_result_markers_are_classified(self) -> None:
        self.assertEqual(
            parse_command_result('[exitCode:0]\nok').kind,
            CommandResultKind.SUCCESS,
        )
        self.assertEqual(
            parse_command_result('[exitCode:2]\nfailed').kind,
            CommandResultKind.ERROR,
        )
        self.assertEqual(
            parse_command_result('[TIMEOUT]').kind,
            CommandResultKind.TIMEOUT,
        )
        self.assertEqual(
            parse_command_result('[JUDGER_ERROR]').kind,
            CommandResultKind.JUDGER_ERROR,
        )
        self.assertTrue(parse_command_result('[TRUNCATED]').truncated)

    def test_command_result_preserves_truncation_marker_after_large_output(
        self,
    ) -> None:
        raw = '[exitCode:0]\n' + ('x' * 65_536) + '\n[TRUNCATED]'

        result = parse_command_result(raw)

        self.assertEqual(result.kind, CommandResultKind.SUCCESS)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.output), 32_768)
        self.assertTrue(result.output.endswith('[TRUNCATED]'))

    def test_command_state_reconciliation_keeps_large_output_trailer(
        self,
    ) -> None:
        raw = '[exitCode:0]\n' + ('x' * 65_536) + '\n[TRUNCATED]'
        observed = observation(
            round_no=2,
            last_command_result=raw,
        )

        state = TaskAgent().reconcile(
            observed,
            TaskAgentState(pending_command_round=1),
        )

        self.assertEqual(state.last_command_kind, CommandResultKind.SUCCESS)
        self.assertLessEqual(len(state.last_command_result), 32_768)
        self.assertTrue(state.last_command_result.endswith('[TRUNCATED]'))

    def test_anytime_does_not_resubmit_identical_llm_answer(self) -> None:
        agent = TaskAgent()
        task_text = 'Return a useful partial answer.'
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
            llm_response='partial',
        )
        first = agent.apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=prompted_state('analysis', task_text),
        )
        second = agent.apply(
            replace(
                observed,
                time=TurnTime.from_round(3),
                last_action_results={2: False},
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=first.state,
        )

        self.assertEqual(first.decision.commands[2].task_answer, 'partial')
        self.assertNotIn(2, second.decision.commands)
        self.assertEqual(len(second.state.submitted_answer_fingerprints), 1)

    def test_anytime_submits_a_distinct_better_answer(self) -> None:
        agent = TaskAgent()
        task_text = 'Return a detailed answer.'
        first_observation = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
            llm_response='partial',
        )
        first = agent.apply(
            first_observation,
            Decision(),
            SCORE_INTENT,
            previous_state=prompted_state('analysis', task_text),
        )
        improved = agent.apply(
            replace(
                first_observation,
                time=TurnTime.from_round(3),
                llm_response='partial with verified detail',
                last_action_results={2: False},
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=first.state,
        )

        self.assertEqual(
            improved.decision.commands[2].task_answer,
            'partial with verified detail',
        )
        self.assertEqual(len(improved.state.submitted_answer_fingerprints), 2)

    def test_explicitly_failed_long_answer_does_not_block_short_answer(self) -> None:
        agent = TaskAgent()
        task_text = 'Return the exact integer only.'
        first_observation = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
            llm_response='a long but incorrect speculative answer',
        )
        first = agent.apply(
            first_observation,
            Decision(),
            SCORE_INTENT,
            previous_state=prompted_state('math', task_text),
        )
        corrected = agent.apply(
            replace(
                first_observation,
                time=TurnTime.from_round(3),
                llm_response='42',
                last_action_results={2: False},
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=first.state,
        )

        self.assertEqual(corrected.decision.commands[2].task_answer, '42')

    def test_pending_prompt_is_not_reissued_every_round(self) -> None:
        agent = TaskAgent(prompt_retry_rounds=2)
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return a concise answer.',
        )
        first = agent.apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=TaskAgentState(active_task_type='analysis'),
        )
        waiting = agent.apply(
            replace(observed, time=TurnTime.from_round(3)),
            Decision(),
            SCORE_INTENT,
            previous_state=first.state,
        )

        self.assertTrue(first.decision.prompt)
        self.assertEqual(waiting.decision.prompt, '')

    def test_llm_response_from_changed_task_is_ignored(self) -> None:
        state = TaskAgentState(
            active_task_type='analysis',
            last_prompt_fingerprint='old-task-fingerprint',
            pending_prompt_fingerprint='old-task-fingerprint',
            pending_prompt_round=1,
        )
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='A different task.',
            llm_response='stale answer',
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertNotEqual(result.state.best_answer, 'stale answer')

    def test_parameterized_sop_never_reuses_old_answer_for_changed_task(self) -> None:
        task_text = 'Return the sum of 40 and 2.'
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
        )
        state = TaskAgentState(
            active_task_type='math',
            sops=(
                TaskSop(
                    'math',
                    '42',
                    task_fingerprint='not-the-current-task',
                    task_template='Return the sum of <n> and <n>.',
                    sample_task='Return the sum of 20 and 22.',
                ),
            ),
        )

        result = TaskAgent(execute_commands=('probe',)).apply(
            observed,
            Decision(),
            intent_for_profile(
                StrategyProfile.SCORE,
                'sandbox enabled',
                feature_flags=RuleFeatureFlags(enable_task_execute_commands=True),
            ),
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertEqual(result.decision.execute_command, '')
        self.assertIn('20 and 22', result.decision.prompt)

    def test_legacy_sop_without_task_identity_is_not_directly_submitted(self) -> None:
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the sum of 40 and 2.',
        )
        state = TaskAgentState(
            active_task_type='math',
            sops=(TaskSop('math', 'stale-answer'),),
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertTrue(result.decision.prompt)

    def test_parameterized_sop_can_fall_back_to_next_sandbox_step(self) -> None:
        task_text = 'Return the sum of 40 and 2.'
        template = 'Return the sum of <n> and <n>.'
        state = TaskAgentState(
            active_task_type='math',
            sops=(
                TaskSop(
                    'math',
                    '42',
                    task_fingerprint='old',
                    task_template=template,
                    sample_task='Return the sum of 20 and 22.',
                ),
            ),
        )
        enabled = intent_for_profile(
            StrategyProfile.SCORE,
            'sandbox enabled',
            feature_flags=RuleFeatureFlags(enable_task_execute_commands=True),
        )
        agent = TaskAgent(execute_commands=('probe',), prompt_retry_rounds=1)
        first = agent.apply(
            observation(
                round_no=2,
                our_units=(unit(2, 1, 1, 'pioneer'),),
                phase_task=task_text,
            ),
            Decision(),
            enabled,
            previous_state=state,
        )
        fallback = agent.apply(
            observation(
                round_no=4,
                our_units=(unit(2, 1, 1, 'pioneer'),),
                phase_task=task_text,
            ),
            Decision(),
            enabled,
            previous_state=first.state,
        )

        self.assertEqual(first.decision.execute_command, '')
        self.assertEqual(fallback.decision.execute_command, 'probe')

    def test_reconcile_consumes_feedback_without_planning_new_work(self) -> None:
        state = TaskAgentState(
            active_task_type='math',
            pending_task_type='math',
            pending_answer='42',
            pending_pioneer_id=2,
            pending_round=2,
            pending_task_text='Return the sum of 20 and 22.',
        )
        observed = observation(
            round_no=3,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            last_action_results={2: True},
        )

        reconciled = TaskAgent().reconcile(observed, state)

        self.assertEqual(reconciled.sops[0].answer, '42')
        self.assertEqual(reconciled.active_task_type, '')

    def test_legal_but_wrong_answer_is_not_learned_as_sop(self) -> None:
        state = TaskAgentState(
            active_task_type='math',
            pending_task_type='math',
            pending_answer='41',
            pending_pioneer_id=2,
            pending_round=2,
            pending_task_text='Return the sum of 20 and 22.',
        )
        observed = replace(
            observation(
                round_no=3,
                our_units=(unit(2, 1, 1, 'pioneer'),),
                last_action_results={2: True},
            ),
            errors=(GameError(2, 'answer incorrect'),),
        )

        reconciled = TaskAgent().reconcile(observed, state)

        self.assertEqual(reconciled.sops, ())
        self.assertEqual(reconciled.completed_task_fingerprint, '')

    def test_equal_reward_task_prefers_faster_validated_sop_type(
        self,
    ) -> None:
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(
                task('slow', 2, 1, score=50, timeout=20),
                task('fast', 1, 2, score=50, timeout=20),
            ),
        )
        state = TaskAgentState(
            sops=(
                TaskSop(
                    'slow',
                    'answer',
                    solve_rounds_ewma=8.0,
                ),
                TaskSop(
                    'fast',
                    'answer',
                    solve_rounds_ewma=2.0,
                ),
            ),
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertEqual(result.state.active_task_type, 'fast')
        self.assertEqual(
            result.decision.commands[2].kind,
            ActionKind.ACCEPT_TASK,
        )

    def test_success_feedback_does_not_resubmit_if_phase_task_lingers(self) -> None:
        task_text = 'Return the sum of 20 and 22.'
        state = TaskAgentState(
            active_task_type='math',
            pending_task_type='math',
            pending_answer='42',
            pending_pioneer_id=2,
            pending_round=2,
            pending_task_text=task_text,
        )
        observed = observation(
            round_no=3,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
            llm_response='42',
            last_action_results={2: True},
        )

        result = TaskAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertTrue(result.state.completed_task_fingerprint)

    def test_task_abandon_policy_is_margin_guarded(self) -> None:
        policy = TaskAbandonPolicy(
            margin=100,
            cooldown_rounds=30,
            cooldown_cost_per_round=20,
        )

        self.assertFalse(
            policy.should_abandon(
                continue_value=1_000,
                alternative_value=1_399,
                remaining_rounds=10,
                survival_risk=0,
            )
        )
        self.assertTrue(
            policy.should_abandon(
                continue_value=100,
                alternative_value=10_000,
                remaining_rounds=1,
                survival_risk=100,
            )
        )

    def test_abandoned_task_stays_suppressed_until_cooldown_expires(self) -> None:
        agent = TaskAgent()
        state = TaskAgentState(
            active_task_type='current',
            selected_task_position=Position(2, 2),
            selected_task_value=10,
            timeout_rounds=1,
        )
        abandoned = agent.apply(
            observation(
                round_no=2,
                our_units=(unit(2, 1, 1, 'pioneer'),),
                tasks=(task('current', 2, 2), task('better', 3, 3, score=100)),
                phase_task='Current task payload.',
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )
        next_round = agent.apply(
            observation(
                round_no=3,
                our_units=(unit(2, 1, 1, 'pioneer'),),
                tasks=(task('current', 2, 2),),
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=abandoned.state,
        )

        self.assertEqual(next_round.decision.commands, {})
        self.assertEqual(next_round.state.abandoned_task_type, 'current')

    def test_economy_intent_accepts_available_task(self) -> None:
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            tasks=(task('economy-task', 2, 2),),
        )

        result = TaskAgent().apply(observed, Decision(), ECONOMY_INTENT)

        self.assertEqual(result.decision.commands[2].kind, ActionKind.ACCEPT_TASK)
        self.assertEqual(result.state.active_task_type, 'economy-task')

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

    def test_survival_intent_does_not_move_pioneer_out_of_active_task(self) -> None:
        base = Decision(commands={2: Action.move(Position(1, 2))})
        observed = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the exact answer.',
        )

        result = TaskAgent().apply(
            observed,
            base,
            SURVIVE_INTENT,
            previous_state=TaskAgentState(active_task_type='analysis'),
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertTrue(result.decision.prompt)

    def test_explicit_survival_interrupt_pauses_active_task_for_one_turn(
        self,
    ) -> None:
        base = Decision(commands={2: Action.move(Position(1, 2))})
        observed = observation(
            round_no=71,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Return the exact answer.',
        )

        result = TaskAgent().apply(
            observed,
            base,
            SURVIVE_INTENT,
            previous_state=TaskAgentState(active_task_type='analysis'),
            force_survival_interrupt=True,
        )

        self.assertEqual(result.decision, base)
        self.assertEqual(result.state.active_task_type, 'analysis')
        self.assertEqual(result.state.abandoned_task_fingerprint, '')
        self.assertIsNone(result.state.abandon_until_round)

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

    def test_execute_command_flow_requires_flag_and_consumes_next_round_result(
        self,
    ) -> None:
        enabled = intent_for_profile(
            StrategyProfile.ECONOMY,
            'sandbox verified',
            feature_flags=RuleFeatureFlags(enable_task_execute_commands=True),
        )
        agent = TaskAgent(execute_commands=('probe-one', 'probe-two'))
        active = TaskAgentState(active_task_type='code')
        first_observation = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Inspect the provided task files.',
        )

        first = agent.apply(
            first_observation,
            Decision(),
            enabled,
            previous_state=active,
        )
        second_observation = replace(
            first_observation,
            time=TurnTime.from_round(2),
            last_command_result='task.md found',
        )
        second = agent.apply(
            second_observation,
            Decision(),
            enabled,
            previous_state=first.state,
        )

        self.assertEqual(first.decision.execute_command, 'probe-one')
        self.assertEqual(second.decision.execute_command, 'probe-two')
        self.assertIn('task.md found', second.decision.prompt)
        self.assertEqual(second.state.command_step, 2)
        self.assertEqual(second.state.pending_command_round, 2)

    def test_execute_commands_remain_off_by_default(self) -> None:
        agent = TaskAgent(execute_commands=('probe-one',))
        observed = observation(
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task='Inspect the provided task files.',
        )

        result = agent.apply(
            observed,
            Decision(),
            ECONOMY_INTENT,
            previous_state=TaskAgentState(active_task_type='code'),
        )

        self.assertEqual(result.decision.execute_command, '')

    def test_successful_submission_becomes_exact_type_sop(self) -> None:
        agent = TaskAgent()
        task_text = 'Return the sum of 20 and 22.'
        active = prompted_state('math', task_text)
        answer_observation = observation(
            round_no=2,
            our_units=(unit(2, 1, 1, 'pioneer'),),
            phase_task=task_text,
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
