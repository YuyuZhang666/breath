import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, TaskPointState, WorldNews
from future_war_agent.strategy.engine import StrategyEngine
from tests.strategy_helpers import observation, unit


def defended_task_observation(
    round_no: int,
    *,
    tasks=(),
    phase_task: str = '',
    llm_response: str = '',
    last_action_results=None,
):
    return observation(
        round_no=round_no,
        our_units=(
            unit(10, 5, 5, 'station', health=1000, level=1),
            unit(11, 3, 3, 'gatling', level=1),
            unit(12, 4, 3, 'railgun', level=1),
            unit(13, 5, 3, 'rocket', level=1),
            unit(14, 1, 1, 'pioneer'),
        ),
        tasks=tasks,
        phase_task=phase_task,
        llm_response=llm_response,
        last_action_results=last_action_results,
        gold=200,
    )


MATH_TASK = TaskPointState(
    task_type='math',
    position=Position(2, 2),
    cooldown_rounds=0,
    score_reward=20,
    gold_reward=5,
    is_valid=True,
    timeout_rounds=10,
)


class BrokenTaskAgent:
    def apply(self, *args, **kwargs):
        raise RuntimeError('task failure')


class Phase5AcceptanceTests(unittest.TestCase):
    def test_real_engine_treasure_overrides_ordinary_preposition(self) -> None:
        observed = observation(
            round_no=1,
            our_units=(
                unit(10, 5, 5, 'station', health=1000, level=1),
                unit(11, 3, 3, 'gatling', level=1),
                unit(12, 4, 3, 'railgun', level=1),
                unit(13, 5, 3, 'rocket', level=1),
                unit(
                    14,
                    1,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
            gold=200,
            world_news=WorldNews(
                folk_legends='{"treasure":{"x":2,"y":2,"items":["StarSand"],"days":[1],"confidence":0.9}}'
            ),
        )

        decision = StrategyEngine().plan(observed)

        self.assertEqual(
            decision.commands[14].kind,
            ActionKind.SUMMON_TREASURE,
        )
        self.assertEqual(validate_decision(observed, decision), decision)

    def test_real_engine_learns_and_reuses_successful_sop(self) -> None:
        engine = StrategyEngine()

        accepted = engine.plan(
            defended_task_observation(1, tasks=(MATH_TASK,))
        )
        prompted = engine.plan(
            defended_task_observation(
                2,
                phase_task='Return the sum of 20 and 22.',
            )
        )
        submitted = engine.plan(
            defended_task_observation(
                3,
                phase_task='Return the sum of 20 and 22.',
                llm_response='42',
            )
        )
        engine.plan(
            defended_task_observation(4, last_action_results={14: True})
        )
        engine.plan(defended_task_observation(5, tasks=(MATH_TASK,)))
        reused = engine.plan(
            defended_task_observation(
                6,
                phase_task='Return the sum of 20 and 22.',
            )
        )

        self.assertEqual(accepted.commands[14].kind, ActionKind.ACCEPT_TASK)
        self.assertIn('20 and 22', prompted.prompt)
        self.assertEqual(submitted.commands[14].task_answer, '42')
        self.assertEqual(reused.prompt, '')
        self.assertEqual(reused.commands[14].task_answer, '42')
        for observed, decision in (
            (defended_task_observation(1, tasks=(MATH_TASK,)), accepted),
            (
                defended_task_observation(
                    3,
                    phase_task='Return the sum of 20 and 22.',
                    llm_response='42',
                ),
                submitted,
            ),
        ):
            self.assertEqual(validate_decision(observed, decision), decision)

    def test_task_failure_returns_base_decision_atomically(self) -> None:
        base = Decision(prompt='base')
        engine = StrategyEngine(
            phase2_planner=lambda _: base,
            task_agent=BrokenTaskAgent(),
        )
        observed = defended_task_observation(1, tasks=(MATH_TASK,))

        decision = engine.plan(observed)

        self.assertIs(decision, base)
        self.assertEqual(engine._sessions.get('team').task_state.sops, ())

    def test_duplicate_active_task_response_is_byte_equivalent(self) -> None:
        engine = StrategyEngine()
        engine.plan(defended_task_observation(1, tasks=(MATH_TASK,)))
        active = defended_task_observation(
            2,
            phase_task='Return the sum of 20 and 22.',
        )

        first = engine.plan(active)
        duplicate = engine.plan(active)

        self.assertIs(first, duplicate)
        self.assertEqual(first.prompt, duplicate.prompt)


if __name__ == '__main__':
    unittest.main()
