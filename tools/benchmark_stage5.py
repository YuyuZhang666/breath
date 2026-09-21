from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter_ns
from hashlib import sha256


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.policy import StrategyProfile, intent_for_profile
from future_war_agent.strategy.task_agent import TaskAgent, TaskAgentState
from future_war_agent.strategy.treasure import (
    TreasureAgent,
    TreasureCandidate,
    TreasureState,
)
from tests.strategy_helpers import observation, unit


ITERATIONS = 5_000
INTENT = intent_for_profile(StrategyProfile.SCORE, 'stage5 benchmark')


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * value))
    return ordered[index]


def benchmark(name: str, callback: Callable[[], object]) -> None:
    run = callback
    for _ in range(100):
        run()
    samples: list[float] = []
    for _ in range(ITERATIONS):
        started = perf_counter_ns()
        run()
        samples.append((perf_counter_ns() - started) / 1_000_000)
    print(
        f'{name}: P50={percentile(samples, 0.50):.4f}ms '
        f'P95={percentile(samples, 0.95):.4f}ms '
        f'P99={percentile(samples, 0.99):.4f}ms'
    )


def main() -> None:
    task_observation = observation(
        round_no=2,
        our_units=(unit(2, 1, 1, 'pioneer'),),
        phase_task='Return a detailed deterministic answer.',
        llm_response='a verified partial answer with all requested fields',
    )
    task_agent = TaskAgent()
    task_fingerprint = sha256(
        task_observation.phase_task.encode('utf-8')
    ).hexdigest()
    task_state = TaskAgentState(
        active_task_type='analysis',
        last_prompt_fingerprint=task_fingerprint,
        pending_prompt_fingerprint=task_fingerprint,
        pending_prompt_round=1,
    )
    benchmark(
        'task_anytime',
        lambda: task_agent.apply(
            task_observation,
            Decision(),
            INTENT,
            previous_state=task_state,
        ),
    )

    candidate = TreasureCandidate(
        position=Position(2, 2),
        items=('StarSand',),
        opening_days=(1,),
        confidence=0.9,
        evidence_revision=1,
        support_evidence=('benchmark',),
    )
    treasure_state = TreasureState(candidates=(candidate,))
    treasure_observation = observation(
        our_units=(
            unit(
                2,
                2,
                1,
                'pioneer',
                backpack=('StarSand',),
                backpack_capacity=10,
            ),
        ),
    )
    treasure_agent = TreasureAgent()
    benchmark(
        'treasure_attempt',
        lambda: treasure_agent.apply(
            treasure_observation,
            Decision(),
            INTENT,
            previous_state=treasure_state,
        ),
    )

    pending_state = replace(
        treasure_state,
        pending_candidate_key=candidate.key,
        pending_round=1,
        pending_day=1,
    )
    feedback = observation(round_no=2, last_summon_treasure_result=3)
    benchmark(
        'treasure_feedback',
        lambda: treasure_agent.reconcile(feedback, pending_state),
    )


if __name__ == '__main__':
    main()
