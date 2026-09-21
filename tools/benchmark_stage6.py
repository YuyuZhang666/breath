from collections.abc import Callable
from pathlib import Path
import sys
from time import perf_counter_ns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityStatus,
)
from future_war_agent.strategy.memory import update_match_memory
from tests.strategy_helpers import observation, robot, unit


ITERATIONS = 5_000


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * value))
    return ordered[index]


def benchmark(name: str, callback: Callable[[], object]) -> None:
    for _ in range(100):
        callback()
    samples: list[float] = []
    for _ in range(ITERATIONS):
        started = perf_counter_ns()
        callback()
        samples.append((perf_counter_ns() - started) / 1_000_000)
    print(
        f'{name}: P50={percentile(samples, 0.50):.4f}ms '
        f'P95={percentile(samples, 0.95):.4f}ms '
        f'P99={percentile(samples, 0.99):.4f}ms'
    )


def main() -> None:
    matrix = CapabilityMatrix.from_manual_config(
        {Capability.ATTACK_ENEMY_STATION: CapabilityStatus.SUPPORTED}
    )
    first = observation(
        round_no=1,
        our_units=(
            unit(10, 12, 12, 'station', health=1000, level=1),
            unit(11, 10, 10, 'worker', health=100),
        ),
        enemy_units=(
            unit(90, 2, 2, 'station', health=1000, level=1),
            unit(91, 3, 2, 'wall', health=300),
            unit(92, 4, 2, 'rocket', health=200, level=2),
            unit(93, 5, 5, 'pioneer', health=100),
        ),
        zones=(Zone(Position(6, 5), 'challengerTaskPoint2'),),
    )
    second = observation(
        round_no=2,
        our_units=(
            unit(10, 12, 12, 'station', health=950, level=1),
            unit(11, 10, 10, 'worker', health=90),
        ),
        enemy_units=(
            unit(90, 2, 2, 'station', health=1000, level=1),
            unit(91, 3, 2, 'wall', health=300),
            unit(92, 4, 2, 'rocket', health=200, level=2),
            unit(93, 5, 5, 'pioneer', health=100),
            unit(94, 3, 3, 'gatling', health=200, level=1),
        ),
        robots=(robot(501, 8, 8, target_team='challenger'),),
        zones=(Zone(Position(6, 5), 'challengerTaskPoint2'),),
    )
    previous = update_match_memory(
        None,
        first,
        capability_matrix=matrix,
    )

    benchmark(
        'match_memory_update',
        lambda: update_match_memory(
            previous,
            second,
            capability_matrix=matrix,
        ),
    )
    updated = update_match_memory(
        previous,
        second,
        capability_matrix=matrix,
    )
    opponent = updated.opponent_memory
    benchmark(
        'opponent_capability_log',
        lambda: (
            opponent.structure_log_entries(),
            opponent.role_log_entries(),
            opponent.evidence_log_entries(),
            matrix.log_entries(),
        ),
    )


if __name__ == '__main__':
    main()
