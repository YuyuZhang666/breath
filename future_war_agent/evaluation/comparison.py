from collections.abc import Callable, Iterable, Sequence
from time import perf_counter_ns

from .models import (
    EvaluationReport,
    ReplayCorpus,
    ReplayResult,
    ReplayVariant,
    VariantAggregate,
)
from .runner import NanosecondClock, run_case


ClockFactory = Callable[[], NanosecondClock]


def evaluate_variants(
    corpus: ReplayCorpus,
    variants: Sequence[ReplayVariant],
    *,
    clock_factory: ClockFactory = lambda: perf_counter_ns,
) -> EvaluationReport:
    if not variants:
        raise ValueError("at least one replay variant is required")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("duplicate variant name")

    results = tuple(
        run_case(variant, case, clock=clock_factory())
        for variant in sorted(variants, key=lambda item: item.name)
        for case in corpus.cases
    )
    aggregates = rank_aggregates(_aggregate(results))
    return EvaluationReport(
        results=results,
        aggregates=aggregates,
        ranking=tuple(item.variant_name for item in aggregates),
    )


def rank_aggregates(
    aggregates: Iterable[VariantAggregate],
) -> tuple[VariantAggregate, ...]:
    return tuple(
        sorted(
            aggregates,
            key=lambda item: (
                -item.league_points,
                -item.surviving_cases,
                -item.score_gain,
                item.max_p99_latency_ns,
                item.variant_name,
            ),
        )
    )


def _aggregate(results: tuple[ReplayResult, ...]) -> tuple[VariantAggregate, ...]:
    grouped: dict[str, list[ReplayResult]] = {}
    for result in results:
        grouped.setdefault(result.variant_name, []).append(result)

    return tuple(
        VariantAggregate(
            variant_name=name,
            case_count=len(items),
            league_points=sum(item.metrics.league_points for item in items),
            surviving_cases=sum(item.metrics.final_station_survived for item in items),
            score_gain=sum(item.metrics.score_gain for item in items),
            answer_submit_count=sum(
                item.metrics.answer_submit_count for item in items
            ),
            max_p99_latency_ns=max(
                item.metrics.p99_latency_ns for item in items
            ),
        )
        for name, items in sorted(grouped.items())
    )
