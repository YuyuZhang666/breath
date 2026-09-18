import json

from .models import EvaluationReport, ReplayMetrics, ReplayResult, VariantAggregate


def report_to_data(report: EvaluationReport) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ranking": list(report.ranking),
        "aggregates": [_aggregate_to_data(item) for item in report.aggregates],
        "results": [_result_to_data(item) for item in report.results],
    }


def report_to_json(report: EvaluationReport) -> str:
    return json.dumps(
        report_to_data(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _aggregate_to_data(value: VariantAggregate) -> dict[str, object]:
    return {
        "variant_name": value.variant_name,
        "case_count": value.case_count,
        "league_points": value.league_points,
        "surviving_cases": value.surviving_cases,
        "score_gain": value.score_gain,
        "answer_submit_count": value.answer_submit_count,
        "max_p99_latency_ns": value.max_p99_latency_ns,
    }


def _result_to_data(value: ReplayResult) -> dict[str, object]:
    return {
        "variant_name": value.variant_name,
        "case_name": value.case_name,
        "outcome": value.outcome.value,
        "response_digests": list(value.response_digests),
        "profile_labels": list(value.profile_labels),
        "elapsed_ns": list(value.elapsed_ns),
        "metrics": _metrics_to_data(value.metrics),
    }


def _metrics_to_data(value: ReplayMetrics) -> dict[str, object]:
    return {
        "league_points": value.league_points,
        "initial_score": value.initial_score,
        "final_score": value.final_score,
        "score_gain": value.score_gain,
        "final_station_survived": value.final_station_survived,
        "minimum_station_health": value.minimum_station_health,
        "command_count": value.command_count,
        "prompt_count": value.prompt_count,
        "execute_command_count": value.execute_command_count,
        "task_accept_count": value.task_accept_count,
        "answer_submit_count": value.answer_submit_count,
        "action_kind_counts": [list(item) for item in value.action_kind_counts],
        "profile_switch_count": value.profile_switch_count,
        "median_latency_ns": value.median_latency_ns,
        "p99_latency_ns": value.p99_latency_ns,
    }
