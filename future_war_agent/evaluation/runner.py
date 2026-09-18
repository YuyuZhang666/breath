import hashlib
import json
from collections import Counter
from collections.abc import Callable
from math import ceil
from time import perf_counter_ns
from typing import Any

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.parser import parse_observation

from .errors import ReplayEvaluationError
from .models import ReplayCase, ReplayMetrics, ReplayResult, ReplayVariant


NanosecondClock = Callable[[], int]


def run_case(
    variant: ReplayVariant,
    case: ReplayCase,
    *,
    clock: NanosecondClock = perf_counter_ns,
) -> ReplayResult:
    planner = variant.planner_factory()
    digests: list[str] = []
    profiles: list[str | None] = []
    elapsed_values: list[int] = []
    scores: list[int] = []
    station_healths: list[int] = []
    action_counts: Counter[str] = Counter()
    command_count = 0
    prompt_count = 0
    execute_command_count = 0

    for turn_index, raw_turn in enumerate(case.turns, start=1):
        try:
            started = clock()
            observation = parse_observation(raw_turn)
            decision = _plan(planner, observation)
            validated = validate_decision(observation, decision)
            payload = decision_to_payload(validated)
            profile = _read_profile(variant, planner, observation)
            elapsed = clock() - started
            if elapsed < 0:
                raise ValueError("monotonic clock moved backwards")
        except Exception as error:
            raise ReplayEvaluationError(
                f"variant {variant.name!r}, case {case.name!r}, "
                f"turn {turn_index}: {error}"
            ) from error

        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digests.append(hashlib.sha256(canonical).hexdigest())
        profiles.append(profile)
        elapsed_values.append(elapsed)
        scores.append(observation.our.total_score)
        station_healths.append(_station_health(observation))

        command_count += len(validated.commands)
        prompt_count += bool(validated.prompt.strip())
        execute_command_count += bool(validated.execute_command.strip())
        action_counts.update(action.kind.value for action in validated.commands.values())

    metrics = ReplayMetrics(
        league_points=case.outcome.league_points,
        initial_score=scores[0],
        final_score=scores[-1],
        score_gain=scores[-1] - scores[0],
        final_station_survived=station_healths[-1] > 0,
        minimum_station_health=min(station_healths),
        command_count=command_count,
        prompt_count=prompt_count,
        execute_command_count=execute_command_count,
        task_accept_count=action_counts[ActionKind.ACCEPT_TASK.value],
        answer_submit_count=action_counts[ActionKind.SUBMIT_ANSWER.value],
        action_kind_counts=tuple(sorted(action_counts.items())),
        profile_switch_count=_profile_switches(profiles),
        median_latency_ns=_median(elapsed_values),
        p99_latency_ns=_nearest_rank(elapsed_values, 0.99),
    )
    return ReplayResult(
        variant_name=variant.name,
        case_name=case.name,
        outcome=case.outcome,
        response_digests=tuple(digests),
        profile_labels=tuple(profiles),
        elapsed_ns=tuple(elapsed_values),
        metrics=metrics,
    )


def _plan(planner: object, observation: Observation) -> Decision:
    plan = getattr(planner, "plan", None)
    if callable(plan):
        decision = plan(observation)
    elif callable(planner):
        decision = planner(observation)
    else:
        raise TypeError("planner must be callable or expose plan(observation)")
    if not isinstance(decision, Decision):
        raise TypeError("planner must return Decision")
    return decision


def _read_profile(
    variant: ReplayVariant,
    planner: object,
    observation: Observation,
) -> str | None:
    if variant.profile_reader is None:
        return None
    raw = variant.profile_reader(planner, observation)
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    label = str(value).strip()
    return label or None


def _station_health(observation: Observation) -> int:
    station = next(
        (unit for unit in observation.our.units if unit.role_type == "station"),
        None,
    )
    return station.health if station is not None else 0


def _profile_switches(profiles: list[str | None]) -> int:
    known = [profile for profile in profiles if profile is not None]
    return sum(current != previous for previous, current in zip(known, known[1:]))


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]
