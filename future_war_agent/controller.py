import logging
from collections.abc import Callable
from inspect import Parameter, signature
from time import monotonic

from future_war_agent.deadline import RequestBudget, RequestDeadlineExceeded
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Planner = Callable[..., Decision]

RESPONSE_BUDGET_SECONDS = 3.5
COMPUTE_BUDGET_SECONDS = 3.0

DEFAULT_STRATEGY_ENGINE = StrategyEngine()
_WEAPON_TYPES = frozenset({'gatling', 'railgun', 'rocket'})
_WEAPON_LOG_LIMIT = 16


def default_planner(
    observation: Observation,
    *,
    request_started_at: float | None = None,
    request_budget: RequestBudget | None = None,
) -> Decision:
    if request_budget is not None:
        if _accepts_keyword(DEFAULT_STRATEGY_ENGINE.plan, 'request_budget'):
            return DEFAULT_STRATEGY_ENGINE.plan(
                observation,
                request_budget=request_budget,
            )
        request_started_at = request_budget.started_at
    if request_started_at is not None and _accepts_keyword(
        DEFAULT_STRATEGY_ENGINE.plan,
        'request_started_at',
    ):
        return DEFAULT_STRATEGY_ENGINE.plan(
            observation,
            request_started_at=request_started_at,
        )
    return DEFAULT_STRATEGY_ENGINE.plan(observation)


def handle_payload(
    payload: object,
    planner: Planner = default_planner,
    *,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
    request_started_at: float | None = None,
    clock: Callable[[], float] = monotonic,
) -> dict[str, object]:
    fallback_payload = safe_payload()
    request_budget = RequestBudget.start(
        started_at=request_started_at,
        response_budget_seconds=RESPONSE_BUDGET_SECONDS,
        compute_budget_seconds=COMPUTE_BUDGET_SECONDS,
        clock=clock,
    )
    token = telemetry.begin()
    telemetry.set(safe_action_generated=True)
    try:
        if request_budget.compute_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        with telemetry.measure('parser_ms'):
            observation = parse_observation(payload)
        telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
        if request_budget.compute_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        planner_kwargs: dict[str, object] = {}
        if _accepts_keyword(planner, 'request_budget'):
            planner_kwargs['request_budget'] = request_budget
        if _accepts_keyword(planner, 'request_started_at'):
            planner_kwargs['request_started_at'] = request_budget.started_at
        decision = planner(observation, **planner_kwargs)
        telemetry.set(pre_validation_action_count=len(decision.commands))
        if request_budget.compute_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        with telemetry.measure('validation_ms'):
            validated = validate_decision(observation, decision)
        telemetry.set(
            post_validation_action_count=len(validated.commands),
            validated_weapon_action_log=_validated_weapon_action_log(
                observation,
                validated,
            ),
        )
        if request_budget.response_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        with telemetry.measure('serialization_ms'):
            response = decision_to_payload(validated)
        serialized_actions = response.get('roleCommandMap', {})
        serialized_action_count = (
            len(serialized_actions)
            if isinstance(serialized_actions, dict)
            else 0
        )
        telemetry.set(serialized_action_count=serialized_action_count)
        if request_budget.response_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        telemetry.set(response_action_count=serialized_action_count)
        return response
    except RequestDeadlineExceeded:
        _record_deadline_fallback(telemetry)
        LOGGER.warning('turn handling reached the request deadline', exc_info=True)
        return fallback_payload
    except Exception:
        telemetry.set(
            fallback_used=True,
            fallback_reason='exception',
            decision_source='safe',
            response_action_count=0,
        )
        LOGGER.exception("turn handling failed")
        return fallback_payload
    finally:
        finished_at = clock()
        telemetry.set(
            request_total_ms=max(
                0.0,
                (finished_at - request_budget.started_at) * 1000,
            ),
            remaining_deadline_ms=max(
                0.0,
                request_budget.response_deadline - finished_at,
            ) * 1000,
        )
        telemetry.finish(token)


def _accepts_keyword(function: Planner, name: str) -> bool:
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _validated_weapon_action_log(
    observation: Observation,
    decision: Decision,
) -> tuple[str, ...]:
    entries: list[str] = []
    for weapon in sorted(observation.our.units, key=lambda item: item.unit_id):
        if weapon.role_type not in _WEAPON_TYPES:
            continue
        action = decision.commands.get(weapon.unit_id)
        final_action = action.kind.value if action is not None else 'none'
        entries.append(
            f'id={weapon.unit_id}:type={weapon.role_type}:'
            f'hp={weapon.health}:pos={weapon.position.x},{weapon.position.y}:'
            f'cooldown={weapon.cooldown}:final={final_action}'
        )
        if len(entries) >= _WEAPON_LOG_LIMIT:
            break
    return tuple(entries)


def _record_deadline_fallback(telemetry: TelemetryRecorder) -> None:
    telemetry.set(
        fallback_used=True,
        fallback_reason='deadline_low',
        timeout_prevented=True,
        decision_source='safe',
        response_action_count=0,
    )
