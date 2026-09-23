import logging
from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path
from time import monotonic

from future_war_agent.deadline import RequestBudget, RequestDeadlineExceeded
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.task_agent import TaskAgent, TaskSopStore
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Planner = Callable[..., Decision]

RESPONSE_BUDGET_SECONDS = 3.5
COMPUTE_BUDGET_SECONDS = 3.0

_SOP_PATH = Path(__file__).resolve().parents[1] / 'task_sops.json'
DEFAULT_STRATEGY_ENGINE = StrategyEngine(
    task_agent=TaskAgent(
        sop_store=TaskSopStore(_SOP_PATH),
        commands_enabled=True,
    ),
)
_WEAPON_TYPES = frozenset({'gatling', 'railgun', 'rocket'})
_WEAPON_LOG_LIMIT = 16
_ACTION_LOG_LIMIT = 32
_UNKNOWN_UNIT_TYPE = 'unknown'
_ACTION_DETAIL_SEPARATOR = ';'


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
        telemetry.set(
            day_no=observation.time.day_no,
            round_in_phase=observation.time.round_in_phase,
        )
        if request_budget.compute_exhausted():
            _record_deadline_fallback(telemetry, decision)
            return fallback_payload
        planner_kwargs: dict[str, object] = {}
        if _accepts_keyword(planner, 'request_budget'):
            planner_kwargs['request_budget'] = request_budget
        if _accepts_keyword(planner, 'request_started_at'):
            planner_kwargs['request_started_at'] = request_budget.started_at
        decision = planner(observation, **planner_kwargs)
        telemetry.set(
            pre_validation_action_count=len(decision.commands),
            planned_action_log=_decision_action_log(observation, decision),
        )
        if request_budget.compute_exhausted():
            _record_deadline_fallback(telemetry)
            return fallback_payload
        with telemetry.measure('validation_ms'):
            validated = validate_decision(observation, decision)
        telemetry.set(
            post_validation_action_count=len(validated.commands),
            validated_action_log=_decision_action_log(
                observation,
                validated,
            ),
            validation_drop_log=_validation_drop_log(
                observation,
                decision,
                validated,
            ),
            action_override_log=(
                tuple(telemetry.current('action_override_log', ()))
                + _decision_override_log(
                    decision,
                    validated,
                    module='validator',
                    reason='legality_validation',
                )
            ),
            opening_final_action_log=_opening_final_action_log(
                observation,
                validated,
            ),
            validated_weapon_action_log=_validated_weapon_action_log(
                observation,
                validated,
            ),
        )
        if request_budget.response_exhausted():
            _record_deadline_fallback(telemetry, validated)
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
            _record_deadline_fallback(telemetry, validated)
            return fallback_payload
        telemetry.set(
            response_action_count=serialized_action_count,
            action_lifecycle_log=_action_lifecycle_log(
                observation,
                decision,
                validated,
                serialized_actions,
            ),
        )
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


def _decision_action_log(
    observation: Observation,
    decision: Decision,
) -> tuple[str, ...]:
    unit_types = {
        unit.unit_id: unit.role_type for unit in observation.our.units
    }
    return tuple(
        f'id={actor_id}:type={unit_types.get(actor_id, _UNKNOWN_UNIT_TYPE)}:'
        f'action={_action_summary(action)}'
        for actor_id, action in sorted(decision.commands.items())[
            :_ACTION_LOG_LIMIT
        ]
    )


def _validation_drop_log(
    observation: Observation,
    planned: Decision,
    validated: Decision,
) -> tuple[str, ...]:
    unit_types = {
        unit.unit_id: unit.role_type for unit in observation.our.units
    }
    entries: list[str] = []
    for actor_id, action in sorted(planned.commands.items()):
        final = validated.commands.get(actor_id)
        if final == action:
            continue
        result = 'dropped' if final is None else _action_summary(final)
        entries.append(
            f'id={actor_id}:type={unit_types.get(actor_id, _UNKNOWN_UNIT_TYPE)}:'
            f'planned={_action_summary(action)}:validated={result}'
        )
        if len(entries) >= _ACTION_LOG_LIMIT:
            break
    return tuple(entries)


def _decision_override_log(
    original: Decision,
    final: Decision,
    *,
    module: str,
    reason: str,
) -> tuple[str, ...]:
    entries: list[str] = []
    actor_ids = sorted(set(original.commands) | set(final.commands))
    for actor_id in actor_ids:
        before = original.commands.get(actor_id)
        after = final.commands.get(actor_id)
        if before == after:
            continue
        entries.append(
            f'ACTION_OVERRIDE:id={actor_id}:'
            f'original={_optional_action_summary(before)}:'
            f'final={_optional_action_summary(after)}:'
            f'module={module}:reason={reason}'
        )
        if len(entries) >= _ACTION_LOG_LIMIT:
            break
    return tuple(entries)


def _opening_final_action_log(
    observation: Observation,
    decision: Decision,
) -> tuple[str, ...]:
    entries: list[str] = []
    for role in sorted(observation.our.units, key=lambda item: item.unit_id):
        if role.role_type != 'worker':
            continue
        action = decision.commands.get(role.unit_id)
        stone = sum(item.casefold() == 'stone' for item in role.backpack)
        entries.append(
            f'id={role.unit_id}:pos={role.position.x},{role.position.y}:'
            f'stone={stone}:final={_optional_action_summary(action)}:'
            f'last_result={observation.last_action_results.get(role.unit_id)}'
        )
    return tuple(entries)


def _action_lifecycle_log(
    observation: Observation,
    planned: Decision,
    validated: Decision,
    serialized_actions: object,
) -> tuple[str, ...]:
    serialized = (
        serialized_actions if isinstance(serialized_actions, dict) else {}
    )
    unit_types = {
        unit.unit_id: unit.role_type for unit in observation.our.units
    }
    actor_ids = set(planned.commands) | set(validated.commands)
    actor_ids.update(
        int(actor_id)
        for actor_id in serialized
        if str(actor_id).lstrip('-').isdigit()
    )
    entries: list[str] = []
    for actor_id in sorted(actor_ids)[:_ACTION_LOG_LIMIT]:
        planned_action = planned.commands.get(actor_id)
        validated_action = validated.commands.get(actor_id)
        response_action = serialized.get(str(actor_id))
        entries.append(
            f'id={actor_id}:type={unit_types.get(actor_id, _UNKNOWN_UNIT_TYPE)}:'
            f'planned={_optional_action_summary(planned_action)}:'
            f'validated={_optional_action_summary(validated_action)}:'
            f'response={_serialized_action_summary(response_action)}'
        )
    return tuple(entries)


def _optional_action_summary(action: object) -> str:
    if action is None:
        return 'none'
    return _action_summary(action)


def _action_summary(action: object) -> str:
    kind = getattr(getattr(action, 'kind', None), 'value', 'unknown')
    details: list[str] = []
    controller_id = getattr(action, 'controller_id', None)
    if controller_id is not None:
        details.append(f'controller={controller_id}')
    targets = getattr(action, 'target_positions', ())
    if targets:
        details.append(
            'targets=' + '|'.join(f'{item.x},{item.y}' for item in targets)
        )
    name = getattr(action, 'name', None)
    if name is not None:
        details.append(f'name={name}')
    quantity = getattr(action, 'quantity', None)
    if quantity is not None:
        details.append(f'quantity={quantity}')
    joined = _ACTION_DETAIL_SEPARATOR.join(details)
    return kind if not details else f'{kind}[{joined}]'


def _serialized_action_summary(action: object) -> str:
    if not isinstance(action, dict):
        return 'none'
    kind = str(action.get('action', 'unknown'))
    details: list[str] = []
    if 'controllerId' in action:
        controller_id = action.get('controllerId')
        details.append(f'controller={controller_id}')
    targets = action.get('targetPos')
    if isinstance(targets, list):
        coordinates: list[str] = []
        for item in targets:
            if not isinstance(item, dict):
                continue
            x_value = item.get('x')
            y_value = item.get('y')
            coordinates.append(f'{x_value},{y_value}')
        if coordinates:
            details.append('targets=' + '|'.join(coordinates))
    if 'name' in action:
        name = action.get('name')
        details.append(f'name={name}')
    if 'num' in action:
        quantity = action.get('num')
        details.append(f'quantity={quantity}')
    joined = _ACTION_DETAIL_SEPARATOR.join(details)
    return kind if not details else f'{kind}[{joined}]'


def _record_deadline_fallback(
    telemetry: TelemetryRecorder,
    original: Decision | None = None,
) -> None:
    overrides = tuple(telemetry.current('action_override_log', ()))
    if original is not None:
        overrides += _decision_override_log(
            original,
            Decision(),
            module='deadline',
            reason='response_deadline',
        )
    telemetry.set(
        fallback_used=True,
        fallback_reason='deadline_low',
        timeout_prevented=True,
        decision_source='safe',
        response_action_count=0,
        action_override_log=overrides,
    )
