import logging
from collections.abc import Callable
from inspect import Parameter, signature
from time import monotonic

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

DEFAULT_STRATEGY_ENGINE = StrategyEngine()


def default_planner(
    observation: Observation,
    *,
    request_started_at: float | None = None,
) -> Decision:
    if request_started_at is None:
        return DEFAULT_STRATEGY_ENGINE.plan(observation)
    return DEFAULT_STRATEGY_ENGINE.plan(
        observation,
        request_started_at=request_started_at,
    )


def handle_payload(
    payload: object,
    planner: Planner = default_planner,
    *,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
) -> dict[str, object]:
    request_started_at = monotonic()
    token = telemetry.begin()
    try:
        with telemetry.measure('parser_ms'):
            observation = parse_observation(payload)
        telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
        if _accepts_keyword(planner, 'request_started_at'):
            decision = planner(
                observation,
                request_started_at=request_started_at,
            )
        else:
            decision = planner(observation)
        with telemetry.measure('validation_ms'):
            validated = validate_decision(observation, decision)
        with telemetry.measure('serialization_ms'):
            return decision_to_payload(validated)
    except Exception:
        telemetry.set(fallback_used=True)
        LOGGER.exception("turn handling failed")
        return safe_payload()
    finally:
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
