import logging
from collections.abc import Callable

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

Planner = Callable[[Observation], Decision]

DEFAULT_STRATEGY_ENGINE = StrategyEngine()


def default_planner(observation: Observation) -> Decision:
    return DEFAULT_STRATEGY_ENGINE.plan(observation)


def handle_payload(
    payload: object,
    planner: Planner = default_planner,
    *,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
) -> dict[str, object]:
    token = telemetry.begin()
    try:
        with telemetry.measure('parser_ms'):
            observation = parse_observation(payload)
        telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
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
