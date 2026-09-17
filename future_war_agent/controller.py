import logging
from collections.abc import Callable

from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.fallback import safe_decision, safe_payload
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.parser import parse_observation


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Planner = Callable[[Observation], Decision]


def default_planner(_: Observation) -> Decision:
    return safe_decision()


def handle_payload(
    payload: object,
    planner: Planner = default_planner,
) -> dict[str, object]:
    try:
        observation = parse_observation(payload)
        decision = planner(observation)
        validated = validate_decision(observation, decision)
        return decision_to_payload(validated)
    except Exception:
        LOGGER.exception("turn handling failed")
        return safe_payload()
