from typing import Any

from .decision.decision import Decision
from .decision.serializer import decision_to_payload


def safe_decision() -> Decision:
    return Decision()


def safe_payload() -> dict[str, Any]:
    return decision_to_payload(safe_decision())
