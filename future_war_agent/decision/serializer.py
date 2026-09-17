from typing import Any

from .actions import Action
from .decision import Decision


def action_to_payload(action: Action) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action.kind.value}

    if action.controller_id is not None:
        payload["controllerId"] = str(action.controller_id)
    if action.target_positions:
        payload["targetPos"] = [
            {"x": position.x, "y": position.y}
            for position in action.target_positions
        ]
    if action.name is not None:
        payload["name"] = action.name
    if action.quantity is not None:
        payload["num"] = action.quantity
    if action.task_answer is not None:
        payload["taskAnswer"] = action.task_answer
    if action.items:
        payload["item"] = list(action.items)

    return payload


def decision_to_payload(decision: Decision) -> dict[str, Any]:
    return {
        "roleCommandMap": {
            str(actor_id): action_to_payload(decision.commands[actor_id])
            for actor_id in sorted(decision.commands)
        },
        "prompt": decision.prompt,
        "executeCmd": decision.execute_command,
    }
