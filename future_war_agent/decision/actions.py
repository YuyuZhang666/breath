from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from future_war_agent.protocol.models import Position


class ActionKind(StrEnum):
    MOVE = "move"
    ATTACK = "attack"
    SELL = "sell"
    BUY = "buy"
    BUILD = "build"
    REMOVE = "remove"
    ACCEPT_TASK = "acceptTask"
    SUBMIT_ANSWER = "submitAnswer"
    SUMMON_TREASURE = "summonTreasure"
    USE = "use"
    DROP = "drop"
    COLLECT = "collect"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    target_positions: tuple[Position, ...] = ()
    controller_id: int | None = None
    name: str | None = None
    quantity: int | None = None
    task_answer: str | None = None
    items: tuple[str, ...] = ()

    @classmethod
    def move(cls, target: Position) -> "Action":
        return cls(ActionKind.MOVE, target_positions=(target,))

    @classmethod
    def attack(
        cls, controller_id: int, targets: Iterable[Position]
    ) -> "Action":
        return cls(
            ActionKind.ATTACK,
            target_positions=tuple(targets),
            controller_id=controller_id,
        )

    @classmethod
    def sell(cls, name: str, quantity: int) -> "Action":
        return cls(ActionKind.SELL, name=name, quantity=quantity)

    @classmethod
    def buy(cls, name: str, quantity: int) -> "Action":
        return cls(ActionKind.BUY, name=name, quantity=quantity)

    @classmethod
    def build(cls, name: str, target: Position) -> "Action":
        return cls(ActionKind.BUILD, target_positions=(target,), name=name)

    @classmethod
    def remove(cls, target: Position) -> "Action":
        return cls(ActionKind.REMOVE, target_positions=(target,))

    @classmethod
    def accept_task(cls) -> "Action":
        return cls(ActionKind.ACCEPT_TASK)

    @classmethod
    def submit_answer(cls, answer: str) -> "Action":
        return cls(ActionKind.SUBMIT_ANSWER, task_answer=answer)

    @classmethod
    def summon_treasure(
        cls, target: Position, items: Iterable[str]
    ) -> "Action":
        return cls(
            ActionKind.SUMMON_TREASURE,
            target_positions=(target,),
            items=tuple(items),
        )

    @classmethod
    def use(cls, name: str, target: Position | None = None) -> "Action":
        targets = () if target is None else (target,)
        return cls(ActionKind.USE, target_positions=targets, name=name)

    @classmethod
    def drop(cls, name: str) -> "Action":
        return cls(ActionKind.DROP, name=name)

    @classmethod
    def collect(cls, target: Position) -> "Action":
        return cls(ActionKind.COLLECT, target_positions=(target,))
