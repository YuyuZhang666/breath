from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .actions import Action


@dataclass(frozen=True, slots=True)
class Decision:
    commands: Mapping[int, Action] = field(default_factory=dict)
    prompt: str = ""
    execute_command: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "commands", MappingProxyType(dict(self.commands)))
