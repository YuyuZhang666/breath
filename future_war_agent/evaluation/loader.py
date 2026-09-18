import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import ReplayFormatError
from .models import MatchOutcome, ReplayCase, ReplayCorpus


MAX_CASES = 256
MAX_TURNS_PER_CASE = 1_000


class _NonStandardJsonConstant(ValueError):
    pass


def load_replay_file(path: str | Path) -> ReplayCorpus:
    replay_path = Path(path)
    try:
        raw = json.loads(
            replay_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ReplayFormatError(
            f"{replay_path.name}: expected valid JSON: {error.msg}"
        ) from error
    except _NonStandardJsonConstant as error:
        raise ReplayFormatError(f"{replay_path.name}: {error}") from error
    except (OSError, UnicodeError) as error:
        raise ReplayFormatError(f"{replay_path}: could not read replay: {error}") from error
    return load_replay_data(raw)


def _reject_json_constant(value: str) -> object:
    raise _NonStandardJsonConstant(
        f"non-standard JSON numeric constant is not allowed: {value}"
    )


def load_replay_data(raw: object) -> ReplayCorpus:
    root = _mapping(raw, "replay")
    cases_raw = _required_sequence(root, "cases", "replay")
    if not cases_raw:
        raise ReplayFormatError("replay.cases must contain at least one case")
    if len(cases_raw) > MAX_CASES:
        raise ReplayFormatError(
            f"replay.cases must contain at most {MAX_CASES} cases"
        )

    seen_names: set[str] = set()
    cases: list[ReplayCase] = []
    for index, raw_case in enumerate(cases_raw):
        path = f"replay.cases[{index}]"
        value = _mapping(raw_case, path)
        name = _required_nonblank_string(value, "name", path).strip()
        if name in seen_names:
            raise ReplayFormatError(f"duplicate replay case name: {name}")
        seen_names.add(name)

        outcome_raw = _required_string(value, "outcome", path)
        try:
            outcome = MatchOutcome(outcome_raw)
        except ValueError as error:
            choices = ", ".join(item.value for item in MatchOutcome)
            raise ReplayFormatError(
                f"{path}.outcome must be one of: {choices}"
            ) from error

        turns_raw = _required_sequence(value, "turns", path)
        if not turns_raw:
            raise ReplayFormatError(f"{path}.turns must contain at least one turn")
        if len(turns_raw) > MAX_TURNS_PER_CASE:
            raise ReplayFormatError(
                f"{path}.turns must contain at most {MAX_TURNS_PER_CASE} turns"
            )
        turns = tuple(
            _freeze_mapping(turn, f"{path}.turns[{turn_index}]")
            for turn_index, turn in enumerate(turns_raw)
        )
        cases.append(ReplayCase(name=name, outcome=outcome, turns=turns))

    return ReplayCorpus(cases=tuple(cases))


def _freeze_mapping(raw: object, path: str) -> Mapping[str, object]:
    return _freeze(_mapping(raw, path))


def _freeze(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return MappingProxyType({str(key): _freeze(value) for key, value in raw.items()})
    if isinstance(raw, list):
        return tuple(_freeze(value) for value in raw)
    return raw


def _mapping(raw: object, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ReplayFormatError(f"{path} must be an object")
    return raw


def _required_sequence(
    value: Mapping[str, Any], key: str, path: str
) -> Sequence[Any]:
    if key not in value:
        raise ReplayFormatError(f"{path}.{key} is required")
    raw = value[key]
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ReplayFormatError(f"{path}.{key} must be an array")
    return raw


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    if key not in value:
        raise ReplayFormatError(f"{path}.{key} is required")
    raw = value[key]
    if not isinstance(raw, str):
        raise ReplayFormatError(f"{path}.{key} must be a string")
    return raw


def _required_nonblank_string(
    value: Mapping[str, Any], key: str, path: str
) -> str:
    raw = _required_string(value, key, path)
    if not raw.strip():
        raise ReplayFormatError(f"{path}.{key} must be nonblank")
    return raw
