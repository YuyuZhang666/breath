from collections.abc import Mapping, Sequence
from typing import Any

from .models import (
    EnemyTeamState,
    GameError,
    Observation,
    OurTeamState,
    Position,
    RobotState,
    ShopItem,
    TaskPointState,
    UnitState,
    WorldNews,
    Zone,
)
from .time import TurnTime


class ProtocolError(ValueError):
    pass


def parse_observation(raw: object) -> Observation:
    root = _mapping(raw, "request")
    round_no = _required_int(root, "roundNo", "request")
    if round_no < 1:
        raise ProtocolError("request.roundNo must be positive")

    map_info = _required_mapping(root, "mapInfo", "request")
    width = _required_int(map_info, "width", "request.mapInfo")
    height = _required_int(map_info, "height", "request.mapInfo")
    if width < 1 or height < 1:
        raise ProtocolError("map dimensions must be positive")

    team_our = _required_mapping(root, "teamOur", "request")
    team_enemy = _optional_mapping(root, "teamEnemy", "request")
    robot_group = _optional_mapping(root, "robot", "request")
    world_news = _optional_mapping(root, "worldNews", "request")

    return Observation(
        time=TurnTime.from_round(round_no),
        width=width,
        height=height,
        zones=tuple(
            _parse_zone(value, f"request.mapInfo.zones[{index}]")
            for index, value in enumerate(
                _optional_sequence(map_info, "zones", "request.mapInfo")
            )
        ),
        our=_parse_our_team(team_our, "request.teamOur"),
        enemy=EnemyTeamState(
            units=tuple(
                _parse_unit(value, f"request.teamEnemy.roles[{index}]")
                for index, value in enumerate(
                    _optional_sequence(team_enemy, "roles", "request.teamEnemy")
                )
            )
        ),
        robots=tuple(
            _parse_robot(value, f"request.robot.roles[{index}]")
            for index, value in enumerate(
                _optional_sequence(robot_group, "roles", "request.robot")
            )
        ),
        phase_task=_optional_string(root, "phaseTask", "request"),
        last_action_results=_parse_action_results(
            root.get("lastRoundRoleActionResults", {})
        ),
        last_summon_treasure_result=_optional_int(
            root, "lastSummonTreasureResult", "request"
        ),
        llm_response=_optional_string(root, "llmResp", "request"),
        world_news=WorldNews(
            official_news=_optional_string(
                world_news, "officialNews", "request.worldNews"
            ),
            folk_legends=_optional_string(
                world_news, "folkLegends", "request.worldNews"
            ),
        ),
        last_command_result=_optional_string(root, "lastCmdResult", "request"),
        vendor_shop=_parse_shop(root, "vendorShopList"),
        weapon_shop=_parse_shop(root, "weaponShopList"),
        errors=tuple(
            _parse_error(value, f"request.errors[{index}]")
            for index, value in enumerate(
                _optional_sequence(root, "errors", "request")
            )
        ),
    )


def _parse_position(raw: object, path: str) -> Position:
    value = _mapping(raw, path)
    return Position(
        x=_required_int(value, "x", path),
        y=_required_int(value, "y", path),
    )


def _parse_zone(raw: object, path: str) -> Zone:
    value = _mapping(raw, path)
    return Zone(
        position=_parse_position(_required(value, "pos", path), f"{path}.pos"),
        neutral_type=_required_string(value, "neutralType", path),
    )


def _parse_unit(raw: object, path: str) -> UnitState:
    value = _mapping(raw, path)
    backpack = _optional_sequence(value, "backpack", path)
    return UnitState(
        unit_id=_required_int(value, "id", path),
        position=_parse_position(_required(value, "pos", path), f"{path}.pos"),
        role_type=_required_string(value, "roleType", path),
        health=_required_int(value, "health", path),
        attack_power=_optional_int(value, "attackPower", path),
        attack_range=_optional_int(value, "attackRange", path),
        backpack_capacity=_optional_int(value, "backPackCapability", path),
        backpack=tuple(
            _string(item, f"{path}.backpack[{index}]")
            for index, item in enumerate(backpack)
        ),
        level=_nullable_int(value, "level", path),
        cooldown=_optional_int(value, "cooldown", path),
        provided_fields=frozenset(
            name
            for name in ("attackPower", "attackRange", "level", "cooldown")
            if name in value and value[name] is not None
        ),
    )


def _parse_robot(raw: object, path: str) -> RobotState:
    value = _mapping(raw, path)
    return RobotState(
        robot_id=_required_int(value, "id", path),
        position=_parse_position(_required(value, "pos", path), f"{path}.pos"),
        role_type=_required_string(value, "roleType", path),
        health=_required_int(value, "health", path),
        abnormal_state=_optional_string(value, "abnormalState", path),
        target_team=_nullable_string(value, "targetTeam", path),
    )


def _parse_task(raw: object, path: str) -> TaskPointState:
    value = _mapping(raw, path)
    is_valid = value.get("isValid", False)
    if not isinstance(is_valid, bool):
        raise ProtocolError(f"{path}.isValid must be a boolean")
    return TaskPointState(
        task_type=_required_string(value, "taskType", path),
        position=_parse_position(
            _required(value, "taskPosition", path), f"{path}.taskPosition"
        ),
        cooldown_rounds=_optional_int(value, "coldDownRounds", path),
        score_reward=_optional_int(value, "scoreReward", path),
        gold_reward=_optional_int(value, "goldReward", path),
        is_valid=is_valid,
        timeout_rounds=_nullable_int(value, "timeoutRounds", path),
    )


def _parse_our_team(value: Mapping[str, Any], path: str) -> OurTeamState:
    return OurTeamState(
        team_type=_optional_string(value, "type", path),
        team_id=_optional_string(value, "teamId", path),
        team_name=_optional_string(value, "teamName", path),
        gold=_optional_int(value, "goldNum", path),
        total_score=_optional_int(value, "totalScore", path),
        tasks=tuple(
            _parse_task(item, f"{path}.playerTasks[{index}]")
            for index, item in enumerate(
                _optional_sequence(value, "playerTasks", path)
            )
        ),
        units=tuple(
            _parse_unit(item, f"{path}.roles[{index}]")
            for index, item in enumerate(_optional_sequence(value, "roles", path))
        ),
    )


def _parse_shop(root: Mapping[str, Any], key: str) -> tuple[ShopItem, ...]:
    return tuple(
        _parse_shop_item(value, f"request.{key}[{index}]")
        for index, value in enumerate(_optional_sequence(root, key, "request"))
    )


def _parse_shop_item(raw: object, path: str) -> ShopItem:
    value = _mapping(raw, path)
    return ShopItem(
        name=_required_string(value, "name", path),
        price=_required_int(value, "price", path),
    )


def _parse_error(raw: object, path: str) -> GameError:
    value = _mapping(raw, path)
    return GameError(
        error_code=_required_int(value, "errorCode", path),
        description=_required_string(value, "description", path),
    )


def _parse_action_results(raw: object) -> dict[int, bool]:
    value = _mapping(raw, "request.lastRoundRoleActionResults")
    result: dict[int, bool] = {}
    for raw_key, raw_result in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ProtocolError(
                "request.lastRoundRoleActionResults keys must be integers"
            ) from error
        if not isinstance(raw_result, bool):
            raise ProtocolError(
                f"request.lastRoundRoleActionResults[{raw_key!r}] must be a boolean"
            )
        result[key] = raw_result
    return result


def _required(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        raise ProtocolError(f"{path}.{key} is required")
    return value[key]


def _mapping(raw: object, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProtocolError(f"{path} must be an object")
    return raw


def _required_mapping(
    value: Mapping[str, Any], key: str, path: str
) -> Mapping[str, Any]:
    return _mapping(_required(value, key, path), f"{path}.{key}")


def _optional_mapping(
    value: Mapping[str, Any], key: str, path: str
) -> Mapping[str, Any]:
    if key not in value or value[key] is None:
        return {}
    return _mapping(value[key], f"{path}.{key}")


def _optional_sequence(
    value: Mapping[str, Any], key: str, path: str
) -> Sequence[Any]:
    if key not in value or value[key] is None:
        return ()
    raw = value[key]
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ProtocolError(f"{path}.{key} must be an array")
    return raw


def _integer(raw: object, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ProtocolError(f"{path} must be an integer")
    return raw


def _required_int(value: Mapping[str, Any], key: str, path: str) -> int:
    return _integer(_required(value, key, path), f"{path}.{key}")


def _optional_int(value: Mapping[str, Any], key: str, path: str) -> int:
    if key not in value or value[key] is None:
        return 0
    return _integer(value[key], f"{path}.{key}")


def _nullable_int(
    value: Mapping[str, Any], key: str, path: str
) -> int | None:
    if key not in value or value[key] is None:
        return None
    return _integer(value[key], f"{path}.{key}")


def _string(raw: object, path: str) -> str:
    if not isinstance(raw, str):
        raise ProtocolError(f"{path} must be a string")
    return raw


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    return _string(_required(value, key, path), f"{path}.{key}")


def _optional_string(value: Mapping[str, Any], key: str, path: str) -> str:
    if key not in value or value[key] is None:
        return ""
    return _string(value[key], f"{path}.{key}")


def _nullable_string(
    value: Mapping[str, Any], key: str, path: str
) -> str | None:
    if key not in value or value[key] is None:
        return None
    return _string(value[key], f"{path}.{key}")
