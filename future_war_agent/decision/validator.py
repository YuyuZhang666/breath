from collections import Counter

from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import Phase
from future_war_agent.strategy.simulation.geometry import is_legal_cone

from .actions import Action, ActionKind
from .decision import Decision


_PERSONAL_ROLES = frozenset({"worker", "pioneer"})
_WEAPON_ROLES = frozenset({"gatling", "railgun", "rocket"})
_WORKER_ACTIONS = frozenset(
    {ActionKind.COLLECT, ActionKind.BUILD, ActionKind.REMOVE}
)
_PIONEER_ACTIONS = frozenset(
    {
        ActionKind.ACCEPT_TASK,
        ActionKind.SUBMIT_ANSWER,
        ActionKind.SUMMON_TREASURE,
    }
)
_TARGETED_REMOTE_ITEMS = frozenset({'dizzyweapon', 'bomb'})
_TARGETED_ADJACENT_ITEMS = frozenset({
    'wallfixer',
    'weaponupgradevoucher1',
    'weaponupgradevoucher2',
    'wallupgradevoucher1',
    'wallupgradevoucher2',
    'stationupgradevoucher1',
    'stationupgradevoucher2',
})


def _is_non_blank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _has_valid_targets(action: Action, minimum: int, maximum: int) -> bool:
    if not minimum <= len(action.target_positions) <= maximum:
        return False
    return all(
        isinstance(target, Position)
        and isinstance(target.x, int)
        and not isinstance(target.x, bool)
        and isinstance(target.y, int)
        and not isinstance(target.y, bool)
        for target in action.target_positions
    )


def _only_fields(
    action: Action,
    *,
    targets: bool = False,
    controller: bool = False,
    name: bool = False,
    quantity: bool = False,
    answer: bool = False,
    items: bool = False,
) -> bool:
    return (
        (targets or not action.target_positions)
        and (controller or action.controller_id is None)
        and (name or action.name is None)
        and (quantity or action.quantity is None)
        and (answer or action.task_answer is None)
        and (items or not action.items)
    )


def _is_structurally_valid(action: object) -> bool:
    if not isinstance(action, Action) or not isinstance(action.kind, ActionKind):
        return False

    if action.kind is ActionKind.MOVE:
        return _has_valid_targets(action, 1, 1) and _only_fields(
            action, targets=True
        )

    if action.kind is ActionKind.ATTACK:
        return (
            _has_valid_targets(action, 1, len(action.target_positions))
            and isinstance(action.controller_id, int)
            and not isinstance(action.controller_id, bool)
            and _only_fields(action, targets=True, controller=True)
        )

    if action.kind in {ActionKind.SELL, ActionKind.BUY}:
        return (
            _is_non_blank(action.name)
            and _is_positive_integer(action.quantity)
            and _only_fields(action, name=True, quantity=True)
        )

    if action.kind is ActionKind.BUILD:
        return (
            _has_valid_targets(action, 1, 1)
            and _is_non_blank(action.name)
            and _only_fields(action, targets=True, name=True)
        )

    if action.kind in {ActionKind.REMOVE, ActionKind.COLLECT}:
        return _has_valid_targets(action, 1, 1) and _only_fields(
            action, targets=True
        )

    if action.kind is ActionKind.ACCEPT_TASK:
        return _only_fields(action)

    if action.kind is ActionKind.SUBMIT_ANSWER:
        return _is_non_blank(action.task_answer) and _only_fields(
            action, answer=True
        )

    if action.kind is ActionKind.SUMMON_TREASURE:
        return (
            _has_valid_targets(action, 1, 1)
            and bool(action.items)
            and all(_is_non_blank(item) for item in action.items)
            and _only_fields(action, targets=True, items=True)
        )

    if action.kind is ActionKind.USE:
        return (
            _has_valid_targets(action, 0, 1)
            and _is_non_blank(action.name)
            and _only_fields(action, targets=True, name=True)
        )

    if action.kind is ActionKind.DROP:
        return _is_non_blank(action.name) and _only_fields(action, name=True)

    return False


def _targets_are_in_bounds(observation: Observation, action: Action) -> bool:
    return all(
        0 <= target.x < observation.width
        and 0 <= target.y < observation.height
        for target in action.target_positions
    )


def _is_valid_personal_action(
    observation: Observation,
    unit: UnitState,
    action: Action,
) -> bool:
    if unit.role_type not in _PERSONAL_ROLES:
        return False
    if action.kind in _WORKER_ACTIONS and unit.role_type != "worker":
        return False
    if action.kind in _PIONEER_ACTIONS and unit.role_type != "pioneer":
        return False
    if action.kind is ActionKind.BUILD and observation.time.phase is not Phase.DAY:
        return False
    backpack = Counter(item.casefold() for item in unit.backpack)
    if action.kind is ActionKind.SUMMON_TREASURE:
        requested = Counter(item.casefold() for item in action.items)
        target = action.target_positions[0]
        if (
            unit.position.chebyshev_distance(target) > 1
            or any(backpack[name] < quantity for name, quantity in requested.items())
        ):
            return False
    if action.kind is ActionKind.USE:
        if action.name is None or backpack[action.name.casefold()] <= 0:
            return False
        if not _valid_strategic_item_use(observation, unit, action):
            return False
    if action.kind is ActionKind.BUY:
        if not _valid_buy(observation, unit, action):
            return False
    if action.kind is ActionKind.SELL:
        if not _valid_sell(observation, unit, action):
            return False
    if action.kind is ActionKind.COLLECT:
        target = action.target_positions[0]
        if (
            unit.position.chebyshev_distance(target) > 1
            or not any(
                zone.position == target
                and zone.neutral_type in {'stone', 'iron', 'copper'}
                for zone in observation.zones
            )
        ):
            return False
    return True


def _valid_strategic_item_use(
    observation: Observation,
    unit: UnitState,
    action: Action,
) -> bool:
    name = (action.name or '').casefold()
    targeted = _TARGETED_REMOTE_ITEMS | _TARGETED_ADJACENT_ITEMS
    if name not in targeted:
        return True
    if len(action.target_positions) != 1:
        return False
    target = action.target_positions[0]
    if (
        name in _TARGETED_ADJACENT_ITEMS
        and unit.position.chebyshev_distance(target) > 1
    ):
        return False
    if name in _TARGETED_REMOTE_ITEMS:
        return True

    target_unit = next(
        (
            candidate
            for candidate in observation.our.units
            if candidate.health > 0 and candidate.position == target
        ),
        None,
    )
    if target_unit is None:
        return False
    if name == 'wallfixer':
        return target_unit.role_type == 'wall'

    expected_level = 1 if name.endswith('1') else 2
    if target_unit.level != expected_level:
        return False
    if name.startswith('station'):
        return target_unit.role_type == 'station'
    if name.startswith('wall'):
        return target_unit.role_type == 'wall'
    return target_unit.role_type in _WEAPON_ROLES


def _valid_buy(
    observation: Observation,
    unit: UnitState,
    action: Action,
) -> bool:
    if action.name is None or action.quantity is None:
        return False
    prices = _adjacent_buy_prices(observation, unit, action.name)
    if not prices:
        return False
    free_slots = max(0, unit.backpack_capacity - len(unit.backpack))
    return (
        action.quantity <= free_slots
        and min(prices) * action.quantity <= observation.our.gold
    )


def _valid_sell(
    observation: Observation,
    unit: UnitState,
    action: Action,
) -> bool:
    if action.name is None or action.quantity is None:
        return False
    backpack = Counter(item.casefold() for item in unit.backpack)
    listed = any(
        item.name == action.name and item.price > 0
        for item in observation.vendor_shop
    )
    adjacent = any(
        zone.neutral_type == 'vendor'
        and unit.position.chebyshev_distance(zone.position) <= 1
        for zone in observation.zones
    )
    return listed and adjacent and backpack[action.name.casefold()] >= action.quantity


def _adjacent_buy_prices(
    observation: Observation,
    unit: UnitState,
    name: str,
) -> tuple[int, ...]:
    adjacent_types = {
        zone.neutral_type
        for zone in observation.zones
        if unit.position.chebyshev_distance(zone.position) <= 1
    }
    prices: list[int] = []
    if 'vendor' in adjacent_types:
        prices.extend(
            item.price
            for item in observation.vendor_shop
            if item.name == name and item.price > 0
        )
    if 'weaponShop' in adjacent_types:
        prices.extend(
            item.price
            for item in observation.weapon_shop
            if item.name == name and item.price > 0
        )
    return tuple(prices)


def _buy_cost(
    observation: Observation,
    unit: UnitState,
    action: Action,
) -> int:
    if action.kind is not ActionKind.BUY or action.name is None:
        return 0
    price = min(
        _adjacent_buy_prices(observation, unit, action.name),
        default=0,
    )
    return price * (action.quantity or 0)


def _is_valid_attack(weapon: UnitState, action: Action) -> bool:
    targets = action.target_positions
    if (
        weapon.cooldown != 0
        or weapon.attack_range <= 0
        or len(set(targets)) != len(targets)
        or any(
            weapon.position.chebyshev_distance(target) > weapon.attack_range
            for target in targets
        )
    ):
        return False
    if weapon.role_type == 'railgun':
        return len(targets) == 1
    if weapon.level is None or weapon.level <= 0:
        return False
    if weapon.role_type == 'gatling':
        return len(targets) == weapon.level and is_legal_cone(
            weapon.position,
            targets,
        )
    if weapon.role_type == 'rocket':
        return len(targets) == weapon.level
    return False


def validate_decision(observation: Observation, decision: Decision) -> Decision:
    living_units = {
        unit.unit_id: unit for unit in observation.our.units if unit.health > 0
    }
    candidates = [
        (unit_id, action)
        for unit_id, action in decision.commands.items()
        if isinstance(unit_id, int)
        and not isinstance(unit_id, bool)
        and unit_id in living_units
        and _is_structurally_valid(action)
        and _targets_are_in_bounds(observation, action)
    ]

    retained: dict[int, Action] = {}
    attack_candidates: list[tuple[int, Action]] = []

    remaining_gold = observation.our.gold
    for unit_id, action in sorted(candidates, key=lambda value: value[0]):
        if action.kind is ActionKind.ATTACK:
            attack_candidates.append((unit_id, action))
            continue
        if _is_valid_personal_action(
            observation,
            living_units[unit_id],
            action,
        ):
            cost = _buy_cost(observation, living_units[unit_id], action)
            if cost > remaining_gold:
                continue
            retained[unit_id] = action
            remaining_gold -= cost

    used_controllers: set[int] = set(retained)
    if observation.time.phase is Phase.NIGHT:
        for weapon_id, action in sorted(attack_candidates):
            weapon = living_units[weapon_id]
            controller_id = action.controller_id
            controller = living_units.get(controller_id)
            if weapon.role_type not in _WEAPON_ROLES:
                continue
            if not _is_valid_attack(weapon, action):
                continue
            if controller is None or controller_id in used_controllers:
                continue
            if controller.role_type not in _PERSONAL_ROLES:
                continue
            if controller.position.chebyshev_distance(weapon.position) > 1:
                continue

            retained[weapon_id] = action
            used_controllers.add(controller_id)

    return Decision(
        commands=retained,
        prompt=decision.prompt,
        execute_command=decision.execute_command,
    )
