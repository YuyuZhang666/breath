from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import re
from typing import TypeVar

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import Phase

from .pathfinding import path_to_interaction
from .policy import StrategicIntent, StrategyProfile
from .world import WorldGrid


MAX_TREASURE_CANDIDATES = 16
MAX_TREASURE_EVIDENCE = 16
MAX_TREASURE_NEWS = 16
MAX_TREASURE_NEGATIVE_EVIDENCE = 64
ATTEMPT_CONFIDENCE = 0.85
LAST_WINDOW_CONFIDENCE = 0.65
_EvidenceT = TypeVar('_EvidenceT')


@dataclass(frozen=True, slots=True)
class TreasureReward:
    score: int
    gold: int
    source: str
    confidence: float

    def __post_init__(self) -> None:
        if self.score < 0 or self.gold < 0:
            raise ValueError('treasure rewards cannot be negative')
        if not self.source.strip():
            raise ValueError('treasure reward source must be nonblank')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('treasure reward confidence must be in [0, 1]')


DEFAULT_TREASURE_REWARD = TreasureReward(
    score=200,
    gold=100,
    source='battle_log',
    confidence=1.0,
)


@dataclass(frozen=True, slots=True)
class TreasureClue:
    position: Position
    items: tuple[str, ...]
    opening_days: tuple[int, ...]
    confidence: float
    evidence: str


@dataclass(frozen=True, slots=True)
class TreasureCandidate:
    position: Position
    items: tuple[str, ...]
    opening_days: tuple[int, ...]
    confidence: float
    evidence_revision: int = 0
    support_evidence: tuple[str, ...] = ()
    conflict_evidence: tuple[str, ...] = ()
    invalid_days: frozenset[int] = frozenset()
    item_conflicts: int = 0
    depleted: bool = False
    attempt_count: int = 0
    last_attempt_evidence_revision: int = -1

    def __post_init__(self) -> None:
        if not self.items or any(not item.strip() for item in self.items):
            raise ValueError('treasure candidate items must be nonblank')
        if len(set(self.items)) != len(self.items):
            raise ValueError('treasure candidate items must be unique')
        if not self.opening_days or any(day <= 0 for day in self.opening_days):
            raise ValueError('treasure opening days must be positive')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('treasure confidence must be in [0, 1]')
        if min(
            self.evidence_revision,
            self.item_conflicts,
            self.attempt_count,
        ) < 0:
            raise ValueError('treasure counters cannot be negative')
        if len(self.support_evidence) > MAX_TREASURE_EVIDENCE:
            raise ValueError('treasure support evidence capacity exceeded')
        if len(self.conflict_evidence) > MAX_TREASURE_EVIDENCE:
            raise ValueError('treasure conflict evidence capacity exceeded')

    @property
    def key(self) -> str:
        identity = (
            self.position.x,
            self.position.y,
            self.items,
            self.opening_days,
        )
        return sha256(repr(identity).encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class TreasureState:
    reward: TreasureReward = DEFAULT_TREASURE_REWARD
    candidates: tuple[TreasureCandidate, ...] = ()
    pending_candidate_key: str = ''
    pending_round: int | None = None
    pending_day: int | None = None
    evidence_revision: int = 0
    attempt_count: int = 0
    last_attempt_evidence_revision: int = -1
    last_failure_code: int = 0
    last_failed_position: Position | None = None
    last_failed_items: tuple[str, ...] = ()
    last_failed_days: tuple[int, ...] = ()
    completed: bool = False
    depleted: bool = False
    processed_news: tuple[str, ...] = ()
    processed_results: tuple[str, ...] = ()
    rumors: tuple[str, ...] = ()
    failed_position_times: tuple[tuple[int, int, int], ...] = ()
    failed_item_multisets: tuple[tuple[str, ...], ...] = ()
    negative_evidence_saturated: bool = False

    def __post_init__(self) -> None:
        if len(self.candidates) > MAX_TREASURE_CANDIDATES:
            raise ValueError('treasure candidate capacity exceeded')
        if len(self.processed_news) > MAX_TREASURE_NEWS:
            raise ValueError('treasure news capacity exceeded')
        if len(self.processed_results) > MAX_TREASURE_NEWS:
            raise ValueError('treasure result capacity exceeded')
        if len(self.rumors) > MAX_TREASURE_NEWS:
            raise ValueError('treasure rumor capacity exceeded')
        if (
            len(self.failed_position_times) > MAX_TREASURE_NEGATIVE_EVIDENCE
            or len(self.failed_item_multisets) > MAX_TREASURE_NEGATIVE_EVIDENCE
        ):
            raise ValueError('treasure negative evidence capacity exceeded')


EMPTY_TREASURE_STATE = TreasureState()


@dataclass(frozen=True, slots=True)
class TreasureAgentResult:
    decision: Decision
    state: TreasureState


def parse_treasure_clues(
    value: str,
    width: int,
    height: int,
) -> tuple[TreasureClue, ...]:
    text = value.replace('\x00', '').strip()
    if not text or len(text) > 32_768:
        return ()
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        clue = _parse_natural_clue(text, width, height)
        return (clue,) if clue is not None else ()
    if isinstance(decoded, dict) and 'treasure' in decoded:
        decoded = decoded['treasure']
    records = decoded if isinstance(decoded, list) else [decoded]
    clues: list[TreasureClue] = []
    for record in records:
        clue = _parse_clue_record(record, text, width, height)
        if clue is not None:
            clues.append(clue)
    return tuple(clues[:MAX_TREASURE_CANDIDATES])


class TreasureAgent:
    def __init__(
        self,
        *,
        reward: TreasureReward = DEFAULT_TREASURE_REWARD,
    ) -> None:
        self._reward = reward

    def reconcile(
        self,
        observation: Observation,
        previous_state: TreasureState | None = None,
    ) -> TreasureState:
        state = (
            previous_state
            if previous_state is not None
            else TreasureState(reward=self._reward)
        )
        state = _reconcile_result(observation, state)
        return _observe_folk_legend(observation, state)

    def apply(
        self,
        observation: Observation,
        base_decision: Decision,
        intent: StrategicIntent,
        *,
        previous_state: TreasureState | None = None,
        task_active: bool = False,
    ) -> TreasureAgentResult:
        state = self.reconcile(observation, previous_state)
        if not _treasure_work_allowed(observation, base_decision, intent, state):
            return TreasureAgentResult(base_decision, state)
        pioneer = _living_pioneer(observation)
        if pioneer is None or task_active:
            return TreasureAgentResult(base_decision, state)
        replace_pioneer_move = _replaceable_pioneer_move(
            observation,
            base_decision,
            pioneer,
        )
        if (
            _pioneer_is_reserved(base_decision, pioneer.unit_id)
            and not replace_pioneer_move
        ):
            return TreasureAgentResult(base_decision, state)
        candidate = _select_candidate(observation, intent, state)
        if candidate is None:
            return TreasureAgentResult(base_decision, state)

        missing = _missing_items(pioneer, candidate.items)
        if missing:
            decision = _plan_item_purchase(
                observation,
                base_decision,
                pioneer,
                candidate,
                missing,
                intent,
                reward=state.reward,
                replace_pioneer_move=replace_pioneer_move,
            )
            return TreasureAgentResult(decision, state)

        if pioneer.position.chebyshev_distance(candidate.position) > 1:
            world = WorldGrid.from_observation(observation)
            path = path_to_interaction(world, pioneer.position, candidate.position)
            if path is None or len(path.path) < 2:
                return TreasureAgentResult(base_decision, state)
            return TreasureAgentResult(
                _overlay_pioneer(
                    observation,
                    base_decision,
                    pioneer,
                    Action.move(path.path[1]),
                    replace_pioneer_move=replace_pioneer_move,
                ),
                state,
            )

        decision = _overlay_pioneer(
            observation,
            base_decision,
            pioneer,
            Action.summon_treasure(candidate.position, candidate.items),
            replace_pioneer_move=replace_pioneer_move,
        )
        emitted = (
            decision.commands.get(pioneer.unit_id) is not None
            and decision.commands[pioneer.unit_id].kind
            is ActionKind.SUMMON_TREASURE
        )
        if not emitted:
            return TreasureAgentResult(base_decision, state)
        updated = replace(
            candidate,
            attempt_count=candidate.attempt_count + 1,
            last_attempt_evidence_revision=state.evidence_revision,
        )
        state = replace(
            state,
            candidates=_replace_candidate(state.candidates, updated),
            pending_candidate_key=candidate.key,
            pending_round=observation.time.round_no,
            pending_day=observation.time.day_no,
            attempt_count=max(state.attempt_count, _attempt_count(state)) + 1,
            last_attempt_evidence_revision=state.evidence_revision,
        )
        return TreasureAgentResult(decision, state)


def _parse_clue_record(
    record: object,
    evidence: str,
    width: int,
    height: int,
) -> TreasureClue | None:
    if not isinstance(record, dict):
        return None
    x = record.get('x')
    y = record.get('y')
    position = record.get('position')
    if isinstance(position, dict):
        x, y = position.get('x'), position.get('y')
    elif isinstance(position, list) and len(position) == 2:
        x, y = position
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not isinstance(y, int)
        or isinstance(y, bool)
        or not 0 <= x < width
        or not 0 <= y < height
    ):
        return None
    raw_items = record.get('items')
    if (
        not isinstance(raw_items, list)
        or not raw_items
        or any(not isinstance(item, str) or not item.strip() for item in raw_items)
    ):
        return None
    items = tuple(dict.fromkeys(item.strip() for item in raw_items))
    raw_days = record.get('days', record.get('day'))
    raw_days = [raw_days] if isinstance(raw_days, int) else raw_days
    if (
        not isinstance(raw_days, list)
        or not raw_days
        or any(
            not isinstance(day, int) or isinstance(day, bool) or day <= 0
            for day in raw_days
        )
    ):
        return None
    confidence = record.get('confidence')
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return None
    return TreasureClue(
        Position(x, y),
        items,
        tuple(sorted(set(raw_days))),
        float(confidence),
        evidence[:512],
    )


def _parse_natural_clue(
    text: str,
    width: int,
    height: int,
) -> TreasureClue | None:
    position_match = re.search(
        r'(?:coordinate|position|坐标|位置|祭坛)'
        r'\s*(?:is|at|为|是|=|:|：)?\s*'
        r'[\(（\[]?\s*(\d+)\s*[,，]\s*(\d+)',
        text,
        flags=re.IGNORECASE,
    )
    if position_match is None:
        return None
    x, y = (int(position_match.group(1)), int(position_match.group(2)))
    if not 0 <= x < width or not 0 <= y < height:
        return None

    item_match = re.search(
        r'(?:items?|offerings?|祭品|物品)'
        r'\s*(?:are|is|为|是|=|:|：)\s*'
        r'([^;；。\n]+)',
        text,
        flags=re.IGNORECASE,
    )
    if item_match is None:
        return None
    raw_items = re.split(
        r'(?:\bday\s*\d+|第\s*\d+\s*天|'
        r'\bconfidence\b|置信度|\bcoordinate\b|\bposition\b|坐标|位置)',
        item_match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    items = tuple(
        dict.fromkeys(
            item.strip().strip('[](){}"\' ')
            for item in re.split(r'[,，、|+]', raw_items)
            if item.strip().strip('[](){}"\' ')
        )
    )
    if (
        not items
        or len(items) > 8
        or any(len(item) > 64 for item in items)
    ):
        return None

    days = tuple(
        sorted(
            {
                int(english or chinese)
                for english, chinese in re.findall(
                    r'\bday\s*(\d+)\b|第\s*(\d+)\s*天',
                    text,
                    flags=re.IGNORECASE,
                )
            }
        )
    )
    if not days or any(day <= 0 for day in days):
        return None
    confidence_match = re.search(
        r'(?:confidence|置信度)\s*(?:=|:|：)?\s*(0(?:\.\d+)?|1(?:\.0+)?)',
        text,
        flags=re.IGNORECASE,
    )
    confidence = (
        float(confidence_match.group(1))
        if confidence_match is not None
        else ATTEMPT_CONFIDENCE
    )
    return TreasureClue(
        position=Position(x, y),
        items=items,
        opening_days=days,
        confidence=confidence,
        evidence=text[:512],
    )


def _observe_folk_legend(
    observation: Observation,
    state: TreasureState,
) -> TreasureState:
    rumor = observation.world_news.folk_legends.replace('\x00', '').strip()
    if not rumor:
        return state
    clues = parse_treasure_clues(rumor, observation.width, observation.height)
    semantic_evidence: object = (
        tuple(
            (
                clue.position.x,
                clue.position.y,
                clue.items,
                clue.opening_days,
                clue.confidence,
            )
            for clue in clues
        )
        if clues
        else rumor
    )
    fingerprint = sha256(repr(semantic_evidence).encode('utf-8')).hexdigest()
    if fingerprint in state.processed_news:
        return state
    revision = state.evidence_revision
    candidates = state.candidates
    for clue in clues:
        revision += 1
        incoming = TreasureCandidate(
            position=clue.position,
            items=clue.items,
            opening_days=clue.opening_days,
            confidence=clue.confidence,
            evidence_revision=revision,
            support_evidence=(clue.evidence,),
        )
        existing = next(
            (value for value in candidates if value.key == incoming.key),
            None,
        )
        if existing is not None:
            incoming = replace(
                existing,
                confidence=max(existing.confidence, incoming.confidence),
                evidence_revision=revision,
                support_evidence=(
                    clue.evidence,
                    *(
                        value
                        for value in existing.support_evidence
                        if value != clue.evidence
                    ),
                )[:MAX_TREASURE_EVIDENCE],
            )
        candidates = _replace_candidate(candidates, incoming)
    return replace(
        state,
        candidates=candidates[:MAX_TREASURE_CANDIDATES],
        evidence_revision=revision,
        processed_news=(fingerprint, *state.processed_news)[:MAX_TREASURE_NEWS],
        rumors=(rumor[:512], *state.rumors)[:MAX_TREASURE_NEWS],
    )


def _reconcile_result(
    observation: Observation,
    state: TreasureState,
) -> TreasureState:
    result = observation.last_summon_treasure_result
    if (
        not state.pending_candidate_key
        or state.pending_round is None
        or observation.time.round_no <= state.pending_round
    ):
        return state
    if result not in {1, 2, 3, 4}:
        return replace(
            state,
            pending_candidate_key='',
            pending_round=None,
            pending_day=None,
        )
    result_key = sha256(
        f'{state.pending_candidate_key}:{state.pending_round}:{result}'.encode('utf-8')
    ).hexdigest()
    if result_key in state.processed_results:
        return state
    candidate = next(
        (
            value
            for value in state.candidates
            if value.key == state.pending_candidate_key
        ),
        None,
    )
    candidates = state.candidates
    completed = state.completed
    depleted = state.depleted
    last_failure_code = 0
    last_failed_position = state.last_failed_position
    last_failed_items = state.last_failed_items
    last_failed_days = state.last_failed_days
    failed_position_times = state.failed_position_times
    failed_item_multisets = state.failed_item_multisets
    negative_evidence_saturated = state.negative_evidence_saturated
    if result == 1:
        completed = True
    elif candidate is not None and result == 2:
        last_failure_code = 2
        last_failed_position = candidate.position
        last_failed_items = candidate.items
        last_failed_days = candidate.opening_days
        failed_position_times, saturated = _append_negative_evidence(
            failed_position_times,
            (
                candidate.position.x,
                candidate.position.y,
                state.pending_round,
            ),
        )
        negative_evidence_saturated = (
            negative_evidence_saturated or saturated
        )
        candidate = replace(
            candidate,
            confidence=max(0.0, candidate.confidence - 0.25),
            conflict_evidence=(
                f'result:2:round:{state.pending_round}',
                *candidate.conflict_evidence,
            )[:MAX_TREASURE_EVIDENCE],
        )
        candidates = _replace_candidate(candidates, candidate)
    elif candidate is not None and result == 3:
        last_failure_code = 3
        last_failed_position = candidate.position
        last_failed_items = candidate.items
        last_failed_days = candidate.opening_days
        failed_item_multisets, saturated = _append_negative_evidence(
            failed_item_multisets,
            _item_multiset(candidate.items),
        )
        negative_evidence_saturated = (
            negative_evidence_saturated or saturated
        )
        candidate = replace(
            candidate,
            confidence=max(0.0, candidate.confidence - 0.25),
            item_conflicts=candidate.item_conflicts + 1,
            conflict_evidence=(
                'result:3:items',
                *candidate.conflict_evidence,
            )[:MAX_TREASURE_EVIDENCE],
        )
        candidates = _replace_candidate(candidates, candidate)
    elif result == 4:
        last_failure_code = 4
        depleted = True
        if candidate is not None:
            candidate = replace(candidate, confidence=0.0, depleted=True)
            candidates = _replace_candidate(candidates, candidate)
    return replace(
        state,
        candidates=candidates,
        pending_candidate_key='',
        pending_round=None,
        pending_day=None,
        completed=completed,
        depleted=depleted,
        last_failure_code=last_failure_code,
        last_failed_position=last_failed_position,
        last_failed_items=last_failed_items,
        last_failed_days=last_failed_days,
        failed_position_times=failed_position_times,
        failed_item_multisets=failed_item_multisets,
        negative_evidence_saturated=negative_evidence_saturated,
        processed_results=(result_key, *state.processed_results)[
            :MAX_TREASURE_NEWS
        ],
    )


def _treasure_work_allowed(
    observation: Observation,
    decision: Decision,
    intent: StrategicIntent,
    state: TreasureState,
) -> bool:
    return (
        observation.time.phase is Phase.DAY
        and intent.profile in {StrategyProfile.ECONOMY, StrategyProfile.SCORE}
        and intent.allow_tasks
        and not state.completed
        and not state.depleted
        and state.pending_round is None
        and not observation.phase_task.strip()
        and not decision.prompt
        and not decision.execute_command
    )


def _select_candidate(
    observation: Observation,
    intent: StrategicIntent,
    state: TreasureState,
) -> TreasureCandidate | None:
    if (
        state.reward.score + state.reward.gold <= 0
        or state.negative_evidence_saturated
    ):
        return None
    attempt_count = max(state.attempt_count, _attempt_count(state))
    eligible: list[TreasureCandidate] = []
    for candidate in state.candidates:
        day = observation.time.day_no
        threshold = (
            LAST_WINDOW_CONFIDENCE
            if day == max(candidate.opening_days)
            else ATTEMPT_CONFIDENCE
        )
        if (
            candidate.depleted
            or day not in candidate.opening_days
            or day in candidate.invalid_days
            or candidate.confidence < threshold
            or (
                candidate.position.x,
                candidate.position.y,
                observation.time.round_no,
            )
            in state.failed_position_times
            or _item_multiset(candidate.items)
            in state.failed_item_multisets
        ):
            continue
        if attempt_count > 0:
            if not intent.feature_flags.enable_treasure_retry:
                continue
            if state.evidence_revision <= state.last_attempt_evidence_revision:
                continue
        eligible.append(candidate)
    return (
        min(
            eligible,
            key=lambda value: (
                -value.confidence,
                -state.reward.score,
                -state.reward.gold,
                value.position.x,
                value.position.y,
                value.items,
            ),
        )
        if eligible
        else None
    )


def _attempt_count(state: TreasureState) -> int:
    return max((value.attempt_count for value in state.candidates), default=0)


def _item_multiset(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(item.casefold() for item in items))


def _append_negative_evidence(
    evidence: tuple[_EvidenceT, ...],
    value: _EvidenceT,
) -> tuple[tuple[_EvidenceT, ...], bool]:
    if value in evidence:
        return evidence, False
    if len(evidence) >= MAX_TREASURE_NEGATIVE_EVIDENCE:
        return evidence, True
    return (value, *evidence), False


def _living_pioneer(observation: Observation) -> UnitState | None:
    return next(
        (
            unit
            for unit in sorted(observation.our.units, key=lambda value: value.unit_id)
            if unit.health > 0 and unit.role_type == 'pioneer'
        ),
        None,
    )


def _pioneer_is_reserved(decision: Decision, pioneer_id: int) -> bool:
    return pioneer_id in decision.commands or any(
        action.controller_id == pioneer_id for action in decision.commands.values()
    )


def _replaceable_pioneer_move(
    observation: Observation,
    decision: Decision,
    pioneer: UnitState,
) -> bool:
    action = decision.commands.get(pioneer.unit_id)
    if (
        observation.time.phase is not Phase.DAY
        or action is None
        or action.kind is not ActionKind.MOVE
        or any(
            retained.controller_id == pioneer.unit_id
            for retained in decision.commands.values()
        )
    ):
        return False
    world = WorldGrid.from_observation(observation)
    rounds_left = 71 - observation.time.round_in_phase
    is_twilight_recall = any(
        path is not None
        and rounds_left <= path.cost + world.rules.twilight_safety_margin
        for weapon in world.weapons
        for path in (path_to_interaction(world, pioneer.position, weapon.position),)
    )
    return not is_twilight_recall


def _missing_items(
    pioneer: UnitState,
    required: tuple[str, ...],
) -> tuple[str, ...]:
    backpack = Counter(item.casefold() for item in pioneer.backpack)
    missing: list[str] = []
    for item in required:
        key = item.casefold()
        if backpack[key] > 0:
            backpack[key] -= 1
        else:
            missing.append(item)
    return tuple(missing)


def _plan_item_purchase(
    observation: Observation,
    base_decision: Decision,
    pioneer: UnitState,
    candidate: TreasureCandidate,
    missing: tuple[str, ...],
    intent: StrategicIntent,
    *,
    reward: TreasureReward,
    replace_pioneer_move: bool,
) -> Decision:
    offers: dict[str, tuple[int, str]] = {}
    for shop_type, shop_items in (
        ('vendor', observation.vendor_shop),
        ('weaponShop', observation.weapon_shop),
    ):
        for item in shop_items:
            offer = (item.price, shop_type)
            if item.price > 0 and (
                item.name not in offers or offer < offers[item.name]
            ):
                offers[item.name] = offer
    if any(item not in offers for item in missing):
        return base_decision
    bundle_cost = sum(offers[item][0] for item in missing)
    expected_gold_return = reward.gold * reward.confidence * candidate.confidence
    investable = max(0, observation.our.gold - intent.gold_reserve)
    free_slots = max(0, pioneer.backpack_capacity - len(pioneer.backpack))
    if (
        bundle_cost > investable
        or bundle_cost > expected_gold_return
        or len(missing) > free_slots
    ):
        return base_decision
    world = WorldGrid.from_observation(observation)
    next_item = sorted(missing, key=lambda value: (offers[value][0], value))[0]
    shop_type = offers[next_item][1]
    shops = world.positions_for_zone(shop_type)
    paths = tuple(
        path
        for shop in shops
        for path in (path_to_interaction(world, pioneer.position, shop),)
        if path is not None
    )
    if not paths:
        return base_decision
    path = min(
        paths,
        key=lambda value: (
            value.cost,
            tuple((position.x, position.y) for position in value.path),
        ),
    )
    if path.cost == 0:
        return _overlay_pioneer(
            observation,
            base_decision,
            pioneer,
            Action.buy(next_item, 1),
            replace_pioneer_move=replace_pioneer_move,
        )
    if len(path.path) < 2:
        return base_decision
    return _overlay_pioneer(
        observation,
        base_decision,
        pioneer,
        Action.move(path.path[1]),
        replace_pioneer_move=replace_pioneer_move,
    )


def _overlay_pioneer(
    observation: Observation,
    base_decision: Decision,
    pioneer: UnitState,
    action: Action,
    *,
    replace_pioneer_move: bool = False,
) -> Decision:
    if _pioneer_is_reserved(base_decision, pioneer.unit_id):
        existing = base_decision.commands.get(pioneer.unit_id)
        if (
            not replace_pioneer_move
            or existing is None
            or existing.kind is not ActionKind.MOVE
        ):
            return base_decision
    commands = dict(base_decision.commands)
    if action.kind is ActionKind.MOVE:
        target = action.target_positions[0]
        occupied = {
            unit.position
            for unit in observation.our.units
            if unit.health > 0 and unit.unit_id != pioneer.unit_id
        }
        reserved = {
            retained.target_positions[0]
            for retained in commands.values()
            if retained.kind in {ActionKind.MOVE, ActionKind.BUILD}
            and retained.target_positions
        }
        if target in occupied or target in reserved:
            return base_decision
    commands[pioneer.unit_id] = action
    return Decision(
        commands=commands,
        prompt=base_decision.prompt,
        execute_command=base_decision.execute_command,
    )


def _replace_candidate(
    candidates: tuple[TreasureCandidate, ...],
    candidate: TreasureCandidate,
) -> tuple[TreasureCandidate, ...]:
    return (
        candidate,
        *(value for value in candidates if value.key != candidate.key),
    )[:MAX_TREASURE_CANDIDATES]
