from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
import re

from future_war_agent.protocol.models import Observation


MAX_PRICE_SNAPSHOTS = 12
MAX_MARKET_SIGNALS = 16
MAX_MARKET_NEWS = 16
MAX_RULE_RELIABILITY = 16
MAX_MARKET_TEXT = 32_768


class PriceDirection(StrEnum):
    DOWN = 'down'
    FLAT = 'flat'
    UP = 'up'
    UNKNOWN = 'unknown'


@dataclass(frozen=True, slots=True)
class DailyPriceSnapshot:
    day: int
    prices: tuple[tuple[str, int], ...]

    def price(self, mineral: str) -> int | None:
        return dict(self.prices).get(mineral)


@dataclass(frozen=True, slots=True)
class MarketSignal:
    mineral: str
    direction: PriceDirection
    start_day: int
    end_day: int
    rule_id: str
    confidence: float
    settled: bool = False

    def __post_init__(self) -> None:
        if not self.mineral.strip() or not self.rule_id.strip():
            raise ValueError('market signal fields must be nonblank')
        if self.start_day <= 0 or self.end_day < self.start_day:
            raise ValueError('market signal day range is invalid')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('market signal confidence must be in [0, 1]')

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.mineral.casefold(),
            self.direction,
            self.start_day,
            self.end_day,
            self.rule_id,
        )


@dataclass(frozen=True, slots=True)
class RuleReliability:
    rule_id: str
    correct: int = 0
    wrong: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct + 1) / (self.correct + self.wrong + 2)


@dataclass(frozen=True, slots=True)
class MarketState:
    snapshots: tuple[DailyPriceSnapshot, ...] = ()
    signals: tuple[MarketSignal, ...] = ()
    processed_news: tuple[str, ...] = ()
    reliability: tuple[RuleReliability, ...] = ()

    def __post_init__(self) -> None:
        if len(self.snapshots) > MAX_PRICE_SNAPSHOTS:
            raise ValueError('market snapshot capacity exceeded')
        if len(self.signals) > MAX_MARKET_SIGNALS:
            raise ValueError('market signal capacity exceeded')
        if len(self.processed_news) > MAX_MARKET_NEWS:
            raise ValueError('market news capacity exceeded')
        if len(self.reliability) > MAX_RULE_RELIABILITY:
            raise ValueError('market reliability capacity exceeded')


EMPTY_MARKET_STATE = MarketState()


@dataclass(frozen=True, slots=True)
class MarketView:
    next_day_directions: tuple[tuple[str, PriceDirection], ...] = ()
    hold_items: frozenset[str] = frozenset()
    sell_items: frozenset[str] = frozenset()

    def direction(self, mineral: str) -> PriceDirection:
        return dict(self.next_day_directions).get(
            mineral,
            PriceDirection.UNKNOWN,
        )


class MarketMemory:
    def observe(
        self,
        observation: Observation,
        previous_state: MarketState | None = None,
    ) -> MarketState:
        state = previous_state if previous_state is not None else EMPTY_MARKET_STATE
        snapshot = _snapshot(observation)
        state = _settle_predictions(state, snapshot)
        state = _record_snapshot(state, snapshot)
        return _observe_news(observation, state)

    def view(
        self,
        observation: Observation,
        state: MarketState,
    ) -> MarketView:
        target_day = observation.time.day_no + 1
        reliability = {item.rule_id: item for item in state.reliability}
        by_mineral: dict[str, set[PriceDirection]] = {}
        for signal in state.signals:
            stats = reliability.get(signal.rule_id)
            if (
                not signal.start_day <= target_day <= signal.end_day
                or signal.confidence < 0.75
                or (
                    stats is not None
                    and stats.wrong >= 2
                    and stats.accuracy < 0.5
                )
            ):
                continue
            by_mineral.setdefault(signal.mineral, set()).add(signal.direction)
        directions: list[tuple[str, PriceDirection]] = []
        for mineral, values in sorted(by_mineral.items()):
            direction = (
                next(iter(values))
                if len(values) == 1
                else PriceDirection.UNKNOWN
            )
            directions.append((mineral, direction))
        return MarketView(
            next_day_directions=tuple(directions),
            hold_items=frozenset(
                mineral
                for mineral, direction in directions
                if direction is PriceDirection.UP
            ),
            sell_items=frozenset(
                mineral
                for mineral, direction in directions
                if direction is PriceDirection.DOWN
            ),
        )


def parse_market_signals(
    text: str,
    *,
    current_day: int,
) -> tuple[MarketSignal, ...]:
    cleaned = text.replace('\x00', '').strip()
    if not cleaned or len(cleaned) > MAX_MARKET_TEXT:
        return ()
    signals: list[MarketSignal] = []
    sentences = re.split(r'[。！？!?;；\n]+', cleaned)
    aliases = {
        'stone': ('stone', '石头', '石料'),
        'iron': ('iron', '铁矿', '铁'),
        'copper': ('copper', '铜矿', '铜'),
    }
    up_words = (
        'shutdown',
        'collapse',
        'shortage',
        'disruption',
        '停工',
        '塌方',
        '短缺',
        '中断',
    )
    down_words = (
        'new mine',
        'reopen',
        'surplus',
        'increased output',
        '新矿',
        '复产',
        '增产',
        '过剩',
    )
    for sentence in sentences:
        folded = sentence.casefold()
        duration_match = re.search(
            r'(?:for\s*)?(\d+)\s*days?|持续\s*(\d+)\s*天|'
            r'(\d+)\s*天',
            sentence,
            flags=re.IGNORECASE,
        )
        if duration_match is None:
            continue
        duration = int(next(value for value in duration_match.groups() if value))
        if not 1 <= duration <= 10:
            continue
        has_up = any(word in folded for word in up_words)
        has_down = any(word in folded for word in down_words)
        if has_up == has_down:
            continue
        direction = PriceDirection.UP if has_up else PriceDirection.DOWN
        rule_id = (
            'supply-reduction-v1'
            if direction is PriceDirection.UP
            else 'supply-expansion-v1'
        )
        for mineral, names in aliases.items():
            if not any(name in folded for name in names):
                continue
            signals.append(
                MarketSignal(
                    mineral=mineral,
                    direction=direction,
                    start_day=current_day + 1,
                    end_day=current_day + duration,
                    rule_id=rule_id,
                    confidence=0.8,
                )
            )
    unique = {signal.key: signal for signal in signals}
    return tuple(
        sorted(
            unique.values(),
            key=lambda signal: signal.key,
        )
    )[:MAX_MARKET_SIGNALS]


def _snapshot(observation: Observation) -> DailyPriceSnapshot | None:
    prices = tuple(
        sorted(
            (item.name.casefold(), item.price)
            for item in observation.vendor_shop
            if item.price >= 0
        )
    )
    if not prices:
        return None
    return DailyPriceSnapshot(observation.time.day_no, prices)


def _record_snapshot(
    state: MarketState,
    snapshot: DailyPriceSnapshot | None,
) -> MarketState:
    if snapshot is None:
        return state
    snapshots = (
        snapshot,
        *(value for value in state.snapshots if value.day != snapshot.day),
    )[:MAX_PRICE_SNAPSHOTS]
    return replace(state, snapshots=snapshots)


def _settle_predictions(
    state: MarketState,
    current: DailyPriceSnapshot | None,
) -> MarketState:
    if current is None:
        return state
    prior = next(
        (
            value
            for value in state.snapshots
            if value.day < current.day
        ),
        None,
    )
    if prior is None:
        return state
    reliability = {item.rule_id: item for item in state.reliability}
    signals: list[MarketSignal] = []
    for signal in state.signals:
        if signal.settled or not signal.start_day <= current.day <= signal.end_day:
            signals.append(signal)
            continue
        before = prior.price(signal.mineral)
        after = current.price(signal.mineral)
        if before is None or after is None:
            signals.append(signal)
            continue
        actual = (
            PriceDirection.UP
            if after > before
            else PriceDirection.DOWN
            if after < before
            else PriceDirection.FLAT
        )
        stats = reliability.get(
            signal.rule_id,
            RuleReliability(signal.rule_id),
        )
        reliability[signal.rule_id] = replace(
            stats,
            correct=stats.correct + int(actual is signal.direction),
            wrong=stats.wrong + int(actual is not signal.direction),
        )
        signals.append(replace(signal, settled=True))
    return replace(
        state,
        signals=tuple(signals),
        reliability=tuple(
            sorted(reliability.values(), key=lambda value: value.rule_id)
        )[:MAX_RULE_RELIABILITY],
    )


def _observe_news(
    observation: Observation,
    state: MarketState,
) -> MarketState:
    text = observation.world_news.official_news.replace('\x00', '').strip()
    if not text or len(text) > MAX_MARKET_TEXT:
        return state
    normalized = ' '.join(text.casefold().split())
    fingerprint = sha256(normalized.encode('utf-8')).hexdigest()
    if fingerprint in state.processed_news:
        return state
    incoming = parse_market_signals(
        text,
        current_day=observation.time.day_no,
    )
    existing = {signal.key: signal for signal in state.signals}
    for signal in incoming:
        existing.setdefault(signal.key, signal)
    return replace(
        state,
        signals=tuple(
            sorted(
                existing.values(),
                key=lambda signal: (
                    -signal.start_day,
                    signal.mineral,
                    signal.rule_id,
                ),
            )
        )[:MAX_MARKET_SIGNALS],
        processed_news=(fingerprint, *state.processed_news)[:MAX_MARKET_NEWS],
    )
