from collections.abc import Callable
from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, Position


@dataclass(frozen=True, slots=True)
class BeliefConfig:
    decay_per_round: int = 10
    max_tracks: int = 64

    def __post_init__(self) -> None:
        if not 1 <= self.decay_per_round <= 100:
            raise ValueError('decay_per_round must be between 1 and 100')
        if self.max_tracks < 1:
            raise ValueError('max_tracks must be positive')


DEFAULT_BELIEF_CONFIG = BeliefConfig()


@dataclass(frozen=True, slots=True)
class OpponentTrack:
    unit_id: int
    role_type: str
    position: Position
    health: int
    last_seen_round: int
    last_updated_tick: int
    confidence: int

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError('confidence must be between 0 and 100')


@dataclass(frozen=True, slots=True)
class OpponentBelief:
    tracks: tuple[OpponentTrack, ...] = ()


PositionNormalizer = Callable[[Position], Position]


def update_opponent_belief(
    previous: OpponentBelief,
    observation: Observation,
    *,
    normalize: PositionNormalizer | None = None,
    tick: int | None = None,
    config: BeliefConfig = DEFAULT_BELIEF_CONFIG,
) -> OpponentBelief:
    normalizer = normalize if normalize is not None else _identity
    current_round = observation.time.round_no
    current_tick = current_round if tick is None else tick
    visible_by_id = {unit.unit_id: unit for unit in observation.enemy.units}
    tracks: dict[int, OpponentTrack] = {}

    for track in previous.tracks:
        if track.unit_id in visible_by_id:
            continue
        elapsed = max(0, current_tick - track.last_updated_tick)
        confidence = max(0, track.confidence - elapsed * config.decay_per_round)
        if confidence > 0:
            tracks[track.unit_id] = OpponentTrack(
                unit_id=track.unit_id,
                role_type=track.role_type,
                position=track.position,
                health=track.health,
                last_seen_round=track.last_seen_round,
                last_updated_tick=current_tick,
                confidence=confidence,
            )

    for unit in observation.enemy.units:
        if unit.health <= 0:
            continue
        tracks[unit.unit_id] = OpponentTrack(
            unit_id=unit.unit_id,
            role_type=unit.role_type.strip(),
            position=normalizer(unit.position),
            health=max(0, unit.health),
            last_seen_round=current_round,
            last_updated_tick=current_tick,
            confidence=100,
        )

    ranked = sorted(
        tracks.values(),
        key=lambda track: (
            -track.confidence,
            -track.last_seen_round,
            track.unit_id,
        ),
    )
    return OpponentBelief(tuple(ranked[: config.max_tracks]))


def _identity(position: Position) -> Position:
    return position
