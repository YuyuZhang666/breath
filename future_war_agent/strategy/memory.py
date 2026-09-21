from dataclasses import dataclass
from fractions import Fraction

from future_war_agent.protocol.models import Observation, Position

from .belief import OpponentBelief, update_opponent_belief
from .capability_matrix import CapabilityMatrix, UNKNOWN_CAPABILITY_MATRIX
from .forecast import NightForecast
from .opponent_memory import OpponentMemory, update_opponent_memory
from .session import observation_fingerprint
from .simulation.certificate import (
    RobotWaveSafetyCertificate,
    WaveClassification,
)


MAX_NORMALIZED_ZONES = 256
MAX_WAVE_SUMMARIES = 16


@dataclass(frozen=True, slots=True)
class NormalizedZone:
    neutral_type: str
    position: Position


@dataclass(frozen=True, slots=True)
class WaveSummary:
    epoch: int
    round_no: int
    classification: WaveClassification
    secured: bool
    station_survival_probability: Fraction
    worst_station_health: int


@dataclass(frozen=True, slots=True)
class MatchMemory:
    team_id: str
    epoch: int = 0
    observation_tick: int = 0
    last_round: int = 0
    last_fingerprint: str = ''
    belief: OpponentBelief = OpponentBelief()
    opponent_memory: OpponentMemory = OpponentMemory()
    capability_matrix: CapabilityMatrix = UNKNOWN_CAPABILITY_MATRIX
    zones: tuple[NormalizedZone, ...] = ()
    wave_summaries: tuple[WaveSummary, ...] = ()
    recent_threat_positions: tuple[Position, ...] = ()
    night_forecast: NightForecast | None = None


class MatchMemoryStore:
    def __init__(
        self,
        *,
        capability_matrix: CapabilityMatrix | None = None,
    ) -> None:
        self._by_team: dict[str, MatchMemory] = {}
        self._capability_matrix = (
            capability_matrix
            if capability_matrix is not None
            else UNKNOWN_CAPABILITY_MATRIX
        )

    @property
    def capability_matrix(self) -> CapabilityMatrix:
        return self._capability_matrix

    def get(self, team_id: str) -> MatchMemory | None:
        if not team_id.strip():
            return None
        return self._by_team.get(team_id)

    def observe(
        self,
        observation: Observation,
        *,
        certificate: RobotWaveSafetyCertificate | None = None,
        forecast: NightForecast | None = None,
    ) -> MatchMemory:
        team_id = observation.our.team_id
        previous = self.get(team_id)
        fingerprint = observation_fingerprint(observation)
        if (
            previous is not None
            and previous.last_fingerprint == fingerprint
            and certificate is None
            and forecast is None
        ):
            return previous
        memory = update_match_memory(
            previous,
            observation,
            certificate=certificate,
            forecast=forecast,
            capability_matrix=self._capability_matrix,
        )
        if team_id.strip():
            self._by_team[team_id] = memory
        return memory


def canonical_position(
    observation: Observation,
    position: Position,
) -> Position:
    station = next(
        (
            unit
            for unit in sorted(observation.our.units, key=lambda item: item.unit_id)
            if unit.health > 0 and unit.role_type == 'station'
        ),
        None,
    )
    if station is None:
        return position
    rotate = (
        station.position.x * 2 < observation.width - 1
        or station.position.y * 2 < observation.height - 1
    )
    if not rotate:
        return position
    return Position(
        observation.width - 1 - position.x,
        observation.height - 1 - position.y,
    )


def observation_side_key(observation: Observation) -> str:
    station = next(
        (
            unit
            for unit in sorted(observation.our.units, key=lambda item: item.unit_id)
            if unit.health > 0 and unit.role_type == 'station'
        ),
        None,
    )
    if station is None:
        return 'unknown'
    rotated = (
        station.position.x * 2 < observation.width - 1
        or station.position.y * 2 < observation.height - 1
    )
    return 'rotated' if rotated else 'native'


def update_match_memory(
    previous: MatchMemory | None,
    observation: Observation,
    *,
    certificate: RobotWaveSafetyCertificate | None = None,
    forecast: NightForecast | None = None,
    capability_matrix: CapabilityMatrix | None = None,
) -> MatchMemory:
    team_id = observation.our.team_id
    matrix = (
        capability_matrix
        if capability_matrix is not None
        else (
            previous.capability_matrix
            if previous is not None
            else UNKNOWN_CAPABILITY_MATRIX
        )
    )
    prior = (
        previous
        if previous is not None
        else MatchMemory(team_id=team_id, capability_matrix=matrix)
    )
    fingerprint = observation_fingerprint(observation)
    same_observation = prior.last_fingerprint == fingerprint
    rollback = (
        previous is not None
        and not same_observation
        and observation.time.round_no < prior.last_round
    )
    epoch = prior.epoch + (1 if rollback else 0)
    if previous is None:
        observation_tick = observation.time.round_no
    elif same_observation:
        observation_tick = prior.observation_tick
    elif rollback:
        observation_tick = prior.observation_tick + 1
    else:
        observation_tick = prior.observation_tick + max(
            0,
            observation.time.round_no - prior.last_round,
        )
    belief = prior.belief
    opponent_memory = prior.opponent_memory
    zones = prior.zones
    recent_threat_positions = (
        () if rollback else prior.recent_threat_positions
    )
    if not same_observation:
        belief = update_opponent_belief(
            prior.belief,
            observation,
            normalize=lambda position: canonical_position(observation, position),
            tick=observation_tick,
        )
        opponent_memory = update_opponent_memory(
            prior.opponent_memory,
            observation,
            epoch=epoch,
            observation_tick=observation_tick,
            side_key=observation_side_key(observation),
            normalize=lambda position: canonical_position(
                observation,
                position,
            ),
        )
        normalized_zones = {
            NormalizedZone(
                neutral_type=zone.neutral_type.strip(),
                position=canonical_position(observation, zone.position),
            )
            for zone in observation.zones
        }
        zones = tuple(
            sorted(
                normalized_zones,
                key=lambda zone: (
                    zone.neutral_type,
                    zone.position.x,
                    zone.position.y,
                ),
            )[:MAX_NORMALIZED_ZONES]
        )
        targeted_positions = {
            robot.position
            for robot in observation.robots
            if robot.health > 0
            and robot.target_team == observation.our.team_type
        }
        if targeted_positions:
            recent_threat_positions = tuple(
                sorted(
                    targeted_positions,
                    key=lambda position: (position.x, position.y),
                )
            )[:64]

    summaries = prior.wave_summaries
    if certificate is not None:
        summary = WaveSummary(
            epoch=epoch,
            round_no=observation.time.round_no,
            classification=certificate.classification,
            secured=certificate.secured,
            station_survival_probability=(
                certificate.station_survival_probability
            ),
            worst_station_health=certificate.worst_station_health,
        )
        summaries = (
            *(
                item
                for item in summaries
                if (item.epoch, item.round_no)
                != (summary.epoch, summary.round_no)
            ),
            summary,
        )[-MAX_WAVE_SUMMARIES:]

    return MatchMemory(
        team_id=team_id,
        epoch=epoch,
        observation_tick=observation_tick,
        last_round=observation.time.round_no,
        last_fingerprint=fingerprint,
        belief=belief,
        opponent_memory=opponent_memory,
        capability_matrix=matrix,
        zones=zones,
        wave_summaries=summaries,
        recent_threat_positions=recent_threat_positions,
        night_forecast=(
            forecast if forecast is not None else prior.night_forecast
        ),
    )
