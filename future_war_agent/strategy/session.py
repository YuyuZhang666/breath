from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, UnitState

from .director import DirectorState
from .features import StrategyFeatures
from .forecast import NightForecast
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .simulation.candidates import SimJointAction
from .simulation.certificate import RobotWaveSafetyCertificate
from .simulation.search import ScenarioWeights
from .task_agent import EMPTY_TASK_STATE, TaskAgentState


class SessionContinuity(StrEnum):
    DUPLICATE = "duplicate"
    REVISION = "revision"
    CONSECUTIVE = "consecutive"
    DISCONTINUITY = "discontinuity"


@dataclass(frozen=True, slots=True)
class StrategySession:
    team_id: str
    last_round: int
    fingerprint: str
    signature: str
    observation: Observation
    decision: Decision
    simulation_action: SimJointAction | None
    certificate: RobotWaveSafetyCertificate | None
    scenario_weights: ScenarioWeights
    features: StrategyFeatures | None = None
    director_state: DirectorState | None = None
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT
    task_state: TaskAgentState = EMPTY_TASK_STATE
    night_forecast: NightForecast | None = None


class SessionStore:
    def __init__(self) -> None:
        self._by_team: dict[str, StrategySession] = {}

    def get(self, team_id: str) -> StrategySession | None:
        if not team_id.strip():
            return None
        return self._by_team.get(team_id)

    def put(self, session: StrategySession) -> None:
        if session.team_id.strip():
            self._by_team[session.team_id] = session


def classify_continuity(
    previous: StrategySession,
    current: Observation,
) -> SessionContinuity:
    current_round = current.time.round_no
    current_signature = static_signature(current)
    if current_signature != previous.signature:
        return SessionContinuity.DISCONTINUITY
    if current_round == previous.last_round:
        if observation_fingerprint(current) == previous.fingerprint:
            return SessionContinuity.DUPLICATE
        return SessionContinuity.REVISION
    if current_round == previous.last_round + 1:
        return SessionContinuity.CONSECUTIVE
    return SessionContinuity.DISCONTINUITY


def observation_fingerprint(observation: Observation) -> str:
    key = (
        (
            observation.time.round_no,
            observation.time.day_no,
            observation.time.phase.value,
            observation.time.round_in_phase,
            observation.time.offset_in_day,
        ),
        observation.width,
        observation.height,
        tuple(
            sorted(
                (
                    zone.position.x,
                    zone.position.y,
                    zone.neutral_type,
                )
                for zone in observation.zones
            )
        ),
        (
            observation.our.team_type,
            observation.our.team_id,
            observation.our.team_name,
            observation.our.gold,
            observation.our.total_score,
            tuple(
                sorted(
                    (
                        task.task_type,
                        task.position.x,
                        task.position.y,
                        task.cooldown_rounds,
                        task.score_reward,
                        task.gold_reward,
                        task.is_valid,
                        task.timeout_rounds,
                    )
                    for task in observation.our.tasks
                )
            ),
            _unit_keys(observation.our.units),
        ),
        _unit_keys(observation.enemy.units),
        tuple(
            sorted(
                (
                    robot.robot_id,
                    robot.position.x,
                    robot.position.y,
                    robot.role_type,
                    robot.health,
                    robot.abnormal_state,
                    robot.target_team,
                )
                for robot in observation.robots
            )
        ),
        observation.phase_task,
        tuple(sorted(observation.last_action_results.items())),
        observation.last_summon_treasure_result,
        observation.llm_response,
        (
            observation.world_news.official_news,
            observation.world_news.folk_legends,
        ),
        observation.last_command_result,
        tuple(sorted((item.name, item.price) for item in observation.vendor_shop)),
        tuple(sorted((item.name, item.price) for item in observation.weapon_shop)),
        tuple(
            sorted((error.error_code, error.description) for error in observation.errors)
        ),
    )
    return _hash_key(key)


def static_signature(observation: Observation) -> str:
    stations = tuple(
        sorted(
            (
                unit.unit_id,
                unit.position.x,
                unit.position.y,
            )
            for unit in observation.our.units
            if unit.role_type == "station"
        )
    )
    key = (
        observation.width,
        observation.height,
        tuple(
            sorted(
                (
                    zone.position.x,
                    zone.position.y,
                    zone.neutral_type,
                )
                for zone in observation.zones
            )
        ),
        stations,
    )
    return _hash_key(key)


def _unit_keys(units: tuple[UnitState, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                unit.unit_id,
                unit.position.x,
                unit.position.y,
                unit.role_type,
                unit.health,
                unit.attack_power,
                unit.attack_range,
                unit.backpack_capacity,
                unit.backpack,
                unit.level,
                unit.cooldown,
                tuple(sorted(unit.provided_fields)),
            )
            for unit in units
        )
    )


def _hash_key(key: tuple[object, ...]) -> str:
    return sha256(repr(key).encode("utf-8")).hexdigest()
