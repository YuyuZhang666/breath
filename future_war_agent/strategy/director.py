from dataclasses import dataclass

from future_war_agent.protocol.models import Observation

from .features import StrategyFeatures, extract_features
from .policy import (
    RuleFeatureFlags,
    StrategicIntent,
    StrategyProfile,
    intent_for_profile,
)
from .simulation.certificate import (
    RobotWaveSafetyCertificate,
    WaveClassification,
)


@dataclass(frozen=True, slots=True)
class DirectorConfig:
    evaluation_interval: int = 5
    score_gold_floor: int = 75
    desperation_station_health: int = 100
    pressure_enemy_station_health: int = 200
    required_personal_roles: int = 1

    def __post_init__(self) -> None:
        if self.required_personal_roles < 0:
            raise ValueError('required_personal_roles cannot be negative')
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")
        if self.score_gold_floor < 0:
            raise ValueError("score_gold_floor cannot be negative")
        if self.desperation_station_health < 1:
            raise ValueError("desperation_station_health must be positive")
        if self.pressure_enemy_station_health < 1:
            raise ValueError("pressure_enemy_station_health must be positive")


DEFAULT_DIRECTOR_CONFIG = DirectorConfig()


@dataclass(frozen=True, slots=True)
class DirectorState:
    profile: StrategyProfile
    since_round: int
    last_evaluated_round: int
    reason: str
    features: StrategyFeatures


@dataclass(frozen=True, slots=True)
class DirectorDecision:
    intent: StrategicIntent
    state: DirectorState
    features: StrategyFeatures


class StrategicDirector:
    def __init__(
        self,
        *,
        config: DirectorConfig = DEFAULT_DIRECTOR_CONFIG,
        feature_flags: RuleFeatureFlags | None = None,
    ) -> None:
        self._config = config
        self._feature_flags = (
            feature_flags if feature_flags is not None else RuleFeatureFlags()
        )

    def select(
        self,
        observation: Observation,
        *,
        previous_state: DirectorState | None = None,
        previous_observation: Observation | None = None,
        certificate: RobotWaveSafetyCertificate | None = None,
    ) -> DirectorDecision:
        features = extract_features(
            observation,
            previous_observation=previous_observation,
            certificate=certificate,
        )
        if previous_state is not None and not self._should_evaluate(
            previous_state,
            features,
        ):
            intent = intent_for_profile(
                previous_state.profile,
                previous_state.reason,
                feature_flags=self._feature_flags,
            )
            state = DirectorState(
                profile=previous_state.profile,
                since_round=previous_state.since_round,
                last_evaluated_round=previous_state.last_evaluated_round,
                reason=previous_state.reason,
                features=features,
            )
            return DirectorDecision(intent, state, features)

        profile, reason = self._candidate(features)
        if previous_state is not None and profile not in {
            StrategyProfile.SURVIVE,
            StrategyProfile.DESPERATION,
        }:
            previous_intent = intent_for_profile(
                previous_state.profile,
                previous_state.reason,
                feature_flags=self._feature_flags,
            )
            held_rounds = features.round_no - previous_state.since_round
            if (
                profile is not previous_state.profile
                and held_rounds < previous_intent.minimum_hold_rounds
            ):
                profile = previous_state.profile
                reason = previous_state.reason

        since_round = features.round_no
        if previous_state is not None and profile is previous_state.profile:
            since_round = previous_state.since_round
        intent = intent_for_profile(
            profile,
            reason,
            feature_flags=self._feature_flags,
        )
        state = DirectorState(
            profile=profile,
            since_round=since_round,
            last_evaluated_round=features.round_no,
            reason=reason,
            features=features,
        )
        return DirectorDecision(intent, state, features)

    def _should_evaluate(
        self,
        previous: DirectorState,
        current: StrategyFeatures,
    ) -> bool:
        if current.round_no - previous.last_evaluated_round >= (
            self._config.evaluation_interval
        ):
            return True
        before = previous.features
        return any(
            (
                current.day_no != before.day_no,
                current.phase is not before.phase,
                current.station_health_loss > 0,
                current.living_personal_role_count
                != before.living_personal_role_count,
                current.living_weapon_count != before.living_weapon_count,
                current.defense_complete != before.defense_complete,
                current.targeted_robot_count != before.targeted_robot_count,
                current.targeted_robot_attack_power
                != before.targeted_robot_attack_power,
                current.wave_classification is not before.wave_classification,
                current.wave_secured != before.wave_secured,
                current.enemy_station_health != before.enemy_station_health,
            )
        )

    def _candidate(
        self,
        features: StrategyFeatures,
    ) -> tuple[StrategyProfile, str]:
        station_health = features.our_station_health
        fatal_visible_threat = (
            station_health is not None
            and station_health <= self._config.desperation_station_health
            and features.targeted_robot_attack_power >= station_health
        )
        if fatal_visible_threat:
            return StrategyProfile.DESPERATION, "visible fatal station threat"

        unsafe_wave = (
            features.wave_classification
            in {
                WaveClassification.WAVE_MARGINAL,
                WaveClassification.WAVE_UNSAFE,
            }
            or features.wave_secured is False
        )
        if (
            station_health is None
            or features.station_health_loss > 0
            or features.targeted_robot_count > 0
            or features.living_personal_role_count
            < self._config.required_personal_roles
            or unsafe_wave
        ):
            return StrategyProfile.SURVIVE, "material survival risk"

        pressure_ready = (
            self._feature_flags.enable_pressure
            and features.enemy_station_health is not None
            and features.enemy_station_health
            <= self._config.pressure_enemy_station_health
            and features.defense_complete
            and features.gold >= self._config.score_gold_floor
        )
        if pressure_ready:
            return StrategyProfile.PRESSURE, "visible enemy weakness"

        if (
            not features.defense_complete
            or features.gold < self._config.score_gold_floor
        ):
            return StrategyProfile.ECONOMY, "defense or reserve incomplete"
        return StrategyProfile.SCORE, "survival and reserve secured"
