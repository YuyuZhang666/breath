from dataclasses import dataclass, replace

from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase

from .defense import core_weapon_readiness
from .features import StrategyFeatures, extract_features
from .forecast import NightForecast, RiskLevel
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
from .safety import (
    CheapestSafePlan,
    SafetyPlanStatus,
    find_cheapest_safe_plan,
)


@dataclass(frozen=True, slots=True)
class DirectorConfig:
    evaluation_interval: int = 5
    night_evaluation_interval: int = 1
    score_gold_floor: int = 75
    desperation_station_health: int = 100
    pressure_enemy_station_health: int = 200
    required_personal_roles: int = 1

    def __post_init__(self) -> None:
        if self.required_personal_roles < 0:
            raise ValueError('required_personal_roles cannot be negative')
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")
        if self.night_evaluation_interval < 1:
            raise ValueError("night_evaluation_interval must be positive")
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
    safety_plan: CheapestSafePlan | None = None


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
        forecast: NightForecast | None = None,
    ) -> DirectorDecision:
        planning_profile = (
            previous_state.profile
            if previous_state is not None
            else StrategyProfile.ECONOMY
        )
        planning_intent = intent_for_profile(
            planning_profile,
            'safety plan evaluation',
            feature_flags=self._feature_flags,
        )
        safety_plan = find_cheapest_safe_plan(
            observation,
            forecast,
            planning_intent.build_plan,
        )
        features = extract_features(
            observation,
            previous_observation=previous_observation,
            certificate=certificate,
            forecast=forecast,
            safety_plan=safety_plan,
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
            intent = _apply_dynamic_reserve(
                intent,
                observation,
                forecast,
                safety_plan,
            )
            intent = _apply_opening_defense_reserve(intent, observation)
            state = DirectorState(
                profile=previous_state.profile,
                since_round=previous_state.since_round,
                last_evaluated_round=previous_state.last_evaluated_round,
                reason=previous_state.reason,
                features=features,
            )
            return DirectorDecision(intent, state, features, safety_plan)

        profile, reason = self._candidate(features, forecast, safety_plan)
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
        intent = _apply_dynamic_reserve(
            intent,
            observation,
            forecast,
            safety_plan,
        )
        intent = _apply_opening_defense_reserve(intent, observation)
        state = DirectorState(
            profile=profile,
            since_round=since_round,
            last_evaluated_round=features.round_no,
            reason=reason,
            features=features,
        )
        return DirectorDecision(intent, state, features, safety_plan)

    def _should_evaluate(
        self,
        previous: DirectorState,
        current: StrategyFeatures,
    ) -> bool:
        previous_intent = intent_for_profile(
            previous.profile,
            previous.reason,
            feature_flags=self._feature_flags,
        )
        hold_expires = previous.since_round + previous_intent.minimum_hold_rounds
        if previous.last_evaluated_round < hold_expires <= current.round_no:
            return True
        interval = (
            self._config.night_evaluation_interval
            if current.phase is Phase.NIGHT
            else self._config.evaluation_interval
        )
        if current.round_no - previous.last_evaluated_round >= interval:
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
                current.targeted_robot_one_turn_attack_power
                != before.targeted_robot_one_turn_attack_power,
                current.wave_classification is not before.wave_classification,
                current.wave_secured != before.wave_secured,
                current.night_risk_level is not before.night_risk_level,
                current.risk_ratio != before.risk_ratio,
                current.survival_margin != before.survival_margin,
                current.safety_plan_cost != before.safety_plan_cost,
                current.gold != before.gold,
                current.enemy_station_health != before.enemy_station_health,
            )
        )

    def _candidate(
        self,
        features: StrategyFeatures,
        forecast: NightForecast | None,
        safety_plan: CheapestSafePlan | None,
    ) -> tuple[StrategyProfile, str]:
        if forecast is not None and forecast.risk_level is not RiskLevel.UNKNOWN:
            if forecast.risk_level is RiskLevel.LETHAL:
                return StrategyProfile.DESPERATION, 'forecast lethal within two rounds'
            if (
                forecast.risk_level is RiskLevel.CRITICAL
                or forecast.survival_margin < 0
            ):
                return StrategyProfile.SURVIVE, 'forecast critical survival risk'
            pressure_ready = (
                forecast.risk_level is RiskLevel.SAFE
                and self._feature_flags.enable_pressure
                and features.enemy_station_health is not None
                and features.enemy_station_health
                <= self._config.pressure_enemy_station_health
                and features.defense_complete
            )
            if pressure_ready:
                return StrategyProfile.PRESSURE, 'safe forecast and visible enemy weakness'
            if (
                safety_plan is not None
                and safety_plan.status is SafetyPlanStatus.UNAVAILABLE
            ):
                return StrategyProfile.SURVIVE, 'no verified safe plan available'
            if (
                not features.defense_complete
                or (
                    safety_plan is not None
                    and safety_plan.status is SafetyPlanStatus.FOUND
                )
            ):
                return StrategyProfile.ECONOMY, 'fund verified safety plan'
            return StrategyProfile.SCORE, 'forecast survival secured'

        station_health = features.our_station_health
        fatal_visible_threat = (
            station_health is not None
            and station_health <= self._config.desperation_station_health
            and features.targeted_robot_one_turn_attack_power >= station_health
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


def _apply_dynamic_reserve(
    intent: StrategicIntent,
    observation: Observation,
    forecast: NightForecast | None,
    safety_plan: CheapestSafePlan | None,
) -> StrategicIntent:
    if forecast is None or safety_plan is None:
        return intent
    if intent.profile is StrategyProfile.DESPERATION:
        return replace(
            intent,
            gold_reserve=0,
            reserve_eligible_actions=frozenset(),
        )
    if safety_plan.status is SafetyPlanStatus.UNKNOWN:
        return intent
    if safety_plan.status is SafetyPlanStatus.ALREADY_SAFE:
        return replace(
            intent,
            gold_reserve=0,
            reserve_eligible_actions=frozenset(),
        )
    if safety_plan.status is SafetyPlanStatus.FOUND:
        return replace(
            intent,
            gold_reserve=safety_plan.gold_cost,
            reserve_eligible_actions=safety_plan.reserve_eligible_actions,
        )
    return replace(
        intent,
        gold_reserve=observation.our.gold,
        reserve_eligible_actions=frozenset(),
    )


def _apply_opening_defense_reserve(
    intent: StrategicIntent,
    observation: Observation,
) -> StrategicIntent:
    if (
        observation.time.day_no != 1
        or observation.time.phase is not Phase.DAY
        or not intent.build_plan.build_weapons
        or intent.profile is StrategyProfile.DESPERATION
    ):
        return intent

    readiness = core_weapon_readiness(
        observation.our.units,
        intent.build_plan.weapon_loadout,
    )
    missing = {
        ('build', weapon_type) for weapon_type in readiness.missing_types
    }
    if not missing:
        return intent
    return replace(
        intent,
        reserve_eligible_actions=(
            intent.reserve_eligible_actions | frozenset(missing)
        ),
    )
