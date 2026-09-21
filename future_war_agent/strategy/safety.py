from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations

from future_war_agent.protocol.models import Observation

from .forecast import NightForecast, RiskLevel, classify_risk
from .policy import BuildPlan
from .rules import DEFAULT_RULES, RulesConfig
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config


class SafetyPlanStatus(StrEnum):
    UNKNOWN = 'unknown'
    ALREADY_SAFE = 'already_safe'
    FOUND = 'found'
    UNAVAILABLE = 'unavailable'


@dataclass(frozen=True, slots=True)
class SafetyPlanAction:
    action_key: tuple[str, str]
    gold_cost: int
    defense_gain: int


@dataclass(frozen=True, slots=True)
class CheapestSafePlan:
    status: SafetyPlanStatus
    actions: tuple[SafetyPlanAction, ...]
    gold_cost: int
    projected_margin: int
    projected_risk_level: RiskLevel
    source_forecast_round: int

    @property
    def reserve_eligible_actions(self) -> frozenset[tuple[str, str]]:
        return frozenset(action.action_key for action in self.actions)


def find_cheapest_safe_plan(
    observation: Observation,
    forecast: NightForecast | None,
    build_plan: BuildPlan,
    *,
    rules: RulesConfig = DEFAULT_RULES,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> CheapestSafePlan | None:
    if forecast is None:
        return None
    if forecast.risk_level is RiskLevel.UNKNOWN or not forecast.complete:
        return CheapestSafePlan(
            status=SafetyPlanStatus.UNKNOWN,
            actions=(),
            gold_cost=0,
            projected_margin=forecast.survival_margin,
            projected_risk_level=RiskLevel.UNKNOWN,
            source_forecast_round=forecast.generated_round,
        )
    if forecast.risk_level is RiskLevel.SAFE:
        return CheapestSafePlan(
            status=SafetyPlanStatus.ALREADY_SAFE,
            actions=(),
            gold_cost=0,
            projected_margin=forecast.survival_margin,
            projected_risk_level=RiskLevel.SAFE,
            source_forecast_round=forecast.generated_round,
        )

    candidates, has_unverified_candidate = _verified_actions(
        observation,
        build_plan,
        rules=rules,
        config=config,
    )
    feasible: list[
        tuple[
            tuple[int, int, tuple[tuple[str, str], ...]],
            CheapestSafePlan,
        ]
    ] = []
    for size in range(1, len(candidates) + 1):
        for selected in combinations(candidates, size):
            gain = sum(action.defense_gain for action in selected)
            projected_damage = max(
                0,
                forecast.predicted_damage_before_dawn - gain,
            )
            projected_margin = forecast.survival_margin + gain
            projected_ratio = Fraction(
                projected_damage,
                max(1, forecast.effective_defense_hp),
            )
            projected_risk = classify_risk(
                projected_ratio,
                projected_margin,
                None,
                observation.time.round_no,
                complete=forecast.complete,
            )
            if projected_risk is not RiskLevel.SAFE:
                continue
            gold_cost = sum(action.gold_cost for action in selected)
            stable_keys = tuple(action.action_key for action in selected)
            plan = CheapestSafePlan(
                status=SafetyPlanStatus.FOUND,
                actions=tuple(selected),
                gold_cost=gold_cost,
                projected_margin=projected_margin,
                projected_risk_level=projected_risk,
                source_forecast_round=forecast.generated_round,
            )
            feasible.append(
                ((gold_cost, len(selected), stable_keys), plan)
            )
    if feasible:
        return min(feasible, key=lambda item: item[0])[1]
    if has_unverified_candidate:
        return CheapestSafePlan(
            status=SafetyPlanStatus.UNKNOWN,
            actions=(),
            gold_cost=0,
            projected_margin=forecast.survival_margin,
            projected_risk_level=RiskLevel.UNKNOWN,
            source_forecast_round=forecast.generated_round,
        )
    return CheapestSafePlan(
        status=SafetyPlanStatus.UNAVAILABLE,
        actions=(),
        gold_cost=observation.our.gold,
        projected_margin=forecast.survival_margin,
        projected_risk_level=forecast.risk_level,
        source_forecast_round=forecast.generated_round,
    )


def _verified_actions(
    observation: Observation,
    build_plan: BuildPlan,
    *,
    rules: RulesConfig,
    config: Phase3Config,
) -> tuple[tuple[SafetyPlanAction, ...], bool]:
    if not build_plan.build_weapons:
        return (), False
    living_counts = Counter(
        unit.role_type
        for unit in observation.our.units
        if unit.health > 0
    )
    actions: list[SafetyPlanAction] = []
    has_unverified_candidate = False
    for weapon_type in build_plan.weapon_loadout:
        if living_counts[weapon_type] > 0:
            living_counts[weapon_type] -= 1
            continue
        gain = _verified_weapon_gain(observation, weapon_type, config)
        if gain <= 0:
            has_unverified_candidate = True
            continue
        actions.append(
            SafetyPlanAction(
                action_key=('build', weapon_type),
                gold_cost=rules.weapon_build_cost,
                defense_gain=gain,
            )
        )
    return tuple(actions), has_unverified_candidate


def _verified_weapon_gain(
    observation: Observation,
    weapon_type: str,
    config: Phase3Config,
) -> int:
    if weapon_type == 'gatling':
        return config.gatling_damage
    if weapon_type == 'rocket':
        return config.rocket_center_damage
    if weapon_type == 'railgun':
        observed_energy = tuple(
            unit.attack_power
            for unit in observation.our.units
            if unit.health > 0
            and unit.role_type == 'railgun'
            and 'attackPower' in unit.provided_fields
        )
        return min(observed_energy, default=0)
    return 0
