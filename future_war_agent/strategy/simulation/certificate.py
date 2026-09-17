from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from .objective import NightObjective


class WaveClassification(StrEnum):
    WAVE_SAFE = "wave_safe"
    WAVE_MARGINAL = "wave_marginal"
    WAVE_UNSAFE = "wave_unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    weight: Fraction
    station_health: int
    controller_losses: int
    key_weapon_losses: int
    minimum_role_health: int
    surviving_asset_health: int
    owned_kill_score: int
    remaining_threat: int
    remaining_one_turn_damage: int
    ended_with_night: bool


@dataclass(frozen=True, slots=True)
class RobotWaveSafetyCertificate:
    classification: WaveClassification
    secured: bool
    station_survival_probability: Fraction
    expected_station_health: Fraction
    p10_station_health: int
    worst_station_health: int
    worst_controller_losses: int
    worst_key_weapon_losses: int
    worst_minimum_role_health: int
    expected_owned_kill_score: Fraction
    expected_remaining_threat: Fraction
    outcomes: tuple[ScenarioOutcome, ...]


def build_certificate(
    outcomes: tuple[ScenarioOutcome, ...],
    objective: NightObjective,
) -> RobotWaveSafetyCertificate:
    if not outcomes:
        raise ValueError("certificate outcomes cannot be empty")
    if any(outcome.weight <= 0 for outcome in outcomes):
        raise ValueError("scenario weights must be positive")
    if sum((outcome.weight for outcome in outcomes), Fraction()) != 1:
        raise ValueError("scenario weights must sum to one")

    station_survival_probability = sum(
        (
            outcome.weight
            for outcome in outcomes
            if outcome.station_health > 0
        ),
        Fraction(),
    )
    expected_station_health = sum(
        (outcome.weight * outcome.station_health for outcome in outcomes),
        Fraction(),
    )
    expected_owned_kill_score = sum(
        (outcome.weight * outcome.owned_kill_score for outcome in outcomes),
        Fraction(),
    )
    expected_remaining_threat = sum(
        (outcome.weight * outcome.remaining_threat for outcome in outcomes),
        Fraction(),
    )
    secured = all(_outcome_is_secured(outcome, objective) for outcome in outcomes)
    if secured:
        classification = WaveClassification.WAVE_SAFE
    elif station_survival_probability == 1:
        classification = WaveClassification.WAVE_MARGINAL
    else:
        classification = WaveClassification.WAVE_UNSAFE

    return RobotWaveSafetyCertificate(
        classification=classification,
        secured=secured,
        station_survival_probability=station_survival_probability,
        expected_station_health=expected_station_health,
        p10_station_health=_weighted_p10(outcomes),
        worst_station_health=min(outcome.station_health for outcome in outcomes),
        worst_controller_losses=max(
            outcome.controller_losses for outcome in outcomes
        ),
        worst_key_weapon_losses=max(
            outcome.key_weapon_losses for outcome in outcomes
        ),
        worst_minimum_role_health=min(
            outcome.minimum_role_health for outcome in outcomes
        ),
        expected_owned_kill_score=expected_owned_kill_score,
        expected_remaining_threat=expected_remaining_threat,
        outcomes=outcomes,
    )


def survival_rank_key(
    certificate: RobotWaveSafetyCertificate,
    root_stable_key: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        -certificate.station_survival_probability,
        -certificate.p10_station_health,
        -certificate.worst_station_health,
        certificate.worst_controller_losses,
        certificate.worst_key_weapon_losses,
        -certificate.worst_minimum_role_health,
        -certificate.expected_station_health,
        certificate.expected_remaining_threat,
        -certificate.expected_owned_kill_score,
        root_stable_key,
    )


def score_rank_key(
    certificate: RobotWaveSafetyCertificate,
    root_stable_key: tuple[object, ...],
) -> tuple[object, ...]:
    worst_surviving_asset_health = min(
        outcome.surviving_asset_health for outcome in certificate.outcomes
    )
    return (
        -certificate.expected_owned_kill_score,
        certificate.expected_remaining_threat,
        -certificate.worst_station_health,
        -certificate.worst_minimum_role_health,
        -worst_surviving_asset_health,
        root_stable_key,
    )


def _outcome_is_secured(
    outcome: ScenarioOutcome,
    objective: NightObjective,
) -> bool:
    if outcome.station_health < objective.minimum_station_health:
        return False
    if objective.protect_all_controllers and outcome.controller_losses != 0:
        return False
    if objective.protect_all_weapons and outcome.key_weapon_losses != 0:
        return False
    if (
        objective.require_post_horizon_buffer
        and not outcome.ended_with_night
        and outcome.station_health - outcome.remaining_one_turn_damage
        < objective.minimum_station_health
    ):
        return False
    return True


def _weighted_p10(outcomes: tuple[ScenarioOutcome, ...]) -> int:
    cumulative = Fraction()
    for outcome in sorted(
        outcomes,
        key=lambda item: (
            item.station_health,
            item.controller_losses,
            item.key_weapon_losses,
            item.remaining_threat,
        ),
    ):
        cumulative += outcome.weight
        if cumulative >= Fraction(1, 10):
            return outcome.station_health
    raise AssertionError("weights were validated to sum to one")
