import json
import logging
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from threading import RLock
from time import perf_counter_ns


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    team_id: str = ''
    round_no: int = 0
    phase: str = 'unknown'
    parser_ms: float = 0.0
    director_ms: float = 0.0
    task_ms: float = 0.0
    treasure_ms: float = 0.0
    phase2_ms: float = 0.0
    phase2_5_ms: float = 0.0
    phase3_ms: float = 0.0
    forecast_ms: float = 0.0
    tail_estimator_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    simulation_ms: float = 0.0
    validation_ms: float = 0.0
    serialization_ms: float = 0.0
    total_ms: float = 0.0
    request_total_ms: float = 0.0
    remaining_deadline_ms: float = 0.0
    root_candidate_count: int = 0
    scenario_count: int = 0
    simulation_count: int = 0
    phase3_level: str = 'none'
    phase3_effective_level: str = 'none'
    forecast_update_kind: str = 'none'
    forecast_mode: str = 'none'
    forecast_reason: str = ''
    night_risk_level: str = 'unknown'
    risk_ratio: float = 0.0
    survival_margin: int = 0
    safety_plan_status: str = 'unknown'
    safety_plan_cost: int = 0
    effective_gold_reserve: int = 0
    task_candidate_count: int = 0
    task_sop_hit: bool = False
    treasure_candidate_count: int = 0
    treasure_attempted: bool = False
    treasure_result: int = 0
    opponent_visible_structure_count: int = 0
    opponent_visible_role_count: int = 0
    opponent_defense_growth_per_100_rounds: float = 0.0
    opponent_task_tendency: str = 'unknown'
    opponent_task_proximity_observations: int = 0
    opponent_hostile_robot_count: int = 0
    opponent_summon_attribution: str = 'unknown'
    opponent_last_observed_damage: int = 0
    opponent_half_index: int = 0
    opponent_structure_log: tuple[str, ...] = ()
    opponent_role_log: tuple[str, ...] = ()
    opponent_evidence_log: tuple[str, ...] = ()
    capability_unknown_count: int = 0
    capability_supported_count: int = 0
    capability_unsupported_count: int = 0
    capability_log: tuple[str, ...] = ()
    compute_governor_action: str = 'normal'
    governor_root_limit: int = 0
    governor_scenario_limit: int = 0
    phase3_disabled_until_round: int = 0
    ewma_phase2_5_ms: float = 0.0
    ewma_root_rollout_ms: float = 0.0
    phase3_fallback_count: int = 0
    phase2_5_fallback_count: int = 0
    phase2_5_combination_count: int = 0
    phase2_5_active_weapon_count: int = 0
    fallback_used: bool = False
    safe_action_generated: bool = False
    fallback_reason: str = 'normal'
    timeout_prevented: bool = False
    decision_source: str = 'safe'
    response_action_count: int = 0
    emergency_fire_ms: float = 0.0
    emergency_fire_action_count: int = 0
    emergency_fire_deadline_hit: bool = False
    watchdog_hit: bool = False
    controller_cache_hit: bool = False
    duplicate_request: bool = False
    forecast_cache_hit: bool = False


@dataclass(slots=True)
class _ActiveTurn:
    started_ns: int
    team_id: str = ''
    round_no: int = 0
    phase: str = 'unknown'
    parser_ms: float = 0.0
    director_ms: float = 0.0
    task_ms: float = 0.0
    treasure_ms: float = 0.0
    phase2_ms: float = 0.0
    phase2_5_ms: float = 0.0
    phase3_ms: float = 0.0
    forecast_ms: float = 0.0
    tail_estimator_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    simulation_ms: float = 0.0
    validation_ms: float = 0.0
    serialization_ms: float = 0.0
    request_total_ms: float = 0.0
    remaining_deadline_ms: float = 0.0
    root_candidate_count: int = 0
    scenario_count: int = 0
    simulation_count: int = 0
    phase3_level: str = 'none'
    phase3_effective_level: str = 'none'
    forecast_update_kind: str = 'none'
    forecast_mode: str = 'none'
    forecast_reason: str = ''
    night_risk_level: str = 'unknown'
    risk_ratio: float = 0.0
    survival_margin: int = 0
    safety_plan_status: str = 'unknown'
    safety_plan_cost: int = 0
    effective_gold_reserve: int = 0
    task_candidate_count: int = 0
    task_sop_hit: bool = False
    treasure_candidate_count: int = 0
    treasure_attempted: bool = False
    treasure_result: int = 0
    opponent_visible_structure_count: int = 0
    opponent_visible_role_count: int = 0
    opponent_defense_growth_per_100_rounds: float = 0.0
    opponent_task_tendency: str = 'unknown'
    opponent_task_proximity_observations: int = 0
    opponent_hostile_robot_count: int = 0
    opponent_summon_attribution: str = 'unknown'
    opponent_last_observed_damage: int = 0
    opponent_half_index: int = 0
    opponent_structure_log: tuple[str, ...] = ()
    opponent_role_log: tuple[str, ...] = ()
    opponent_evidence_log: tuple[str, ...] = ()
    capability_unknown_count: int = 0
    capability_supported_count: int = 0
    capability_unsupported_count: int = 0
    capability_log: tuple[str, ...] = ()
    compute_governor_action: str = 'normal'
    governor_root_limit: int = 0
    governor_scenario_limit: int = 0
    phase3_disabled_until_round: int = 0
    ewma_phase2_5_ms: float = 0.0
    ewma_root_rollout_ms: float = 0.0
    phase3_fallback_count: int = 0
    phase2_5_fallback_count: int = 0
    phase2_5_combination_count: int = 0
    phase2_5_active_weapon_count: int = 0
    fallback_used: bool = False
    safe_action_generated: bool = False
    fallback_reason: str = 'normal'
    timeout_prevented: bool = False
    decision_source: str = 'safe'
    response_action_count: int = 0
    emergency_fire_ms: float = 0.0
    emergency_fire_action_count: int = 0
    emergency_fire_deadline_hit: bool = False
    watchdog_hit: bool = False
    controller_cache_hit: bool = False
    duplicate_request: bool = False
    forecast_cache_hit: bool = False


class TelemetryRecorder:
    def __init__(
        self,
        *,
        clock: Callable[[], int] = perf_counter_ns,
        max_samples: int = 1024,
    ) -> None:
        if max_samples <= 0:
            raise ValueError('max_samples must be positive')
        self._clock = clock
        self._active: ContextVar[_ActiveTurn | None] = ContextVar(
            f'turn_telemetry_{id(self)}',
            default=None,
        )
        self._samples: deque[TurnTelemetry] = deque(maxlen=max_samples)
        self._lock = RLock()

    def begin(self) -> Token[_ActiveTurn | None] | None:
        if self._active.get() is not None:
            return None
        return self._active.set(_ActiveTurn(started_ns=self._clock()))

    def identify(self, *, team_id: str, round_no: int, phase: str) -> None:
        active = self._active.get()
        if active is None:
            return
        active.team_id = team_id
        active.round_no = round_no
        active.phase = phase

    @contextmanager
    def measure(self, field: str) -> Iterator[None]:
        active = self._active.get()
        if active is None or not hasattr(active, field):
            yield
            return
        started = self._clock()
        try:
            yield
        finally:
            elapsed_ms = max(0, self._clock() - started) / 1_000_000
            setattr(active, field, getattr(active, field) + elapsed_ms)

    def set(self, **values: object) -> None:
        active = self._active.get()
        if active is None:
            return
        for name, value in values.items():
            if hasattr(active, name):
                setattr(active, name, value)

    def increment(self, field: str, amount: int = 1) -> None:
        active = self._active.get()
        if active is None or not hasattr(active, field):
            return
        setattr(active, field, int(getattr(active, field)) + amount)

    def current(self, field: str, default: object = None) -> object:
        active = self._active.get()
        if active is None or not hasattr(active, field):
            return default
        return getattr(active, field)

    def finish(
        self,
        token: Token[_ActiveTurn | None] | None,
    ) -> TurnTelemetry | None:
        if token is None:
            return None
        active = self._active.get()
        if active is None:
            self._active.reset(token)
            return None
        total_ms = max(0, self._clock() - active.started_ns) / 1_000_000
        sample = TurnTelemetry(
            team_id=active.team_id,
            round_no=active.round_no,
            phase=active.phase,
            parser_ms=active.parser_ms,
            director_ms=active.director_ms,
            task_ms=active.task_ms,
            treasure_ms=active.treasure_ms,
            phase2_ms=active.phase2_ms,
            phase2_5_ms=active.phase2_5_ms,
            phase3_ms=active.phase3_ms,
            forecast_ms=active.forecast_ms,
            tail_estimator_ms=active.tail_estimator_ms,
            candidate_generation_ms=active.candidate_generation_ms,
            simulation_ms=active.simulation_ms,
            validation_ms=active.validation_ms,
            serialization_ms=active.serialization_ms,
            total_ms=total_ms,
            request_total_ms=active.request_total_ms,
            remaining_deadline_ms=active.remaining_deadline_ms,
            root_candidate_count=active.root_candidate_count,
            scenario_count=active.scenario_count,
            simulation_count=active.simulation_count,
            phase3_level=active.phase3_level,
            phase3_effective_level=active.phase3_effective_level,
            forecast_update_kind=active.forecast_update_kind,
            forecast_mode=active.forecast_mode,
            forecast_reason=active.forecast_reason,
            night_risk_level=active.night_risk_level,
            risk_ratio=active.risk_ratio,
            survival_margin=active.survival_margin,
            safety_plan_status=active.safety_plan_status,
            safety_plan_cost=active.safety_plan_cost,
            effective_gold_reserve=active.effective_gold_reserve,
            task_candidate_count=active.task_candidate_count,
            task_sop_hit=active.task_sop_hit,
            treasure_candidate_count=active.treasure_candidate_count,
            treasure_attempted=active.treasure_attempted,
            treasure_result=active.treasure_result,
            opponent_visible_structure_count=(
                active.opponent_visible_structure_count
            ),
            opponent_visible_role_count=active.opponent_visible_role_count,
            opponent_defense_growth_per_100_rounds=(
                active.opponent_defense_growth_per_100_rounds
            ),
            opponent_task_tendency=active.opponent_task_tendency,
            opponent_task_proximity_observations=(
                active.opponent_task_proximity_observations
            ),
            opponent_hostile_robot_count=active.opponent_hostile_robot_count,
            opponent_summon_attribution=active.opponent_summon_attribution,
            opponent_last_observed_damage=active.opponent_last_observed_damage,
            opponent_half_index=active.opponent_half_index,
            opponent_structure_log=active.opponent_structure_log,
            opponent_role_log=active.opponent_role_log,
            opponent_evidence_log=active.opponent_evidence_log,
            capability_unknown_count=active.capability_unknown_count,
            capability_supported_count=active.capability_supported_count,
            capability_unsupported_count=active.capability_unsupported_count,
            capability_log=active.capability_log,
            compute_governor_action=active.compute_governor_action,
            governor_root_limit=active.governor_root_limit,
            governor_scenario_limit=active.governor_scenario_limit,
            phase3_disabled_until_round=active.phase3_disabled_until_round,
            ewma_phase2_5_ms=active.ewma_phase2_5_ms,
            ewma_root_rollout_ms=active.ewma_root_rollout_ms,
            phase3_fallback_count=active.phase3_fallback_count,
            phase2_5_fallback_count=active.phase2_5_fallback_count,
            phase2_5_combination_count=active.phase2_5_combination_count,
            phase2_5_active_weapon_count=active.phase2_5_active_weapon_count,
            fallback_used=active.fallback_used,
            safe_action_generated=active.safe_action_generated,
            fallback_reason=active.fallback_reason,
            timeout_prevented=active.timeout_prevented,
            decision_source=active.decision_source,
            response_action_count=active.response_action_count,
            emergency_fire_ms=active.emergency_fire_ms,
            emergency_fire_action_count=active.emergency_fire_action_count,
            emergency_fire_deadline_hit=active.emergency_fire_deadline_hit,
            watchdog_hit=active.watchdog_hit,
            controller_cache_hit=active.controller_cache_hit,
            duplicate_request=active.duplicate_request,
            forecast_cache_hit=active.forecast_cache_hit,
        )
        self._active.reset(token)
        with self._lock:
            self._samples.append(sample)
        try:
            LOGGER.info(
                'turn_telemetry %s',
                json.dumps(asdict(sample), sort_keys=True, separators=(',', ':')),
            )
        except Exception:
            LOGGER.exception('turn telemetry logging failed')
        return sample

    def snapshot(self) -> tuple[TurnTelemetry, ...]:
        with self._lock:
            return tuple(self._samples)


DEFAULT_TELEMETRY = TelemetryRecorder()
