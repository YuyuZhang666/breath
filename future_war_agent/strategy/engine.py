import logging
from collections.abc import Callable
from inspect import Parameter, signature
from threading import RLock
from time import monotonic, perf_counter_ns

from future_war_agent.deadline import RequestBudget, RequestDeadlineExceeded
from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.time import Phase
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .compute import ComputeGovernor, ComputeTurnUsage
from .build_recovery import EMPTY_BUILD_RECOVERY_STATE, BuildRecoveryState
from .director import StrategicDirector
from .features import extract_features
from .forecast import (
    ForecastUpdateKind,
    ForecastRefresh,
    NightForecast,
    RiskLevel,
    forecast_recompute_reason,
    rebase_day_forecast,
    refresh_night_forecast,
)
from .items import has_item
from .market import EMPTY_MARKET_STATE, MarketMemory, MarketView
from .memory import MatchMemoryStore
from .night import (
    ControllerAssignment,
    ControllerAssignmentCache,
    EmergencyFirePlan,
    plan_emergency_night_fire,
)
from .planner import plan_turn
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent, StrategyProfile
from .reconcile import reconcile_scenario_weights, uniform_scenario_weights
from .session import (
    SessionContinuity,
    SessionStore,
    StrategySession,
    classify_continuity,
    observation_fingerprint,
    static_signature,
)
from .simulation.config import (
    DEFAULT_PHASE3_CONFIG,
    Phase3Config,
    Phase3Level,
)
from .simulation.errors import DeadlineExceeded, UnsupportedSimulation
from .simulation.objective import NightObjective
from .simulation.search import (
    ScenarioWeights,
    SearchResult,
    search_night,
)
from .simulation.trigger import select_phase3_level
from .task_agent import EMPTY_TASK_STATE, TaskAgent
from .treasure import EMPTY_TREASURE_STATE, TreasureAgent
from .world import WorldGrid


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Phase2Planner = Callable[..., Decision]
NightSearcher = Callable[..., SearchResult]
ObjectiveProvider = Callable[[Observation], NightObjective]
ScenarioReconciler = Callable[..., ScenarioWeights]
NightForecaster = Callable[..., ForecastRefresh]
EmergencyPlanner = Callable[..., EmergencyFirePlan]


class StrategyEngine:
    def __init__(
        self,
        *,
        config: Phase3Config = DEFAULT_PHASE3_CONFIG,
        phase2_planner: Phase2Planner = plan_turn,
        night_searcher: NightSearcher = search_night,
        objective_provider: ObjectiveProvider | None = None,
        director: StrategicDirector | None = None,
        task_agent: TaskAgent | None = None,
        treasure_agent: TreasureAgent | None = None,
        memory_store: MatchMemoryStore | None = None,
        clock: Callable[[], float] = monotonic,
        session_store: SessionStore | None = None,
        scenario_reconciler: ScenarioReconciler = reconcile_scenario_weights,
        controller_assignment_cache: ControllerAssignmentCache | None = None,
        night_forecaster: NightForecaster = refresh_night_forecast,
        telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
        compute_governor: ComputeGovernor | None = None,
        market_memory: MarketMemory | None = None,
        emergency_planner: EmergencyPlanner = plan_emergency_night_fire,
    ) -> None:
        self._config = config
        self._phase2_planner = phase2_planner
        self._phase2_accepts_intent = _accepts_intent(phase2_planner)
        self._phase2_accepts_assignments = _accepts_keyword(
            phase2_planner,
            'controller_assignments',
        )
        self._phase2_accepts_telemetry = _accepts_keyword(
            phase2_planner,
            'telemetry',
        )
        self._phase2_accepts_fortification_threats = _accepts_keyword(
            phase2_planner,
            'fortification_threats',
        )
        self._night_searcher = night_searcher
        self._search_accepts_assignments = _accepts_keyword(
            night_searcher,
            'controller_assignments',
        )
        self._search_accepts_level = _accepts_keyword(night_searcher, 'level')
        self._search_accepts_intent = _accepts_keyword(night_searcher, 'intent')
        self._search_accepts_baseline = _accepts_keyword(
            night_searcher,
            'baseline_decision',
        )
        self._search_accepts_budget = _accepts_keyword(
            night_searcher,
            'budget',
        )
        self._objective_provider = objective_provider
        self._director = director if director is not None else StrategicDirector()
        self._task_agent = task_agent if task_agent is not None else TaskAgent()
        self._task_accepts_survival_interrupt = _accepts_keyword(
            self._task_agent.apply,
            'force_survival_interrupt',
        )
        self._phase2_accepts_expected_wall_losses = _accepts_keyword(
            phase2_planner,
            'expected_wall_losses',
        )
        self._phase2_accepts_market_view = _accepts_keyword(
            phase2_planner,
            'market_view',
        )
        self._phase2_accepts_previous_decision = _accepts_keyword(
            phase2_planner,
            'previous_decision',
        )
        self._phase2_accepts_previously_built_walls = _accepts_keyword(
            phase2_planner,
            'previously_built_wall_sites',
        )
        self._phase2_accepts_build_recovery = _accepts_keyword(
            phase2_planner,
            'build_recovery',
        )
        self._treasure_agent = (
            treasure_agent if treasure_agent is not None else TreasureAgent()
        )
        self._memory = (
            memory_store if memory_store is not None else MatchMemoryStore()
        )
        self._clock = clock
        self._sessions = session_store if session_store is not None else SessionStore()
        self._scenario_reconciler = scenario_reconciler
        self._controller_assignments = (
            controller_assignment_cache
            if controller_assignment_cache is not None
            else ControllerAssignmentCache()
        )
        self._night_forecaster = night_forecaster
        self._forecast_accepts_assignments = _accepts_keyword(
            night_forecaster,
            'controller_assignments',
        )
        self._forecast_accepts_clock = _accepts_keyword(
            night_forecaster,
            'clock',
        )
        self._forecast_accepts_deadline = _accepts_keyword(
            night_forecaster,
            'deadline',
        )
        self._forecast_accepts_allow_full = _accepts_keyword(
            night_forecaster,
            'allow_full',
        )
        self._telemetry = telemetry
        self._compute_governor = (
            compute_governor
            if compute_governor is not None
            else ComputeGovernor()
        )
        self._market = (
            market_memory if market_memory is not None else MarketMemory()
        )
        self._emergency_planner = emergency_planner
        self._lock = RLock()

    def plan(
        self,
        observation: Observation,
        *,
        request_started_at: float | None = None,
        request_budget: RequestBudget | None = None,
    ) -> Decision:
        if request_budget is None:
            if request_started_at is None:
                request_started_at = self._clock()
            config = self._compute_governor.config
            request_budget = RequestBudget.start(
                started_at=request_started_at,
                response_budget_seconds=(
                    config.total_decision_budget_seconds
                    + config.emergency_reserve_seconds
                ),
                compute_budget_seconds=config.total_decision_budget_seconds,
                clock=self._clock,
            )
        compute_usage = ComputeTurnUsage()
        token = self._telemetry.begin()
        self._telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
        try:
            reserve = self._compute_governor.config.emergency_reserve_seconds
            remaining = request_budget.remaining_compute()
            if remaining <= reserve:
                return self._plan_emergency(
                    observation,
                    request_budget,
                    deadline_stage='engine_entry',
                )
            lock_timeout = remaining - reserve
            with self._telemetry.measure('engine_lock_wait_ms'):
                lock_acquired = self._lock.acquire(timeout=lock_timeout)
            if not lock_acquired:
                self._telemetry.set(
                    engine_lock_timed_out=True,
                    deadline_stage='engine_lock',
                )
                return self._plan_emergency(
                    observation,
                    request_budget,
                    deadline_stage='engine_lock',
                )
            try:
                if request_budget.remaining_compute() <= reserve:
                    self._lock.release()
                    lock_acquired = False
                    return self._plan_emergency(
                        observation,
                        request_budget,
                        deadline_stage='post_lock',
                    )
                try:
                    return self._plan_locked(
                        observation,
                        request_budget,
                        compute_usage,
                    )
                finally:
                    try:
                        self._compute_governor.observe_turn(
                            observation.our.team_id,
                            round_no=observation.time.round_no,
                            day_no=observation.time.day_no,
                            phase2_5_ms=compute_usage.phase2_5_ms,
                            phase3_ms=compute_usage.phase3_ms,
                            roots_evaluated=compute_usage.roots_evaluated,
                            scenarios_per_root=compute_usage.scenarios_per_root,
                            exact_horizon=compute_usage.exact_horizon,
                            watchdog_hit=compute_usage.watchdog_hit,
                            completed_phase3_ms=(
                                compute_usage.completed_phase3_ms
                            ),
                        )
                    except Exception:
                        LOGGER.exception(
                            'ComputeGovernor update failed for team %s round %s',
                            observation.our.team_id,
                            observation.time.round_no,
                        )
            finally:
                if lock_acquired:
                    self._lock.release()
        finally:
            self._telemetry.finish(token)

    def _plan_emergency(
        self,
        observation: Observation,
        request_budget: RequestBudget,
        *,
        deadline_stage: str = 'strategy',
    ) -> Decision:
        config = self._compute_governor.config
        self._telemetry.set(
            safe_action_generated=True,
            compute_governor_action='emergency_reserve',
            phase3_level=Phase3Level.NONE.value,
            fallback_used=True,
            fallback_reason='deadline_low',
            timeout_prevented=True,
            decision_source='safe',
            deadline_stage=deadline_stage,
        )
        if observation.time.phase is not Phase.NIGHT:
            return Decision()
        self._telemetry.set(
            forecast_mode='skipped',
            forecast_reason='deadline_low',
        )
        now = request_budget.clock()
        emergency_deadline = min(
            request_budget.compute_deadline
            - config.emergency_postprocess_guard_seconds,
            now + config.emergency_fire_budget_seconds,
        )
        if now >= emergency_deadline:
            return Decision()
        try:
            with self._telemetry.measure('emergency_fire_ms'):
                plan = self._emergency_planner(
                    observation,
                    clock=request_budget.clock,
                    deadline=emergency_deadline,
                )
        except Exception:
            LOGGER.exception(
                'Emergency fire failed for team %s round %s',
                observation.our.team_id,
                observation.time.round_no,
            )
            return Decision()
        action_count = len(plan.decision.commands)
        self._telemetry.set(
            emergency_fire_action_count=action_count,
            emergency_fire_deadline_hit=plan.deadline_hit,
            decision_source=('emergency_fire' if action_count else 'safe'),
        )
        return plan.decision

    def current_profile(self, team_id: str) -> StrategyProfile | None:
        with self._lock:
            session = self._sessions.get(team_id)
            return session.intent.profile if session is not None else None

    def _plan_locked(
        self,
        observation: Observation,
        request_budget: RequestBudget,
        compute_usage: ComputeTurnUsage,
    ) -> Decision:
        team_id = observation.our.team_id
        request_deadline = request_budget.compute_deadline
        reserve = self._compute_governor.config.emergency_reserve_seconds
        governor_emergency = request_budget.remaining_compute() <= reserve
        self._telemetry.set(
            safe_action_generated=True,
            decision_source='safe',
        )
        if governor_emergency and observation.time.phase is Phase.NIGHT:
            return self._plan_emergency(observation, request_budget)
        if not team_id.strip():
            if governor_emergency:
                self._telemetry.set(
                    fallback_used=True,
                    fallback_reason='deadline_low',
                    timeout_prevented=True,
                )
                return Decision()
            return self._plan_phase2(observation, DEFAULT_STRATEGIC_INTENT)

        fingerprint = observation_fingerprint(observation)
        signature = static_signature(observation)
        previous = self._sessions.get(team_id)
        own_station_status = _own_station_status(observation, previous)
        own_station_alive = own_station_status == 'alive'
        self._telemetry.set(
            own_station_alive=own_station_alive,
            own_station_status=own_station_status,
        )
        continuity = (
            classify_continuity(previous, observation)
            if previous is not None
            else SessionContinuity.DISCONTINUITY
        )
        if (
            previous is not None
            and continuity is SessionContinuity.DISCONTINUITY
            and (
                observation.time.round_no <= previous.last_round
                or signature != previous.signature
            )
        ):
            self._compute_governor.reset(team_id)
        if continuity is SessionContinuity.DUPLICATE:
            if previous is None:
                raise AssertionError("duplicate continuity requires a session")
            self._telemetry.set(duplicate_request=True)
            return previous.decision

        match_memory = None
        try:
            match_memory = self._memory.observe(
                observation,
                clear_forecast=not own_station_alive,
                previous_decision=(
                    previous.decision
                    if previous is not None
                    and continuity is SessionContinuity.CONSECUTIVE
                    else None
                ),
            )
            opponent_memory = match_memory.opponent_memory
            (
                capability_unknown_count,
                capability_supported_count,
                capability_unsupported_count,
            ) = match_memory.capability_matrix.status_counts()
            self._telemetry.set(
                opponent_visible_structure_count=(
                    opponent_memory.visible_structure_count
                ),
                opponent_visible_role_count=opponent_memory.visible_role_count,
                opponent_defense_growth_per_100_rounds=float(
                    opponent_memory.defense_growth_per_100_rounds
                ),
                opponent_task_tendency=opponent_memory.task_tendency.value,
                opponent_task_proximity_observations=(
                    opponent_memory.task_proximity_observations
                ),
                opponent_hostile_robot_count=(
                    opponent_memory.visible_hostile_robot_count
                ),
                opponent_summon_attribution=(
                    opponent_memory.summon_attribution.value
                ),
                opponent_last_observed_damage=(
                    opponent_memory.last_observed_damage
                ),
                opponent_half_index=opponent_memory.half_index,
                opponent_structure_log=(
                    opponent_memory.structure_log_entries(
                        current_round=observation.time.round_no,
                    )
                ),
                opponent_role_log=opponent_memory.role_log_entries(
                    current_round=observation.time.round_no,
                ),
                opponent_evidence_log=(
                    opponent_memory.evidence_log_entries()
                ),
                capability_unknown_count=capability_unknown_count,
                capability_supported_count=capability_supported_count,
                capability_unsupported_count=capability_unsupported_count,
                capability_log=match_memory.capability_matrix.log_entries(),
                build_failure_count=len(
                    match_memory.build_recovery.failures
                ),
                build_failure_log=(
                    match_memory.build_recovery.log_entries(
                        observation.time.round_no
                    )
                ),
                build_cooldown_count=sum(
                    observation.time.round_no
                    <= record.cooldown_until_round
                    for record in match_memory.build_recovery.failures
                ),
                build_reroute_count=sum(
                    record.consecutive_failures >= 2
                    for record in match_memory.build_recovery.failures
                ),
            )
        except Exception:
            LOGGER.exception(
                'Phase 6 memory update failed for team %s round %s',
                team_id,
                observation.time.round_no,
            )

        weights = (
            previous.scenario_weights
            if previous is not None
            and continuity is not SessionContinuity.DISCONTINUITY
            else uniform_scenario_weights()
        )
        previous_for_director = (
            previous
            if previous is not None
            and continuity is not SessionContinuity.DISCONTINUITY
            else None
        )
        previous_market_state = (
            previous.market_state
            if previous is not None
            and signature == previous.signature
            and observation.time.round_no >= previous.last_round
            else None
        )
        market_state = (
            previous_market_state
            if previous_market_state is not None
            else EMPTY_MARKET_STATE
        )
        market_view: MarketView | None = None
        try:
            market_state = self._market.observe(
                observation,
                previous_market_state,
            )
            market_view = self._market.view(observation, market_state)
            self._telemetry.set(
                market_signal_count=len(market_state.signals),
                market_hold_count=len(market_view.hold_items),
                market_sell_count=len(market_view.sell_items),
            )
        except Exception:
            self._telemetry.set(fallback_used=True)
            LOGGER.exception(
                'Market memory update failed for team %s round %s',
                team_id,
                observation.time.round_no,
            )
        controller_assignments: tuple[ControllerAssignment, ...] | None = None
        controller_assignment_mode: str | None = None
        controller_assignment_exclusions: frozenset[int] | None = None
        task_controller_exclusions = _active_task_controller_exclusions(
            observation
        )
        task_survival_interrupt = False
        task_assignment_failed = False
        night_forecast: NightForecast | None = (
            previous_for_director.night_forecast
            if previous_for_director is not None
            else None
        )
        forecast_invalidation_reason: str | None = None
        if observation.time.phase is Phase.NIGHT and not own_station_alive:
            night_forecast = None
            self._telemetry.set(
                forecast_update_kind='none',
                forecast_mode='skipped',
                forecast_reason=(
                    'own_station_destroyed'
                    if own_station_status == 'destroyed'
                    else 'own_station_state_invalid'
                ),
                forecast_margin_source='invalid',
            )
        elif observation.time.phase is Phase.NIGHT:
            previous_observation = (
                previous_for_director.observation
                if previous_for_director is not None
                else None
            )
            previous_decision = (
                previous_for_director.decision
                if previous_for_director is not None
                else None
            )
            forecast_invalidation_reason = forecast_recompute_reason(
                observation,
                previous_observation=previous_observation,
                previous_forecast=night_forecast,
                previous_decision=previous_decision,
            )
            if governor_emergency:
                self._telemetry.set(
                    compute_governor_action='emergency_reserve',
                    forecast_mode=('cached' if night_forecast is not None else 'skipped'),
                    forecast_reason='deadline_low',
                    fallback_used=True,
                    fallback_reason='deadline_low',
                    timeout_prevented=True,
                )
                if night_forecast is not None:
                    self._telemetry.set(
                        forecast_update_kind=night_forecast.update_kind.value,
                        forecast_cache_hit=True,
                        night_risk_level=night_forecast.risk_level.value,
                        risk_ratio=float(night_forecast.risk_ratio),
                        survival_margin=night_forecast.survival_margin,
                        **_forecast_provenance(
                            night_forecast,
                            observation.time.round_no,
                        ),
                    )
            elif forecast_invalidation_reason is not None:
                try:
                    request_budget.checkpoint('forecast controller assignment')
                    controller_assignments, cache_hit = (
                        self._controller_assignments.resolve(
                            observation,
                            WorldGrid.from_observation(observation),
                            mode_key=StrategyProfile.SURVIVE.value,
                            excluded_role_ids=task_controller_exclusions,
                        )
                    )
                    controller_assignment_mode = StrategyProfile.SURVIVE.value
                    controller_assignment_exclusions = (
                        task_controller_exclusions
                    )
                    self._telemetry.set(controller_cache_hit=cache_hit)
                except Exception:
                    if task_controller_exclusions:
                        task_assignment_failed = True
                    self._telemetry.set(fallback_used=True)
                    LOGGER.exception(
                        'forecast controller assignment failed for team %s round %s',
                        team_id,
                        observation.time.round_no,
                    )
            if not governor_emergency:
                try:
                    request_budget.checkpoint('night forecast')
                    forecast_deadline = request_budget.child_deadline(
                        (
                            self._compute_governor.config.forecast_full_watchdog_seconds
                            if self._compute_governor.config.night_full_forecast_enabled
                            else self._compute_governor.config.forecast_watchdog_seconds
                        )
                    )
                    forecast_kwargs = {
                        'previous_observation': previous_observation,
                        'previous_forecast': night_forecast,
                        'previous_decision': previous_decision,
                        'config': self._config,
                    }
                    if self._forecast_accepts_assignments:
                        forecast_kwargs['controller_assignments'] = (
                            controller_assignments
                        )
                    if self._forecast_accepts_clock:
                        forecast_kwargs['clock'] = request_budget.clock
                    if self._forecast_accepts_deadline:
                        forecast_kwargs['deadline'] = forecast_deadline
                    if self._forecast_accepts_allow_full:
                        forecast_kwargs['allow_full'] = (
                            self._compute_governor.config.night_full_forecast_enabled
                        )
                    with self._telemetry.measure('forecast_ms'):
                        forecast_refresh = self._night_forecaster(
                            observation,
                            **forecast_kwargs,
                        )
                    night_forecast = forecast_refresh.forecast
                    self._telemetry.set(
                        forecast_update_kind=night_forecast.update_kind.value,
                        forecast_mode=night_forecast.update_kind.value,
                        forecast_reason=forecast_refresh.reason,
                        forecast_cache_hit=not forecast_refresh.recomputed,
                        night_risk_level=night_forecast.risk_level.value,
                        risk_ratio=float(night_forecast.risk_ratio),
                        survival_margin=night_forecast.survival_margin,
                        **_forecast_provenance(
                            night_forecast,
                            observation.time.round_no,
                        ),
                    )
                except (DeadlineExceeded, RequestDeadlineExceeded):
                    compute_usage.watchdog_hit = True
                    self._telemetry.set(
                        fallback_used=True,
                        fallback_reason='forecast_timeout',
                        timeout_prevented=True,
                        watchdog_hit=True,
                        forecast_mode=(
                            'cached' if night_forecast is not None else 'skipped'
                        ),
                        forecast_reason='forecast_timeout',
                        **_forecast_provenance(
                            night_forecast,
                            observation.time.round_no,
                        ),
                    )
                    LOGGER.warning(
                        'NightForecast watchdog expired for team %s round %s',
                        team_id,
                        observation.time.round_no,
                        exc_info=True,
                    )
                except Exception:
                    self._telemetry.set(
                        fallback_used=True,
                        fallback_reason='exception',
                        forecast_mode=(
                            'cached' if night_forecast is not None else 'skipped'
                        ),
                        forecast_reason='exception',
                        **_forecast_provenance(
                            night_forecast,
                            observation.time.round_no,
                        ),
                    )
                    LOGGER.exception(
                        'NightForecast failed for team %s round %s',
                        team_id,
                        observation.time.round_no,
                    )
        elif night_forecast is not None:
            try:
                night_forecast = rebase_day_forecast(
                    observation,
                    night_forecast,
                )
                self._telemetry.set(
                    forecast_update_kind=night_forecast.update_kind.value,
                    forecast_mode=night_forecast.update_kind.value,
                    forecast_reason='day rebase',
                    forecast_cache_hit=True,
                    night_risk_level=night_forecast.risk_level.value,
                    risk_ratio=float(night_forecast.risk_ratio),
                    survival_margin=night_forecast.survival_margin,
                    **_forecast_provenance(
                        night_forecast,
                        observation.time.round_no,
                    ),
                )
            except Exception:
                night_forecast = None
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    'day forecast rebase failed for team %s round %s',
                    team_id,
                    observation.time.round_no,
                )
        if observation.time.phase is Phase.NIGHT and task_controller_exclusions:
            task_survival_interrupt = (
                task_assignment_failed
                or night_forecast is None
                or night_forecast.risk_level
                in {RiskLevel.CRITICAL, RiskLevel.LETHAL}
            )
            self._telemetry.set(
                task_pioneer_reserved=not task_survival_interrupt,
                task_survival_interrupted=task_survival_interrupt,
            )
        desired_controller_exclusions = (
            frozenset()
            if task_survival_interrupt
            else task_controller_exclusions
        )
        director_failed = False
        try:
            with self._telemetry.measure('director_ms'):
                director_decision = self._director.select(
                    observation,
                    previous_state=(
                        previous_for_director.director_state
                        if previous_for_director is not None
                        else None
                    ),
                    previous_observation=(
                        previous_for_director.observation
                        if previous_for_director is not None
                        else None
                    ),
                    certificate=(
                        previous_for_director.certificate
                        if previous_for_director is not None
                        else None
                    ),
                    forecast=night_forecast,
                )
            intent = director_decision.intent
            features = director_decision.features
            director_state = director_decision.state
            # Keep custom/test Director implementations source-compatible while
            # the built-in Director exposes the richer Stage 3 result.
            safety_plan = getattr(director_decision, 'safety_plan', None)
            self._telemetry.set(
                safety_plan_status=(
                    safety_plan.status.value
                    if safety_plan is not None
                    else 'unknown'
                ),
                safety_plan_cost=(
                    safety_plan.gold_cost if safety_plan is not None else 0
                ),
                effective_gold_reserve=intent.gold_reserve,
            )
        except Exception:
            director_failed = True
            self._telemetry.set(fallback_used=True)
            LOGGER.exception(
                'Phase 4 director failed; using default intent for team %s round %s',
                team_id,
                observation.time.round_no,
            )
            intent = DEFAULT_STRATEGIC_INTENT
            director_state = None
            safety_plan = None
            try:
                features = extract_features(
                    observation,
                    forecast=night_forecast,
                )
            except Exception:
                LOGGER.exception('Phase 4 fallback feature extraction failed')
                features = None
        fortification_threats = (
            match_memory.fortification_threat_positions
            if match_memory is not None
            else ()
        )
        previously_built_wall_sites = (
            frozenset(match_memory.seen_friendly_wall_positions)
            if match_memory is not None
            else frozenset()
        )
        build_recovery = (
            match_memory.build_recovery
            if match_memory is not None
            else EMPTY_BUILD_RECOVERY_STATE
        )
        self._telemetry.set(
            fortification_anchor_day=(
                match_memory.fortification_day_no
                if match_memory is not None
                else 0
            ),
            fortification_anchor_threat_count=len(fortification_threats),
            historically_built_wall_count=len(previously_built_wall_sites),
        )
        governor_emergency = request_budget.remaining_compute() <= reserve
        if (
            observation.time.phase is Phase.NIGHT
            and own_station_alive
            and not governor_emergency
            and (
                controller_assignments is None
                or controller_assignment_mode != intent.profile.value
                or controller_assignment_exclusions
                != desired_controller_exclusions
            )
        ):
            try:
                controller_assignments, cache_hit = (
                    self._controller_assignments.resolve(
                        observation,
                        WorldGrid.from_observation(observation),
                        mode_key=intent.profile.value,
                        excluded_role_ids=desired_controller_exclusions,
                    )
                )
                controller_assignment_mode = intent.profile.value
                controller_assignment_exclusions = (
                    desired_controller_exclusions
                )
                self._telemetry.set(controller_cache_hit=cache_hit)
            except Exception:
                if desired_controller_exclusions:
                    task_survival_interrupt = True
                    desired_controller_exclusions = frozenset()
                    controller_assignments = None
                    self._telemetry.set(
                        task_pioneer_reserved=False,
                        task_survival_interrupted=True,
                    )
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    'controller assignment cache failed for team %s round %s',
                    team_id,
                    observation.time.round_no,
                )

        simulation_action = None
        fresh_certificate = None
        certificate = (
            previous_for_director.certificate
            if previous_for_director is not None
            else None
        )

        phase2_5_before = float(
            self._telemetry.current('phase2_5_ms', 0.0)
        )
        if governor_emergency:
            decision = self._plan_emergency(observation, request_budget)
        else:
            decision = self._plan_phase2(
                observation,
                intent,
                controller_assignments,
                fortification_threats,
                (
                    night_forecast.expected_wall_losses
                    if night_forecast is not None
                    else 0
                ),
                market_view,
                (
                    previous_for_director.decision
                    if previous_for_director is not None
                    else None
                ),
                previously_built_wall_sites,
                build_recovery,
            )
            if decision != Decision():
                self._telemetry.set(decision_source='phase2')
        compute_usage.phase2_5_ms = max(0.0, float(
            self._telemetry.current('phase2_5_ms', 0.0)
        ) - phase2_5_before)
        governor_emergency = request_budget.remaining_compute() <= reserve
        if governor_emergency:
            self._telemetry.set(
                compute_governor_action='emergency_reserve',
                phase3_level=Phase3Level.NONE.value,
                fallback_used=True,
                fallback_reason='deadline_low',
                timeout_prevented=True,
            )

        phase3_skip_reason = _phase3_skip_reason(
            observation=observation,
            own_station_status=own_station_status,
            director_failed=director_failed,
            previous=previous,
            continuity=continuity,
            emergency_medicine=_requires_emergency_medicine(observation, intent),
            governor_emergency=governor_emergency,
            prior_watchdog=compute_usage.watchdog_hit,
        )
        phase3_eligible = phase3_skip_reason == 'none'
        self._telemetry.set(phase3_skip_reason=phase3_skip_reason)

        if phase3_eligible:
            trigger = select_phase3_level(
                observation,
                previous_for_director.observation,
                intent,
                forecast=night_forecast,
            )
            requested_level = trigger.level
            if (
                requested_level is Phase3Level.FULL
                and night_forecast is not None
                and night_forecast.update_kind is ForecastUpdateKind.LIGHTWEIGHT
            ):
                requested_level = Phase3Level.LITE
            self._telemetry.set(phase3_level=requested_level.value)
            if requested_level is Phase3Level.NONE:
                self._telemetry.set(phase3_skip_reason='no_material_night_event')
            else:
                if continuity is SessionContinuity.CONSECUTIVE:
                    try:
                        weights = self._scenario_reconciler(
                            previous,
                            observation,
                            config=self._config,
                        )
                    except Exception:
                        self._telemetry.set(fallback_used=True)
                        LOGGER.exception(
                            'scenario reconciliation failed; using prior weights'
                        )
                objective = (
                    self._objective_provider(observation)
                    if self._objective_provider is not None
                    else intent.night_objective
                )
                compute_plan = self._compute_governor.plan(
                    team_id,
                    requested_level=requested_level,
                    phase3_config=self._config,
                    round_no=observation.time.round_no,
                    day_no=observation.time.day_no,
                    now=request_budget.clock(),
                    request_deadline=request_deadline,
                )
                governor_emergency = compute_plan.emergency_return
                governor_state = compute_plan.state
                first_attempt = (
                    compute_plan.attempts[0]
                    if compute_plan.attempts
                    else None
                )
                self._telemetry.set(
                    phase3_effective_level=compute_plan.effective_level.value,
                    compute_governor_action=(
                        ','.join(compute_plan.reasons)
                        if compute_plan.reasons
                        else 'normal'
                    ),
                    governor_root_limit=(
                        first_attempt.budget.root_candidates
                        if first_attempt is not None
                        else 0
                    ),
                    governor_scenario_limit=(
                        first_attempt.budget.scenarios
                        if first_attempt is not None
                        else 0
                    ),
                    phase3_disabled_until_round=(
                        governor_state.phase3_disabled_until_round
                    ),
                    ewma_phase2_5_ms=governor_state.ewma_phase2_5_ms,
                    ewma_root_rollout_ms=governor_state.ewma_root_rollout_ms,
                    phase3_skip_reason=(
                        'none'
                        if compute_plan.attempts
                        else _terminal_compute_reason(compute_plan.reasons)
                    ),
                )
                for planned_attempt in compute_plan.attempts:
                    attempt = planned_attempt.level
                    budget = planned_attempt.budget
                    search_kwargs = {
                        'objective': objective,
                        'config': self._config,
                        'clock': request_budget.clock,
                        'deadline': min(
                            compute_plan.chain_deadline,
                            request_budget.clock() + budget.seconds,
                        ),
                    }
                    if self._search_accepts_budget:
                        search_kwargs['budget'] = budget
                    if self._search_accepts_assignments:
                        search_kwargs['controller_assignments'] = (
                            controller_assignments
                        )
                    if self._search_accepts_level:
                        search_kwargs['level'] = attempt
                    if self._search_accepts_intent:
                        search_kwargs['intent'] = intent
                    if self._search_accepts_baseline:
                        search_kwargs['baseline_decision'] = decision
                    phase3_started_ns = perf_counter_ns()
                    try:
                        request_budget.checkpoint('phase 3 search')
                        with self._telemetry.measure('phase3_ms'):
                            result = self._night_searcher(
                                observation,
                                weights,
                                **search_kwargs,
                            )
                        stats = getattr(result, 'stats', None)
                        if stats is not None:
                            compute_usage.roots_evaluated += stats.roots_evaluated
                            compute_usage.scenarios_per_root = max(
                                compute_usage.scenarios_per_root,
                                stats.scenarios_per_root,
                            )
                            compute_usage.exact_horizon = max(
                                compute_usage.exact_horizon,
                                stats.maximum_steps,
                            )
                            self._telemetry.set(
                                root_candidate_count=stats.roots_generated,
                                scenario_count=stats.scenarios_per_root,
                                simulation_count=(
                                    stats.roots_evaluated
                                    * stats.scenarios_per_root
                                ),
                                candidate_generation_ms=(
                                    stats.candidate_generation_ms
                                ),
                                simulation_ms=stats.simulation_ms,
                                tail_estimator_ms=stats.tail_estimator_ms,
                                phase3_completed_root_count=(
                                    stats.roots_evaluated
                                ),
                            )
                            if getattr(stats, 'deadline_hit', False):
                                self._telemetry.set(watchdog_hit=True)
                                compute_usage.watchdog_hit = True
                        completed_attempt_ms = (
                            perf_counter_ns() - phase3_started_ns
                        ) / 1_000_000
                        compute_usage.completed_phase3_ms += completed_attempt_ms
                        if _eliminates_critical_night_fire(
                            observation,
                            night_forecast,
                            decision,
                            result.decision,
                        ):
                            self._telemetry.increment('phase3_fallback_count')
                            self._telemetry.set(
                                fallback_used=True,
                                fallback_reason='phase3_attack_regression',
                                phase3_executed_level=attempt.value,
                                phase3_skip_reason='phase3_attack_regression',
                            )
                            break
                        if result.decision != decision:
                            self._telemetry.set(decision_source='phase3')
                        decision = result.decision
                        self._telemetry.set(
                            phase3_executed_level=attempt.value,
                            phase3_skip_reason='none',
                        )
                        simulation_action = result.simulation_action
                        certificate = result.certificate
                        fresh_certificate = result.certificate
                        break
                    except (DeadlineExceeded, RequestDeadlineExceeded):
                        compute_usage.watchdog_hit = True
                        self._telemetry.increment('phase3_fallback_count')
                        self._telemetry.set(
                            fallback_used=True,
                            watchdog_hit=True,
                            fallback_reason='phase3_timeout',
                            phase3_skip_reason='phase3_timeout',
                            timeout_prevented=True,
                        )
                        LOGGER.warning(
                            'Phase 3 %s watchdog expired for team %s round %s',
                            attempt.value,
                            team_id,
                            observation.time.round_no,
                            exc_info=True,
                        )
                    except UnsupportedSimulation:
                        self._telemetry.increment('phase3_fallback_count')
                        self._telemetry.set(
                            fallback_used=True,
                            phase3_skip_reason='unsupported_simulation',
                        )
                        LOGGER.warning(
                            'Phase 3 %s unavailable for team %s round %s',
                            attempt.value,
                            team_id,
                            observation.time.round_no,
                            exc_info=True,
                        )
                        break
                    except Exception:
                        self._telemetry.increment('phase3_fallback_count')
                        self._telemetry.set(
                            fallback_used=True,
                            phase3_skip_reason='exception',
                        )
                        LOGGER.exception(
                            'Phase 3 %s failed for team %s round %s',
                            attempt.value,
                            team_id,
                            observation.time.round_no,
                        )
                    finally:
                        compute_usage.phase3_ms += (
                            perf_counter_ns() - phase3_started_ns
                        ) / 1_000_000

        if request_budget.remaining_compute() <= reserve:
            governor_emergency = True
            self._telemetry.set(
                compute_governor_action='emergency_reserve',
                fallback_used=True,
                fallback_reason='deadline_low',
                timeout_prevented=True,
            )

        task_state = (
            previous_for_director.task_state
            if previous_for_director is not None
            else EMPTY_TASK_STATE
        )
        previous_treasure_state = (
            previous.treasure_state
            if previous is not None
            and signature == previous.signature
            and observation.time.round_no > previous.last_round
            else None
        )
        treasure_state = previous_treasure_state
        try:
            task_state = self._task_agent.reconcile(observation, task_state)
        except Exception:
            self._telemetry.set(fallback_used=True)
            LOGGER.exception(
                'Task feedback reconciliation failed for team %s round %s',
                team_id,
                observation.time.round_no,
            )
        try:
            treasure_state = self._treasure_agent.reconcile(
                observation,
                treasure_state,
            )
        except Exception:
            self._telemetry.set(fallback_used=True)
            LOGGER.exception(
                'Treasure feedback reconciliation failed for team %s round %s',
                team_id,
                observation.time.round_no,
            )
            if treasure_state is None:
                treasure_state = EMPTY_TREASURE_STATE
        if not governor_emergency:
            try:
                with self._telemetry.measure('task_ms'):
                    task_kwargs = {'previous_state': task_state}
                    if self._task_accepts_survival_interrupt:
                        task_kwargs['force_survival_interrupt'] = (
                            task_survival_interrupt
                        )
                    task_result = self._task_agent.apply(
                        observation,
                        decision,
                        intent,
                        **task_kwargs,
                    )
                task_input_decision = decision
                decision = task_result.decision
                task_state = task_result.state
                if decision != task_input_decision:
                    self._telemetry.set(decision_source='task')
                pending_candidate = next(
                    (
                        candidate
                        for candidate in task_state.candidates
                        if candidate.answer == task_state.pending_answer
                    ),
                    None,
                )
                self._telemetry.set(
                    task_candidate_count=len(task_state.candidates),
                    task_sop_hit=(
                        pending_candidate is not None
                        and pending_candidate.source == 'sop'
                    ),
                )
            except Exception:
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    'Phase 5 task agent failed; using base decision for team %s round %s',
                    team_id,
                    observation.time.round_no,
                )
            governor_emergency = request_budget.remaining_compute() <= reserve
            if governor_emergency:
                self._telemetry.set(
                    compute_governor_action='emergency_reserve',
                    fallback_used=True,
                    fallback_reason='deadline_low',
                    timeout_prevented=True,
                )
            else:
                try:
                    treasure_input_decision = decision
                    with self._telemetry.measure('treasure_ms'):
                        treasure_result = self._treasure_agent.apply(
                            observation,
                            decision,
                            intent,
                            previous_state=treasure_state,
                            task_active=bool(
                                task_state.active_task_type
                                or observation.phase_task.strip()
                            ),
                        )
                    decision = treasure_result.decision
                    treasure_state = treasure_result.state
                    self._telemetry.set(
                        treasure_candidate_count=len(treasure_state.candidates),
                        treasure_attempted=any(
                            action.kind is ActionKind.SUMMON_TREASURE
                            for action in decision.commands.values()
                        ),
                        treasure_result=observation.last_summon_treasure_result,
                    )
                    if decision != treasure_input_decision:
                        self._telemetry.set(decision_source='treasure')
                except Exception:
                    self._telemetry.set(fallback_used=True)
                    LOGGER.exception(
                        'Phase 5 treasure agent failed; using task decision for team %s round %s',
                        team_id,
                        observation.time.round_no,
                    )

        if (
            fresh_certificate is not None
            or night_forecast is not None
            or not own_station_alive
        ):
            try:
                self._memory.observe(
                    observation,
                    certificate=fresh_certificate,
                    forecast=night_forecast,
                    clear_forecast=not own_station_alive,
                )
            except Exception:
                LOGGER.exception(
                    'Phase 6 certificate memory update failed for team %s round %s',
                    team_id,
                    observation.time.round_no,
                )

        self._sessions.put(
            StrategySession(
                team_id=team_id,
                last_round=observation.time.round_no,
                fingerprint=fingerprint,
                signature=signature,
                observation=observation,
                decision=decision,
                simulation_action=simulation_action,
                certificate=certificate,
                scenario_weights=weights,
                features=features,
                director_state=director_state,
                intent=intent,
                task_state=task_state,
                treasure_state=treasure_state,
                night_forecast=night_forecast,
                market_state=market_state,
            )
        )
        return decision

    def _plan_phase2(
        self,
        observation: Observation,
        intent: StrategicIntent,
        controller_assignments: tuple[ControllerAssignment, ...] | None = None,
        fortification_threats: tuple[Position, ...] = (),
        expected_wall_losses: int = 0,
        market_view: MarketView | None = None,
        previous_decision: Decision | None = None,
        previously_built_wall_sites: frozenset[Position] = frozenset(),
        build_recovery: BuildRecoveryState = EMPTY_BUILD_RECOVERY_STATE,
    ) -> Decision:
        kwargs: dict[str, object] = {}
        if self._phase2_accepts_intent:
            kwargs['intent'] = intent
        if self._phase2_accepts_assignments:
            kwargs['controller_assignments'] = controller_assignments
        if self._phase2_accepts_telemetry:
            kwargs['telemetry'] = self._telemetry
        if self._phase2_accepts_fortification_threats:
            kwargs['fortification_threats'] = fortification_threats
        if self._phase2_accepts_expected_wall_losses:
            kwargs['expected_wall_losses'] = expected_wall_losses
        if self._phase2_accepts_market_view:
            kwargs['market_view'] = market_view
        if self._phase2_accepts_previous_decision:
            kwargs['previous_decision'] = previous_decision
        if self._phase2_accepts_previously_built_walls:
            kwargs['previously_built_wall_sites'] = (
                previously_built_wall_sites
            )
        if self._phase2_accepts_build_recovery:
            kwargs['build_recovery'] = build_recovery
        try:
            with self._telemetry.measure('phase2_ms'):
                return self._phase2_planner(observation, **kwargs)
        except Exception:
            self._telemetry.set(fallback_used=True)
            LOGGER.exception(
                'Phase 2 failed; using empty legal decision for team %s round %s',
                observation.our.team_id,
                observation.time.round_no,
            )
            return Decision()


def _eliminates_critical_night_fire(
    observation: Observation,
    forecast: NightForecast | None,
    baseline: Decision,
    candidate: Decision,
) -> bool:
    if (
        observation.time.phase is not Phase.NIGHT
        or forecast is None
        or forecast.risk_level not in {RiskLevel.CRITICAL, RiskLevel.LETHAL}
    ):
        return False
    baseline_attacks = sum(
        action.kind is ActionKind.ATTACK
        for action in baseline.commands.values()
    )
    candidate_attacks = sum(
        action.kind is ActionKind.ATTACK
        for action in candidate.commands.values()
    )
    return baseline_attacks > 0 and candidate_attacks == 0


def _own_station_status(
    observation: Observation,
    previous: StrategySession | None,
) -> str:
    stations = tuple(
        unit for unit in observation.our.units if unit.role_type == 'station'
    )
    living = tuple(unit for unit in stations if unit.health > 0)
    if len(living) == 1:
        return 'alive'
    if len(living) > 1:
        return 'state_invalid'
    if any(unit.health <= 0 for unit in stations):
        return 'destroyed'
    if previous is not None and observation.time.round_no > previous.last_round:
        prior_stations = tuple(
            unit
            for unit in previous.observation.our.units
            if unit.role_type == 'station'
        )
        if any(unit.health > 0 for unit in prior_stations) or any(
            unit.health <= 0 for unit in prior_stations
        ):
            return 'destroyed'
    return 'state_invalid'


def _phase3_skip_reason(
    *,
    observation: Observation,
    own_station_status: str,
    director_failed: bool,
    previous: StrategySession | None,
    continuity: SessionContinuity,
    emergency_medicine: bool,
    governor_emergency: bool,
    prior_watchdog: bool,
) -> str:
    if own_station_status == 'destroyed':
        return 'own_station_destroyed'
    if own_station_status != 'alive':
        return 'own_station_state_invalid'
    if director_failed:
        return 'director_failed'
    if observation.time.phase is not Phase.NIGHT:
        return 'not_night'
    if previous is None:
        return 'no_previous_observation'
    if continuity is not SessionContinuity.CONSECUTIVE:
        return 'non_consecutive_observation'
    if emergency_medicine:
        return 'emergency_medicine'
    if governor_emergency:
        return 'emergency_reserve'
    if prior_watchdog:
        return 'prior_watchdog'
    return 'none'


def _terminal_compute_reason(reasons: tuple[str, ...]) -> str:
    for reason in reversed(reasons):
        if reason in {
            'compute_exhausted',
            'emergency_reserve',
            'watchdog_cooldown',
            'not_requested',
        }:
            return reason
    return reasons[-1] if reasons else 'compute_exhausted'


def _forecast_provenance(
    forecast: NightForecast | None,
    current_round: int,
) -> dict[str, object]:
    if forecast is None:
        return {
            'forecast_generated_round': 0,
            'forecast_updated_round': 0,
            'forecast_age_rounds': 0,
            'forecast_margin_source': 'none',
            'forecast_conservative_bound': False,
        }
    conservative = (
        'incremental_cache_lightweight_correction'
        in forecast.uncertainty_reasons
    )
    return {
        'forecast_generated_round': forecast.generated_round,
        'forecast_updated_round': forecast.updated_round,
        'forecast_age_rounds': max(0, current_round - forecast.updated_round),
        'forecast_margin_source': (
            'conservative_bound'
            if conservative
            else forecast.update_kind.value
        ),
        'forecast_conservative_bound': conservative,
    }


def _accepts_intent(planner: Phase2Planner) -> bool:
    return _accepts_keyword(planner, 'intent')


def _accepts_keyword(function: Callable[..., object], name: str) -> bool:
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _active_task_controller_exclusions(
    observation: Observation,
) -> frozenset[int]:
    if (
        observation.time.phase is not Phase.NIGHT
        or not observation.phase_task.strip()
    ):
        return frozenset()
    return frozenset(
        unit.unit_id
        for unit in observation.our.units
        if unit.health > 0 and unit.role_type.strip().lower() == 'pioneer'
    )


def _requires_emergency_medicine(
    observation: Observation,
    intent: StrategicIntent,
) -> bool:
    policy = intent.item_policy
    if policy.medicine_health_threshold <= 0:
        return False
    return any(
        role.health > 0
        and role.role_type in {'worker', 'pioneer'}
        and role.health <= policy.medicine_health_threshold
        and has_item(role.backpack, policy.medicine_name)
        for role in observation.our.units
    )
