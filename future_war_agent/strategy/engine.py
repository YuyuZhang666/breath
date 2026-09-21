import logging
from collections.abc import Callable
from inspect import Parameter, signature
from threading import RLock
from time import monotonic, perf_counter_ns

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.time import Phase
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .compute import ComputeGovernor, ComputeTurnUsage
from .director import StrategicDirector
from .features import extract_features
from .forecast import (
    ForecastRefresh,
    NightForecast,
    forecast_recompute_reason,
    rebase_day_forecast,
    refresh_night_forecast,
)
from .items import has_item
from .memory import MatchMemoryStore
from .night import ControllerAssignment, ControllerAssignmentCache
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
from .world import WorldGrid


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Phase2Planner = Callable[..., Decision]
NightSearcher = Callable[..., SearchResult]
ObjectiveProvider = Callable[[Observation], NightObjective]
ScenarioReconciler = Callable[..., ScenarioWeights]
NightForecaster = Callable[..., ForecastRefresh]


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
        memory_store: MatchMemoryStore | None = None,
        clock: Callable[[], float] = monotonic,
        session_store: SessionStore | None = None,
        scenario_reconciler: ScenarioReconciler = reconcile_scenario_weights,
        controller_assignment_cache: ControllerAssignmentCache | None = None,
        night_forecaster: NightForecaster = refresh_night_forecast,
        telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
        compute_governor: ComputeGovernor | None = None,
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
        self._telemetry = telemetry
        self._compute_governor = (
            compute_governor
            if compute_governor is not None
            else ComputeGovernor()
        )
        self._lock = RLock()

    def plan(
        self,
        observation: Observation,
        *,
        request_started_at: float | None = None,
    ) -> Decision:
        if request_started_at is None:
            request_started_at = self._clock()
        compute_usage = ComputeTurnUsage()
        token = self._telemetry.begin()
        self._telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
        try:
            with self._lock:
                try:
                    return self._plan_locked(
                        observation,
                        request_started_at,
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
                        )
                    except Exception:
                        LOGGER.exception(
                            'ComputeGovernor update failed for team %s round %s',
                            observation.our.team_id,
                            observation.time.round_no,
                        )
        finally:
            self._telemetry.finish(token)

    def current_profile(self, team_id: str) -> StrategyProfile | None:
        with self._lock:
            session = self._sessions.get(team_id)
            return session.intent.profile if session is not None else None

    def _plan_locked(
        self,
        observation: Observation,
        request_started_at: float,
        compute_usage: ComputeTurnUsage,
    ) -> Decision:
        team_id = observation.our.team_id
        if not team_id.strip():
            return self._plan_phase2(observation, DEFAULT_STRATEGIC_INTENT)

        fingerprint = observation_fingerprint(observation)
        signature = static_signature(observation)
        previous = self._sessions.get(team_id)
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
            match_memory = self._memory.observe(observation)
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
        controller_assignments: tuple[ControllerAssignment, ...] | None = None
        controller_assignment_mode: str | None = None
        night_forecast: NightForecast | None = (
            previous_for_director.night_forecast
            if previous_for_director is not None
            else None
        )
        if observation.time.phase is Phase.NIGHT:
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
            invalidation_reason = forecast_recompute_reason(
                observation,
                previous_observation=previous_observation,
                previous_forecast=night_forecast,
                previous_decision=previous_decision,
            )
            if invalidation_reason is not None:
                try:
                    controller_assignments, cache_hit = (
                        self._controller_assignments.resolve(
                            observation,
                            WorldGrid.from_observation(observation),
                            mode_key=StrategyProfile.SURVIVE.value,
                        )
                    )
                    controller_assignment_mode = StrategyProfile.SURVIVE.value
                    self._telemetry.set(controller_cache_hit=cache_hit)
                except Exception:
                    self._telemetry.set(fallback_used=True)
                    LOGGER.exception(
                        'forecast controller assignment failed for team %s round %s',
                        team_id,
                        observation.time.round_no,
                    )
            try:
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
                with self._telemetry.measure('forecast_ms'):
                    forecast_refresh = self._night_forecaster(
                        observation,
                        **forecast_kwargs,
                    )
                night_forecast = forecast_refresh.forecast
                self._telemetry.set(
                    forecast_update_kind=night_forecast.update_kind.value,
                    forecast_cache_hit=not forecast_refresh.recomputed,
                    night_risk_level=night_forecast.risk_level.value,
                    risk_ratio=float(night_forecast.risk_ratio),
                    survival_margin=night_forecast.survival_margin,
                )
            except Exception:
                self._telemetry.set(fallback_used=True)
                if invalidation_reason is not None:
                    night_forecast = None
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
                    forecast_cache_hit=True,
                    night_risk_level=night_forecast.risk_level.value,
                    risk_ratio=float(night_forecast.risk_ratio),
                    survival_margin=night_forecast.survival_margin,
                )
            except Exception:
                night_forecast = None
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    'day forecast rebase failed for team %s round %s',
                    team_id,
                    observation.time.round_no,
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
            match_memory.recent_threat_positions
            if match_memory is not None
            else ()
        )
        if (
            observation.time.phase is Phase.NIGHT
            and (
                controller_assignments is None
                or controller_assignment_mode != intent.profile.value
            )
        ):
            try:
                controller_assignments, cache_hit = (
                    self._controller_assignments.resolve(
                        observation,
                        WorldGrid.from_observation(observation),
                        mode_key=intent.profile.value,
                    )
                )
                controller_assignment_mode = intent.profile.value
                self._telemetry.set(controller_cache_hit=cache_hit)
            except Exception:
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
        decision = self._plan_phase2(
            observation,
            intent,
            controller_assignments,
            fortification_threats,
        )
        compute_usage.phase2_5_ms = max(0.0, float(
            self._telemetry.current('phase2_5_ms', 0.0)
        ) - phase2_5_before)
        request_deadline = (
            request_started_at
            + self._compute_governor.config.total_decision_budget_seconds
        )
        governor_emergency = (
            self._clock()
            >= request_deadline
            - self._compute_governor.config.emergency_reserve_seconds
        )
        if governor_emergency:
            self._telemetry.set(
                compute_governor_action='emergency_reserve',
                phase3_level=Phase3Level.NONE.value,
            )

        phase3_eligible = (
            not director_failed
            and observation.time.phase is Phase.NIGHT
            and previous is not None
            and (
                continuity is SessionContinuity.CONSECUTIVE
                or (
                    continuity is SessionContinuity.REVISION
                    and previous.simulation_action is not None
                )
            )
            and not _requires_emergency_medicine(observation, intent)
            and not governor_emergency
        )

        if phase3_eligible:
            trigger = select_phase3_level(
                observation,
                previous_for_director.observation,
                intent,
                forecast=night_forecast,
            )
            self._telemetry.set(phase3_level=trigger.level.value)
            if trigger.level is not Phase3Level.NONE:
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
                    requested_level=trigger.level,
                    phase3_config=self._config,
                    round_no=observation.time.round_no,
                    day_no=observation.time.day_no,
                    now=self._clock(),
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
                )
                for planned_attempt in compute_plan.attempts:
                    attempt = planned_attempt.level
                    budget = planned_attempt.budget
                    search_kwargs = {
                        'objective': objective,
                        'config': self._config,
                        'clock': self._clock,
                        'deadline': min(
                            compute_plan.chain_deadline,
                            self._clock() + budget.seconds,
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
                    try:
                        phase3_started_ns = perf_counter_ns()
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
                            )
                            if getattr(stats, 'deadline_hit', False):
                                self._telemetry.set(watchdog_hit=True)
                                compute_usage.watchdog_hit = True
                        decision = result.decision
                        simulation_action = result.simulation_action
                        certificate = result.certificate
                        fresh_certificate = result.certificate
                        break
                    except DeadlineExceeded:
                        compute_usage.watchdog_hit = True
                        self._telemetry.increment('phase3_fallback_count')
                        self._telemetry.set(
                            fallback_used=True,
                            watchdog_hit=True,
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
                        self._telemetry.set(fallback_used=True)
                        LOGGER.warning(
                            'Phase 3 %s unavailable for team %s round %s',
                            attempt.value,
                            team_id,
                            observation.time.round_no,
                            exc_info=True,
                        )
                    except Exception:
                        self._telemetry.increment('phase3_fallback_count')
                        self._telemetry.set(fallback_used=True)
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

        if (
            self._clock()
            >= request_deadline
            - self._compute_governor.config.emergency_reserve_seconds
        ):
            governor_emergency = True
            self._telemetry.set(compute_governor_action='emergency_reserve')

        task_state = (
            previous_for_director.task_state
            if previous_for_director is not None
            else EMPTY_TASK_STATE
        )
        if not governor_emergency:
            try:
                with self._telemetry.measure('task_ms'):
                    task_result = self._task_agent.apply(
                        observation,
                        decision,
                        intent,
                        previous_state=task_state,
                    )
                decision = task_result.decision
                task_state = task_result.state
            except Exception:
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    'Phase 5 task agent failed; using base decision for team %s round %s',
                    team_id,
                    observation.time.round_no,
                )
                task_state = EMPTY_TASK_STATE

        if fresh_certificate is not None or night_forecast is not None:
            try:
                self._memory.observe(
                    observation,
                    certificate=fresh_certificate,
                    forecast=night_forecast,
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
                night_forecast=night_forecast,
            )
        )
        return decision

    def _plan_phase2(
        self,
        observation: Observation,
        intent: StrategicIntent,
        controller_assignments: tuple[ControllerAssignment, ...] | None = None,
        fortification_threats: tuple[Position, ...] = (),
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
