import logging
from collections.abc import Callable
from inspect import Parameter, signature
from threading import RLock
from time import monotonic

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .director import StrategicDirector
from .features import extract_features
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
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .simulation.errors import DeadlineExceeded, UnsupportedSimulation
from .simulation.objective import NightObjective
from .simulation.search import (
    ScenarioWeights,
    SearchResult,
    search_night,
)
from .task_agent import EMPTY_TASK_STATE, TaskAgent
from .world import WorldGrid


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Phase2Planner = Callable[..., Decision]
NightSearcher = Callable[..., SearchResult]
ObjectiveProvider = Callable[[Observation], NightObjective]
ScenarioReconciler = Callable[..., ScenarioWeights]


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
        telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
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
        self._night_searcher = night_searcher
        self._search_accepts_assignments = _accepts_keyword(
            night_searcher,
            'controller_assignments',
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
        self._telemetry = telemetry
        self._lock = RLock()

    def plan(self, observation: Observation) -> Decision:
        token = self._telemetry.begin()
        self._telemetry.identify(
            team_id=observation.our.team_id,
            round_no=observation.time.round_no,
            phase=observation.time.phase.value,
        )
        try:
            with self._lock:
                return self._plan_locked(observation)
        finally:
            self._telemetry.finish(token)

    def current_profile(self, team_id: str) -> StrategyProfile | None:
        with self._lock:
            session = self._sessions.get(team_id)
            return session.intent.profile if session is not None else None

    def _plan_locked(self, observation: Observation) -> Decision:
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
        if continuity is SessionContinuity.DUPLICATE:
            if previous is None:
                raise AssertionError("duplicate continuity requires a session")
            self._telemetry.set(duplicate_request=True)
            return previous.decision

        try:
            self._memory.observe(observation)
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
                )
            intent = director_decision.intent
            features = director_decision.features
            director_state = director_decision.state
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
            try:
                features = extract_features(observation)
            except Exception:
                LOGGER.exception('Phase 4 fallback feature extraction failed')
                features = None
        controller_assignments: tuple[ControllerAssignment, ...] | None = None
        if observation.time.phase is Phase.NIGHT:
            try:
                controller_assignments, cache_hit = (
                    self._controller_assignments.resolve(
                        observation,
                        WorldGrid.from_observation(observation),
                        mode_key=intent.profile.value,
                    )
                )
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
        )

        if phase3_eligible:
            self._telemetry.set(phase3_level='legacy_full')
            try:
                if continuity is SessionContinuity.CONSECUTIVE:
                    weights = self._scenario_reconciler(
                        previous,
                        observation,
                        config=self._config,
                    )
                deadline = self._clock() + self._config.watchdog_seconds
                objective = (
                    self._objective_provider(observation)
                    if self._objective_provider is not None
                    else intent.night_objective
                )
                search_kwargs = {
                    'objective': objective,
                    'config': self._config,
                    'clock': self._clock,
                    'deadline': deadline,
                }
                if self._search_accepts_assignments:
                    search_kwargs['controller_assignments'] = (
                        controller_assignments
                    )
                with self._telemetry.measure('phase3_ms'):
                    result = self._night_searcher(
                        observation,
                        weights,
                        **search_kwargs,
                    )
                stats = getattr(result, 'stats', None)
                if stats is not None:
                    self._telemetry.set(
                        root_candidate_count=stats.roots_generated,
                        scenario_count=stats.scenarios_per_root,
                        simulation_count=(
                            stats.roots_evaluated * stats.scenarios_per_root
                        ),
                        candidate_generation_ms=stats.candidate_generation_ms,
                        simulation_ms=stats.simulation_ms,
                    )
                decision = result.decision
                simulation_action = result.simulation_action
                certificate = result.certificate
                fresh_certificate = result.certificate
            except DeadlineExceeded:
                self._telemetry.increment('phase3_fallback_count')
                self._telemetry.set(fallback_used=True, watchdog_hit=True)
                LOGGER.warning(
                    "Phase 3 watchdog expired; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                    exc_info=True,
                )
                decision = self._plan_phase2(
                    observation,
                    intent,
                    controller_assignments,
                )
            except UnsupportedSimulation:
                self._telemetry.increment('phase3_fallback_count')
                self._telemetry.set(fallback_used=True)
                LOGGER.warning(
                    "Phase 3 unavailable; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                    exc_info=True,
                )
                decision = self._plan_phase2(
                    observation,
                    intent,
                    controller_assignments,
                )
            except Exception:
                self._telemetry.increment('phase3_fallback_count')
                self._telemetry.set(fallback_used=True)
                LOGGER.exception(
                    "Phase 3 failed; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                )
                decision = self._plan_phase2(
                    observation,
                    intent,
                    controller_assignments,
                )
        else:
            decision = self._plan_phase2(
                observation,
                intent,
                controller_assignments,
            )

        task_state = (
            previous_for_director.task_state
            if previous_for_director is not None
            else EMPTY_TASK_STATE
        )
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

        if fresh_certificate is not None:
            try:
                self._memory.observe(
                    observation,
                    certificate=fresh_certificate,
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
            )
        )
        return decision

    def _plan_phase2(
        self,
        observation: Observation,
        intent: StrategicIntent,
        controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    ) -> Decision:
        kwargs: dict[str, object] = {}
        if self._phase2_accepts_intent:
            kwargs['intent'] = intent
        if self._phase2_accepts_assignments:
            kwargs['controller_assignments'] = controller_assignments
        if self._phase2_accepts_telemetry:
            kwargs['telemetry'] = self._telemetry
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
