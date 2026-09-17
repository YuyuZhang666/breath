import logging
from collections.abc import Callable
from inspect import Parameter, signature
from threading import RLock
from time import monotonic

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase

from .director import StrategicDirector
from .features import extract_features
from .planner import plan_turn
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
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
        clock: Callable[[], float] = monotonic,
        session_store: SessionStore | None = None,
        scenario_reconciler: ScenarioReconciler = reconcile_scenario_weights,
    ) -> None:
        self._config = config
        self._phase2_planner = phase2_planner
        self._phase2_accepts_intent = _accepts_intent(phase2_planner)
        self._night_searcher = night_searcher
        self._objective_provider = objective_provider
        self._director = director if director is not None else StrategicDirector()
        self._clock = clock
        self._sessions = session_store if session_store is not None else SessionStore()
        self._scenario_reconciler = scenario_reconciler
        self._lock = RLock()

    def plan(self, observation: Observation) -> Decision:
        with self._lock:
            return self._plan_locked(observation)

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
            return previous.decision

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
        simulation_action = None
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
                result = self._night_searcher(
                    observation,
                    weights,
                    objective=objective,
                    config=self._config,
                    clock=self._clock,
                    deadline=deadline,
                )
                decision = result.decision
                simulation_action = result.simulation_action
                certificate = result.certificate
            except (UnsupportedSimulation, DeadlineExceeded):
                LOGGER.warning(
                    "Phase 3 unavailable; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                    exc_info=True,
                )
                decision = self._plan_phase2(observation, intent)
            except Exception:
                LOGGER.exception(
                    "Phase 3 failed; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                )
                decision = self._plan_phase2(observation, intent)
        else:
            decision = self._plan_phase2(observation, intent)

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
            )
        )
        return decision

    def _plan_phase2(
        self,
        observation: Observation,
        intent: StrategicIntent,
    ) -> Decision:
        if self._phase2_accepts_intent:
            return self._phase2_planner(observation, intent=intent)
        return self._phase2_planner(observation)


def _accepts_intent(planner: Phase2Planner) -> bool:
    try:
        parameters = signature(planner).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == 'intent'
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
        and policy.medicine_name in role.backpack
        for role in observation.our.units
    )
