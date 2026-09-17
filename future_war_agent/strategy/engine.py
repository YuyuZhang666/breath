import logging
from collections.abc import Callable
from threading import RLock
from time import monotonic

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase

from .planner import plan_turn
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
from .simulation.objective import DEFAULT_NIGHT_OBJECTIVE, NightObjective
from .simulation.search import (
    ScenarioWeights,
    SearchResult,
    search_night,
)


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Phase2Planner = Callable[[Observation], Decision]
NightSearcher = Callable[..., SearchResult]
ObjectiveProvider = Callable[[Observation], NightObjective]
ScenarioReconciler = Callable[..., ScenarioWeights]


def _default_objective(_: Observation) -> NightObjective:
    return DEFAULT_NIGHT_OBJECTIVE


class StrategyEngine:
    def __init__(
        self,
        *,
        config: Phase3Config = DEFAULT_PHASE3_CONFIG,
        phase2_planner: Phase2Planner = plan_turn,
        night_searcher: NightSearcher = search_night,
        objective_provider: ObjectiveProvider = _default_objective,
        clock: Callable[[], float] = monotonic,
        session_store: SessionStore | None = None,
        scenario_reconciler: ScenarioReconciler = reconcile_scenario_weights,
    ) -> None:
        self._config = config
        self._phase2_planner = phase2_planner
        self._night_searcher = night_searcher
        self._objective_provider = objective_provider
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
            return self._phase2_planner(observation)

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
        simulation_action = None
        certificate = None

        phase3_eligible = (
            observation.time.phase is Phase.NIGHT
            and previous is not None
            and (
                continuity is SessionContinuity.CONSECUTIVE
                or (
                    continuity is SessionContinuity.REVISION
                    and previous.simulation_action is not None
                )
            )
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
                objective = self._objective_provider(observation)
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
                decision = self._phase2_planner(observation)
            except Exception:
                LOGGER.exception(
                    "Phase 3 failed; using Phase 2 for team %s round %s",
                    team_id,
                    observation.time.round_no,
                )
                decision = self._phase2_planner(observation)
        else:
            decision = self._phase2_planner(observation)

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
            )
        )
        return decision
