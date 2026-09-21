from dataclasses import dataclass, replace
from threading import RLock

from .simulation.config import Phase3Budget, Phase3Config, Phase3Level


@dataclass(frozen=True, slots=True)
class ComputeGovernorConfig:
    ewma_alpha: float = 0.25
    slow_phase3_ms: float = 250.0
    very_slow_phase3_ms: float = 300.0
    consecutive_very_slow_limit: int = 2
    cooldown_rounds_after_watchdog: int = 10
    total_decision_budget_seconds: float = 3.0
    phase3_hard_stop_seconds: float = 0.350
    forecast_watchdog_seconds: float = 0.100
    forecast_full_watchdog_seconds: float = 0.750
    night_full_forecast_enabled: bool = False
    emergency_reserve_seconds: float = 0.500
    budget_utilization: float = 0.80

    def __post_init__(self) -> None:
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError('ewma_alpha must be in (0, 1]')
        if self.slow_phase3_ms <= 0:
            raise ValueError('slow_phase3_ms must be positive')
        if self.very_slow_phase3_ms < self.slow_phase3_ms:
            raise ValueError('very_slow_phase3_ms cannot be below slow_phase3_ms')
        if self.consecutive_very_slow_limit <= 0:
            raise ValueError('consecutive_very_slow_limit must be positive')
        if self.cooldown_rounds_after_watchdog <= 0:
            raise ValueError('cooldown_rounds_after_watchdog must be positive')
        if self.total_decision_budget_seconds <= 0:
            raise ValueError('total_decision_budget_seconds must be positive')
        if self.phase3_hard_stop_seconds <= 0:
            raise ValueError('phase3_hard_stop_seconds must be positive')
        if self.forecast_watchdog_seconds <= 0:
            raise ValueError('forecast_watchdog_seconds must be positive')
        if self.forecast_full_watchdog_seconds <= 0:
            raise ValueError('forecast_full_watchdog_seconds must be positive')
        if not 0 < self.emergency_reserve_seconds < self.total_decision_budget_seconds:
            raise ValueError('emergency reserve must fit inside the decision budget')
        if not 0 < self.budget_utilization <= 1:
            raise ValueError('budget_utilization must be in (0, 1]')


@dataclass(frozen=True, slots=True)
class ComputeGovernorState:
    ewma_phase2_5_ms: float = 0.0
    ewma_root_rollout_ms: float = 0.0
    watchdog_hits: int = 0
    phase3_disabled_until_round: int = 0
    consecutive_very_slow: int = 0
    full_disabled_day: int | None = None
    halve_root_round: int = 0
    last_phase3_ms: float = 0.0
    last_scenarios_per_root: int = 1
    last_exact_horizon: int = 1
    last_observed_round: int = 0
    last_watchdog_round: int = 0


@dataclass(frozen=True, slots=True)
class ComputeAttempt:
    level: Phase3Level
    budget: Phase3Budget


@dataclass(slots=True)
class ComputeTurnUsage:
    phase2_5_ms: float = 0.0
    phase3_ms: float = 0.0
    roots_evaluated: int = 0
    scenarios_per_root: int = 0
    exact_horizon: int = 0
    watchdog_hit: bool = False


@dataclass(frozen=True, slots=True)
class ComputePlan:
    requested_level: Phase3Level
    effective_level: Phase3Level
    attempts: tuple[ComputeAttempt, ...]
    chain_deadline: float
    reasons: tuple[str, ...]
    state: ComputeGovernorState

    @property
    def emergency_return(self) -> bool:
        return 'emergency_reserve' in self.reasons


DEFAULT_COMPUTE_GOVERNOR_CONFIG = ComputeGovernorConfig()


class ComputeGovernor:
    def __init__(
        self,
        config: ComputeGovernorConfig = DEFAULT_COMPUTE_GOVERNOR_CONFIG,
    ) -> None:
        self._config = config
        self._states: dict[str, ComputeGovernorState] = {}
        self._lock = RLock()

    @property
    def config(self) -> ComputeGovernorConfig:
        return self._config

    def state_for(self, team_id: str) -> ComputeGovernorState:
        with self._lock:
            return self._states.get(team_id, ComputeGovernorState())

    def reset(self, team_id: str) -> None:
        with self._lock:
            self._states.pop(team_id, None)

    def plan(
        self,
        team_id: str,
        *,
        requested_level: Phase3Level,
        phase3_config: Phase3Config,
        round_no: int,
        day_no: int,
        now: float,
        request_deadline: float,
    ) -> ComputePlan:
        with self._lock:
            state = self._states.get(team_id, ComputeGovernorState())
            reasons: list[str] = []
            remaining = request_deadline - now
            reserve = self._config.emergency_reserve_seconds
            chain_seconds = min(
                self._config.phase3_hard_stop_seconds,
                max(0.0, remaining - reserve),
            )
            chain_deadline = now + chain_seconds

            if requested_level is Phase3Level.NONE:
                return ComputePlan(
                    requested_level,
                    Phase3Level.NONE,
                    (),
                    chain_deadline,
                    ('not_requested',),
                    state,
                )

            if round_no <= state.phase3_disabled_until_round:
                reasons.append('watchdog_cooldown')
                return ComputePlan(
                    requested_level,
                    Phase3Level.NONE,
                    (),
                    chain_deadline,
                    tuple(reasons),
                    state,
                )
            if remaining <= reserve:
                reasons.append('emergency_reserve')
                return ComputePlan(
                    requested_level,
                    Phase3Level.NONE,
                    (),
                    chain_deadline,
                    tuple(reasons),
                    state,
                )

            effective_level = requested_level
            if (
                requested_level is Phase3Level.FULL
                and state.full_disabled_day == day_no
            ):
                effective_level = Phase3Level.LITE
                reasons.append('full_disabled_for_night')

            levels = (
                (Phase3Level.FULL, Phase3Level.LITE)
                if effective_level is Phase3Level.FULL
                else (Phase3Level.LITE,)
            )
            attempts_list: list[ComputeAttempt] = []
            halve_roots = state.halve_root_round == round_no
            for level in levels:
                adaptive = self._adaptive_budget(
                    phase3_config.budget_for(level),
                    state,
                    chain_seconds=chain_seconds,
                    halve_roots=halve_roots,
                    reasons=reasons,
                )
                if adaptive is None:
                    if level is Phase3Level.FULL:
                        effective_level = Phase3Level.LITE
                        reasons.append('full_downgraded_to_lite')
                    continue
                attempts_list.append(ComputeAttempt(level, adaptive))
            attempts = tuple(attempts_list)
            if not attempts:
                effective_level = Phase3Level.NONE
                reasons.append('compute_exhausted')
            if state.halve_root_round <= round_no:
                state = replace(state, halve_root_round=0)
                self._states[team_id] = state
            return ComputePlan(
                requested_level,
                effective_level,
                attempts,
                chain_deadline,
                tuple(dict.fromkeys(reasons)),
                state,
            )

    def observe_turn(
        self,
        team_id: str,
        *,
        round_no: int,
        day_no: int,
        phase2_5_ms: float,
        phase3_ms: float,
        roots_evaluated: int,
        scenarios_per_root: int,
        exact_horizon: int,
        watchdog_hit: bool,
    ) -> ComputeGovernorState:
        if not team_id.strip():
            return ComputeGovernorState()
        with self._lock:
            state = self._states.get(team_id, ComputeGovernorState())
            phase2_ewma = state.ewma_phase2_5_ms
            if phase2_5_ms > 0:
                phase2_ewma = self._ewma(phase2_ewma, phase2_5_ms)

            root_ewma = state.ewma_root_rollout_ms
            consecutive = state.consecutive_very_slow
            full_disabled_day = state.full_disabled_day
            halve_round = state.halve_root_round
            last_phase3_ms = state.last_phase3_ms
            last_scenarios = state.last_scenarios_per_root
            last_horizon = state.last_exact_horizon
            if phase3_ms > 0:
                root_sample = phase3_ms / max(1, roots_evaluated)
                root_ewma = self._ewma(root_ewma, root_sample)
                last_phase3_ms = phase3_ms
                last_scenarios = max(1, scenarios_per_root)
                last_horizon = max(1, exact_horizon)
                halve_round = (
                    round_no + 1
                    if phase3_ms > self._config.slow_phase3_ms
                    else 0
                )
                if round_no != state.last_observed_round:
                    consecutive = (
                        consecutive + 1
                        if phase3_ms > self._config.very_slow_phase3_ms
                        else 0
                    )
                if consecutive >= self._config.consecutive_very_slow_limit:
                    full_disabled_day = day_no

            disabled_until = state.phase3_disabled_until_round
            watchdog_hits = state.watchdog_hits
            last_watchdog_round = state.last_watchdog_round
            if watchdog_hit:
                if round_no != last_watchdog_round:
                    watchdog_hits += 1
                last_watchdog_round = round_no
                disabled_until = max(
                    disabled_until,
                    round_no + self._config.cooldown_rounds_after_watchdog,
                )

            updated = ComputeGovernorState(
                ewma_phase2_5_ms=phase2_ewma,
                ewma_root_rollout_ms=root_ewma,
                watchdog_hits=watchdog_hits,
                phase3_disabled_until_round=disabled_until,
                consecutive_very_slow=consecutive,
                full_disabled_day=full_disabled_day,
                halve_root_round=halve_round,
                last_phase3_ms=last_phase3_ms,
                last_scenarios_per_root=last_scenarios,
                last_exact_horizon=last_horizon,
                last_observed_round=round_no,
                last_watchdog_round=last_watchdog_round,
            )
            self._states[team_id] = updated
            return updated

    def _adaptive_budget(
        self,
        base: Phase3Budget,
        state: ComputeGovernorState,
        *,
        chain_seconds: float,
        halve_roots: bool,
        reasons: list[str],
    ) -> Phase3Budget | None:
        roots = base.root_candidates
        scenarios = base.scenarios
        seconds = min(base.seconds, chain_seconds)
        if halve_roots:
            roots = max(1, (roots + 1) // 2)
            reasons.append('slow_previous_turn')

        if state.ewma_root_rollout_ms > 0:
            per_scenario_root_ms = (
                state.ewma_root_rollout_ms
                / max(1, state.last_scenarios_per_root)
            )
            per_scenario_root_ms *= (
                base.exact_horizon / max(1, state.last_exact_horizon)
            )
            target_ms = seconds * 1000 * self._config.budget_utilization
            projected_ms = per_scenario_root_ms * roots * scenarios
            if projected_ms > target_ms and scenarios > 1:
                scenarios = 1
                reasons.append('scenarios_reduced')
            affordable_roots = int(
                target_ms / max(per_scenario_root_ms * scenarios, 0.001)
            )
            if affordable_roots < 2:
                return None
            if affordable_roots < roots:
                roots = affordable_roots
                reasons.append('roots_reduced')

        return Phase3Budget(
            root_candidates=roots,
            scenarios=scenarios,
            exact_horizon=base.exact_horizon,
            seconds=max(0.001, seconds),
        )

    def _ewma(self, previous: float, sample: float) -> float:
        if previous <= 0:
            return sample
        alpha = self._config.ewma_alpha
        return alpha * sample + (1 - alpha) * previous
