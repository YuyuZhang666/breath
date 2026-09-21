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
    phase2_ms: float = 0.0
    phase3_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    simulation_ms: float = 0.0
    validation_ms: float = 0.0
    serialization_ms: float = 0.0
    total_ms: float = 0.0
    root_candidate_count: int = 0
    scenario_count: int = 0
    simulation_count: int = 0
    phase3_level: str = 'none'
    phase3_fallback_count: int = 0
    fallback_used: bool = False
    watchdog_hit: bool = False
    controller_cache_hit: bool = False
    duplicate_request: bool = False


@dataclass(slots=True)
class _ActiveTurn:
    started_ns: int
    team_id: str = ''
    round_no: int = 0
    phase: str = 'unknown'
    parser_ms: float = 0.0
    director_ms: float = 0.0
    task_ms: float = 0.0
    phase2_ms: float = 0.0
    phase3_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    simulation_ms: float = 0.0
    validation_ms: float = 0.0
    serialization_ms: float = 0.0
    root_candidate_count: int = 0
    scenario_count: int = 0
    simulation_count: int = 0
    phase3_level: str = 'none'
    phase3_fallback_count: int = 0
    fallback_used: bool = False
    watchdog_hit: bool = False
    controller_cache_hit: bool = False
    duplicate_request: bool = False


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
            phase2_ms=active.phase2_ms,
            phase3_ms=active.phase3_ms,
            candidate_generation_ms=active.candidate_generation_ms,
            simulation_ms=active.simulation_ms,
            validation_ms=active.validation_ms,
            serialization_ms=active.serialization_ms,
            total_ms=total_ms,
            root_candidate_count=active.root_candidate_count,
            scenario_count=active.scenario_count,
            simulation_count=active.simulation_count,
            phase3_level=active.phase3_level,
            phase3_fallback_count=active.phase3_fallback_count,
            fallback_used=active.fallback_used,
            watchdog_hit=active.watchdog_hit,
            controller_cache_hit=active.controller_cache_hit,
            duplicate_request=active.duplicate_request,
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
