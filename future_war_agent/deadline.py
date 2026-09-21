from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


class RequestDeadlineExceeded(TimeoutError):
    """Raised when a request can no longer safely continue a stage."""


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """Absolute deadlines shared by every stage of one HTTP request."""

    started_at: float
    response_deadline: float
    compute_deadline: float
    clock: Callable[[], float] = monotonic

    @classmethod
    def start(
        cls,
        *,
        started_at: float | None = None,
        response_budget_seconds: float = 3.5,
        compute_budget_seconds: float = 3.0,
        clock: Callable[[], float] = monotonic,
    ) -> "RequestBudget":
        if response_budget_seconds <= 0:
            raise ValueError("response_budget_seconds must be positive")
        if compute_budget_seconds <= 0:
            raise ValueError("compute_budget_seconds must be positive")
        if compute_budget_seconds > response_budget_seconds:
            raise ValueError(
                "compute_budget_seconds cannot exceed response_budget_seconds"
            )
        if started_at is None:
            started_at = clock()
        return cls(
            started_at=started_at,
            response_deadline=started_at + response_budget_seconds,
            compute_deadline=started_at + compute_budget_seconds,
            clock=clock,
        )

    def remaining_response(self) -> float:
        return max(0.0, self.response_deadline - self.clock())

    def remaining_compute(self) -> float:
        return max(0.0, self.compute_deadline - self.clock())

    def compute_exhausted(self) -> bool:
        return self.clock() >= self.compute_deadline

    def response_exhausted(self) -> bool:
        return self.clock() >= self.response_deadline

    def child_deadline(self, seconds: float) -> float:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        return min(self.compute_deadline, self.clock() + seconds)

    def checkpoint(self, stage: str, *, response: bool = False) -> None:
        deadline = self.response_deadline if response else self.compute_deadline
        if self.clock() >= deadline:
            kind = "response" if response else "compute"
            raise RequestDeadlineExceeded(
                f"{kind} deadline expired before {stage}"
            )
