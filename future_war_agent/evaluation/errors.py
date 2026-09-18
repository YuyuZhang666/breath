class ReplayError(ValueError):
    """Base class for replay input and execution failures."""


class ReplayFormatError(ReplayError):
    """Raised when an offline replay corpus violates its contract."""


class ReplayEvaluationError(ReplayError):
    """Raised when a replay turn cannot be evaluated."""
