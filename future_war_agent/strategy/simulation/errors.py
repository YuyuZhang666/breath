class UnsupportedSimulation(RuntimeError):
    """The observation cannot be modeled without inventing rules."""


class DeadlineExceeded(RuntimeError):
    """The atomic Phase 3 work budget expired."""
