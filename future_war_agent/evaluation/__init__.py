"""Deterministic offline strategy replay and evaluation."""

from .comparison import evaluate_variants, rank_aggregates
from .errors import ReplayEvaluationError, ReplayFormatError
from .loader import load_replay_data, load_replay_file
from .models import (
    EvaluationReport,
    MatchOutcome,
    ReplayCase,
    ReplayCorpus,
    ReplayMetrics,
    ReplayResult,
    ReplayVariant,
    VariantAggregate,
)
from .runner import run_case
from .report import report_to_data, report_to_json

__all__ = [
    "EvaluationReport",
    "MatchOutcome",
    "ReplayCase",
    "ReplayCorpus",
    "ReplayEvaluationError",
    "ReplayFormatError",
    "ReplayMetrics",
    "ReplayResult",
    "ReplayVariant",
    "VariantAggregate",
    "evaluate_variants",
    "load_replay_data",
    "load_replay_file",
    "run_case",
    "rank_aggregates",
    "report_to_data",
    "report_to_json",
]
