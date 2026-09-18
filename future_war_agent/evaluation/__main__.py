import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.protocol.models import Observation

from .comparison import evaluate_variants
from .errors import ReplayError
from .loader import load_replay_file
from .models import ReplayVariant
from .report import report_to_json


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    parser = argparse.ArgumentParser(
        prog="python -m future_war_agent.evaluation",
        description="Evaluate the production strategy against a replay corpus.",
    )
    parser.add_argument("replay", type=Path)
    arguments = parser.parse_args(argv)

    try:
        corpus = load_replay_file(arguments.replay)
        report = evaluate_variants(
            corpus,
            (
                ReplayVariant(
                    "production",
                    StrategyEngine,
                    profile_reader=_production_profile,
                ),
            ),
        )
    except ReplayError as error:
        print(f"replay evaluation failed: {error}", file=errors)
        return 1

    print(report_to_json(report), file=output)
    return 0


def _production_profile(
    planner: object, observation: Observation
) -> object | None:
    if not isinstance(planner, StrategyEngine):
        raise TypeError("production replay requires StrategyEngine")
    return planner.current_profile(observation.our.team_id)


if __name__ == "__main__":
    raise SystemExit(main())
