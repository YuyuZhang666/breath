import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from future_war_agent.seclog import KEY_FILE_NAME, configure
from future_war_agent.server import serve


USAGE = "Usage: python main.py <port>"
ROOT = Path(__file__).resolve().parent
LOG_KEY_FILE = ROOT / KEY_FILE_NAME


def parse_port(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(USAGE)
    try:
        port = int(argv[0])
    except ValueError as error:
        raise SystemExit(USAGE) from error
    if not 1 <= port <= 65535:
        raise SystemExit(USAGE)
    return port


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[[int], None] = serve,
    *,
    key_file: str | Path = LOG_KEY_FILE,
) -> int:
    port = parse_port(sys.argv[1:] if argv is None else argv)
    configure(level=logging.INFO, key_file=key_file)
    runner(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
