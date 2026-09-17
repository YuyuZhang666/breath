import logging
import sys
from collections.abc import Callable, Sequence

from future_war_agent.server import serve


USAGE = "Usage: python main.py <port>"


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
) -> int:
    port = parse_port(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    runner(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
