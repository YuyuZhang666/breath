"""Command-line tools for encrypted Future War Agent logs."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from .seclog import DEFAULT_KEY, decrypt_lines
else:  # Support copying seclog.py and logtool.py to another computer.
    from seclog import DEFAULT_KEY, decrypt_lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logtool.py")
    commands = parser.add_subparsers(dest="command", required=True)
    decrypt_parser = commands.add_parser(
        "decrypt",
        help="decrypt an ENC1 log file",
    )
    decrypt_parser.add_argument("file", type=Path)
    decrypt_parser.add_argument("--key", default=DEFAULT_KEY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "decrypt":
        raise AssertionError(f"unsupported command: {args.command}")

    text = args.file.read_text(encoding="utf-8", errors="replace")
    output = decrypt_lines(text.splitlines(), args.key)
    sys.stdout.write(output)
    if text.endswith(("\n", "\r")):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
