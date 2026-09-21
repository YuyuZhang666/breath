"""Command-line tools for encrypted Future War Agent logs."""

import argparse
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from .seclog import (
        KEY_FILE_NAME,
        KeyFileError,
        decrypt_lines,
        load_key_file,
    )
else:  # Support copying seclog.py and logtool.py to another computer.
    from seclog import (  # type: ignore[no-redef]
        KEY_FILE_NAME,
        KeyFileError,
        decrypt_lines,
        load_key_file,
    )


GENERATED_KEY_BYTES = 48


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logtool.py")
    commands = parser.add_subparsers(dest="command", required=True)
    decrypt_parser = commands.add_parser(
        "decrypt",
        help="decrypt an ENC1 log file",
    )
    decrypt_parser.add_argument("file", type=Path)
    decrypt_parser.add_argument(
        "--key-file",
        type=Path,
        default=Path(KEY_FILE_NAME),
        help=f"UTF-8 key file (default: {KEY_FILE_NAME})",
    )

    generate_parser = commands.add_parser(
        "generate-key",
        help="create a new random log key file",
    )
    generate_parser.add_argument(
        "--key-file",
        type=Path,
        default=Path(KEY_FILE_NAME),
        help=f"destination key file (default: {KEY_FILE_NAME})",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing key file",
    )
    return parser


def _generate_key_file(path: Path, *, overwrite: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(secrets.token_urlsafe(GENERATED_KEY_BYTES) + "\n")
    finally:
        if descriptor != -1:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.command == "generate-key":
        try:
            _generate_key_file(args.key_file, overwrite=args.force)
        except FileExistsError:
            parser.error(
                f"key file already exists; use --force to replace it: "
                f"{args.key_file}"
            )
        except OSError as error:
            parser.error(f"cannot write key file {args.key_file}: {error}")
        sys.stdout.write(f"{args.key_file}\n")
        return 0

    if args.command != "decrypt":
        raise AssertionError(f"unsupported command: {args.command}")

    try:
        key = load_key_file(args.key_file, minimum_bytes=1)
    except KeyFileError as error:
        parser.error(str(error))
    text = args.file.read_text(encoding="utf-8", errors="replace")
    output = decrypt_lines(text.splitlines(), key)
    sys.stdout.write(output)
    if text.endswith(("\n", "\r")):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
