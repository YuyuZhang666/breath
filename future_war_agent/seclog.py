"""Shared-key encryption helpers for line-oriented application logs.

The format intentionally matches the portable ``ENC1`` scheme used by the
companion log tool. It relies only on the Python standard library.
"""

import hashlib
import logging
import sys
from base64 import b64decode, b64encode
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO


KEY_FILE_NAME = "log_secret.key"
MIN_KEY_BYTES = 32
MAX_KEY_FILE_BYTES = 4096
MARKER = "ENC1:"
IV = b"future-war-iv"


class KeyFileError(ValueError):
    """Raised when a log encryption key file cannot be used safely."""


def load_key_file(
    path: str | Path,
    *,
    minimum_bytes: int = MIN_KEY_BYTES,
) -> str:
    """Read and validate a UTF-8 key without exposing it in diagnostics."""

    key_path = Path(path)
    try:
        raw = key_path.read_bytes()
    except FileNotFoundError:
        raise KeyFileError(
            f"log encryption key file not found: {key_path}"
        ) from None
    except OSError as error:
        raise KeyFileError(
            f"cannot read log encryption key file: {key_path}"
        ) from error

    if len(raw) > MAX_KEY_FILE_BYTES:
        raise KeyFileError(
            f"log encryption key file is too large: {key_path}"
        )
    try:
        key = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as error:
        raise KeyFileError(
            f"log encryption key file must be UTF-8: {key_path}"
        ) from error
    if "\r" in key or "\n" in key:
        raise KeyFileError(
            f"log encryption key file must contain one line: {key_path}"
        )
    if len(key.encode("utf-8")) < minimum_bytes:
        raise KeyFileError(
            "log encryption key must contain at least "
            f"{minimum_bytes} UTF-8 bytes: {key_path}"
        )
    return key


def _keystream(key: bytes, length: int) -> bytes:
    if length < 0:
        raise ValueError("length must be non-negative")
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(
            hashlib.sha256(
                IV + key + counter.to_bytes(8, "big"),
            ).digest()
        )
        counter += 1
    return bytes(output[:length])


def encrypt(plaintext: str, key: str) -> str:
    """Encrypt one UTF-8 log entry and return its line-safe representation."""

    data = plaintext.encode("utf-8")
    stream = _keystream(key.encode("utf-8"), len(data))
    cipher = bytes(value ^ stream[index] for index, value in enumerate(data))
    return MARKER + b64encode(cipher).decode("ascii")


def decrypt(payload: str, key: str) -> str:
    """Decrypt an ``ENC1`` entry, passing plaintext entries through unchanged."""

    if not is_encrypted(payload):
        return payload
    cipher = b64decode(payload[len(MARKER) :], validate=True)
    stream = _keystream(key.encode("utf-8"), len(cipher))
    plain = bytes(value ^ stream[index] for index, value in enumerate(cipher))
    return plain.decode("utf-8", errors="replace")


def is_encrypted(payload: str) -> bool:
    return payload.startswith(MARKER)


class _MaximumLevelFilter(logging.Filter):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.maximum = maximum

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.maximum


class EncryptingHandler(logging.Handler):
    """Format and encrypt each accepted log record as one physical line."""

    terminator = "\n"

    def __init__(
        self,
        stream: TextIO | None,
        key: str,
    ) -> None:
        super().__init__()
        if not key:
            raise ValueError("an explicit log encryption key is required")
        self.stream = stream if stream is not None else sys.stdout
        self.key = key

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = encrypt(self.format(record), self.key)
            self.stream.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Report encryption failures without echoing the plaintext record."""

        try:
            sys.stderr.write("encrypted logging failed\n")
        except Exception:
            pass

    def flush(self) -> None:
        self.acquire()
        try:
            if self.stream is not None and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()


def configure(
    level: int = logging.INFO,
    key: str | None = None,
    stream: TextIO | None = None,
    *,
    key_file: str | Path = KEY_FILE_NAME,
) -> None:
    """Configure root logging with encrypted routine logs and plaintext errors."""

    active_key = load_key_file(key_file) if key is None else key
    if not active_key:
        raise ValueError("an explicit log encryption key is required")
    target = stream if stream is not None else sys.stdout
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    encrypted = EncryptingHandler(target, active_key)
    encrypted.setLevel(logging.DEBUG)
    encrypted.addFilter(_MaximumLevelFilter(logging.WARNING))
    encrypted.setFormatter(formatter)
    root.addHandler(encrypted)

    plain_errors = logging.StreamHandler(target)
    plain_errors.setLevel(logging.ERROR)
    plain_errors.setFormatter(formatter)
    root.addHandler(plain_errors)


def decrypt_lines(
    lines: Iterable[str],
    key: str,
) -> str:
    """Decrypt a mixed encrypted/plaintext log stream line by line."""

    return "\n".join(decrypt(line.rstrip("\r\n"), key) for line in lines)
