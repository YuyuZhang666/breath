"""Shared-key encryption helpers for line-oriented application logs.

The format intentionally matches the portable ``ENC1`` scheme used by the
companion log tool. It relies only on the Python standard library.
"""

import hashlib
import logging
import sys
from base64 import b64decode, b64encode
from collections.abc import Iterable
from typing import TextIO


DEFAULT_KEY = "future-war-agent-log-key-v1"
MARKER = "ENC1:"
IV = b"future-war-iv"


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


def encrypt(plaintext: str, key: str = DEFAULT_KEY) -> str:
    """Encrypt one UTF-8 log entry and return its line-safe representation."""

    data = plaintext.encode("utf-8")
    stream = _keystream(key.encode("utf-8"), len(data))
    cipher = bytes(value ^ stream[index] for index, value in enumerate(data))
    return MARKER + b64encode(cipher).decode("ascii")


def decrypt(payload: str, key: str = DEFAULT_KEY) -> str:
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
        stream: TextIO | None = None,
        key: str = DEFAULT_KEY,
    ) -> None:
        super().__init__()
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
    key: str = DEFAULT_KEY,
    stream: TextIO | None = None,
) -> None:
    """Configure root logging with encrypted routine logs and plaintext errors."""

    target = stream if stream is not None else sys.stdout
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    encrypted = EncryptingHandler(target, key)
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
    key: str = DEFAULT_KEY,
) -> str:
    """Decrypt a mixed encrypted/plaintext log stream line by line."""

    return "\n".join(decrypt(line.rstrip("\r\n"), key) for line in lines)
