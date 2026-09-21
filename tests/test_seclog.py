import binascii
import io
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from future_war_agent.logtool import main as logtool_main
from future_war_agent.seclog import (
    MARKER,
    EncryptingHandler,
    KeyFileError,
    configure,
    decrypt,
    decrypt_lines,
    encrypt,
    is_encrypted,
    load_key_file,
)


LEGACY_KEY = "future-war-agent-log-key-v1"
TEST_KEY = "test-log-key-0123456789-abcdefghijklmnop"
ROOT = Path(__file__).resolve().parents[1]


class SecureLogTests(unittest.TestCase):
    def setUp(self) -> None:
        root = logging.getLogger()
        self.original_handlers = list(root.handlers)
        self.original_level = root.level

    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            if handler not in self.original_handlers:
                handler.close()
        for handler in self.original_handlers:
            root.addHandler(handler)
        root.setLevel(self.original_level)

    def test_protocol_vectors_lock_cross_computer_compatibility(self) -> None:
        vectors = (
            ("", "ENC1:"),
            ("hello", "ENC1:DyseLd8="),
            ("\u4f5c\u6218\u65e5\u5fd7", "ENC1:g/Pupzi8Oso28Ifj"),
            ("A" * 33, "ENC1:Jg8zAPFlnRzSVHk12YtwRGjz36U9H3FDJJgzoON48Qhk"),
        )

        for plaintext, ciphertext in vectors:
            with self.subTest(plaintext=plaintext):
                self.assertEqual(encrypt(plaintext, LEGACY_KEY), ciphertext)
                self.assertEqual(decrypt(ciphertext, LEGACY_KEY), plaintext)

    def test_encrypt_is_deterministic_and_round_trips_utf8(self) -> None:
        plaintext = "turn_telemetry \u56de\u5408=7 \U0001f6e1"

        first = encrypt(plaintext, "test-key")
        second = encrypt(plaintext, "test-key")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith(MARKER))
        self.assertTrue(is_encrypted(first))
        self.assertEqual(decrypt(first, "test-key"), plaintext)

    def test_decrypt_passes_plaintext_through(self) -> None:
        self.assertFalse(is_encrypted("plain error"))
        self.assertEqual(decrypt("plain error", TEST_KEY), "plain error")

    def test_decrypt_rejects_malformed_base64(self) -> None:
        with self.assertRaises(binascii.Error):
            decrypt("ENC1:not-base64", TEST_KEY)

    def test_decrypt_lines_supports_mixed_logs_and_crlf(self) -> None:
        encrypted = encrypt("routine", TEST_KEY)

        self.assertEqual(
            decrypt_lines(
                [encrypted + "\n", "plain error\r\n"],
                TEST_KEY,
            ),
            "routine\nplain error",
        )

    def test_load_key_file_requires_a_nontrivial_single_line_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "log_secret.key"
            with self.assertRaises(KeyFileError):
                load_key_file(path)

            path.write_text("too-short\n", encoding="utf-8")
            with self.assertRaises(KeyFileError):
                load_key_file(path)

            path.write_text(TEST_KEY + "\n", encoding="utf-8")
            self.assertEqual(load_key_file(path), TEST_KEY)

    def test_configure_loads_key_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "log_secret.key"
            path.write_text(TEST_KEY + "\n", encoding="utf-8")
            stream = io.StringIO()
            configure(stream=stream, key_file=path)
            logging.getLogger(f"{__name__}.key_file").info("protected")

        line = stream.getvalue().strip()
        self.assertIn("protected", decrypt(line, TEST_KEY))

    def test_configure_encrypts_through_warning_and_keeps_errors_plain(self) -> None:
        stream = io.StringIO()
        configure(level=logging.DEBUG, key="test-key", stream=stream)
        logger = logging.getLogger(f"{__name__}.levels")
        logger.setLevel(logging.NOTSET)

        logger.debug("debug detail")
        logger.info("turn telemetry")
        logger.warning("watchdog fallback")
        logger.error("visible failure")
        logger.critical("visible critical failure")

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertTrue(all(line.startswith(MARKER) for line in lines[:3]))
        self.assertTrue(all(not line.startswith(MARKER) for line in lines[3:]))
        self.assertIn("ERROR", lines[3])
        self.assertIn("CRITICAL", lines[4])
        decrypted = [decrypt(line, "test-key") for line in lines[:3]]
        self.assertIn("DEBUG", decrypted[0])
        self.assertIn("INFO", decrypted[1])
        self.assertIn("WARNING", decrypted[2])

    def test_exception_traceback_is_plaintext_only(self) -> None:
        stream = io.StringIO()
        configure(level=logging.INFO, key=TEST_KEY, stream=stream)
        logger = logging.getLogger(f"{__name__}.exception")

        try:
            raise RuntimeError("visible diagnostic")
        except RuntimeError:
            logger.exception("operation failed")

        output = stream.getvalue()
        self.assertNotIn(MARKER, output)
        self.assertIn("ERROR", output)
        self.assertIn("RuntimeError: visible diagnostic", output)

    def test_reconfigure_replaces_handlers_without_duplicate_output(self) -> None:
        first_stream = io.StringIO()
        second_stream = io.StringIO()
        configure(key=TEST_KEY, stream=first_stream)
        configure(key=TEST_KEY, stream=second_stream)

        logging.getLogger(f"{__name__}.repeat").info("one record")

        self.assertEqual(first_stream.getvalue(), "")
        self.assertEqual(len(second_stream.getvalue().splitlines()), 1)

    def test_concurrent_records_remain_complete_lines(self) -> None:
        stream = io.StringIO()
        configure(key=TEST_KEY, stream=stream)
        logger = logging.getLogger(f"{__name__}.concurrent")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda value: logger.info("record-%03d", value), range(100)))

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 100)
        self.assertTrue(all(line.startswith(MARKER) for line in lines))
        plaintext = [decrypt(line, TEST_KEY) for line in lines]
        for value in range(100):
            self.assertTrue(
                any(f"record-{value:03d}" in line for line in plaintext),
                value,
            )

    def test_handler_failure_does_not_echo_plaintext(self) -> None:
        class BrokenStream:
            def write(self, value: str) -> None:
                raise OSError("broken")

            def flush(self) -> None:
                raise OSError("broken")

        handler = EncryptingHandler(BrokenStream(), TEST_KEY)
        stderr = io.StringIO()
        record = logging.LogRecord(
            "secure",
            logging.INFO,
            __file__,
            1,
            "private strategy detail",
            (),
            None,
        )

        with redirect_stderr(stderr):
            handler.handle(record)

        self.assertIn("encrypted logging failed", stderr.getvalue())
        self.assertNotIn("private strategy detail", stderr.getvalue())

    def test_logtool_decrypts_file_and_preserves_plaintext_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent.log"
            key_path = Path(temp_dir) / "log_secret.key"
            key_path.write_text(TEST_KEY + "\n", encoding="utf-8")
            path.write_text(
                encrypt("secret turn", TEST_KEY) + "\nplain error\n",
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                result = logtool_main(
                    ["decrypt", str(path), "--key-file", str(key_path)]
                )

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "secret turn\nplain error\n")

    def test_logtool_runs_standalone_on_another_computer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for name in ("seclog.py", "logtool.py"):
                shutil.copy2(ROOT / "future_war_agent" / name, temp_path / name)
            (temp_path / "log_secret.key").write_text(
                TEST_KEY + "\n",
                encoding="utf-8",
            )
            (temp_path / "agent.log").write_text(
                encrypt("cross-computer", TEST_KEY) + "\nplain error\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "logtool.py",
                    "decrypt",
                    "agent.log",
                    "--key-file",
                    "log_secret.key",
                ],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "cross-computer\nplain error\n")

    def test_logtool_generates_and_rotates_a_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "log_secret.key"
            output = io.StringIO()

            with redirect_stdout(output):
                result = logtool_main(
                    ["generate-key", "--key-file", str(key_path)]
                )
            first_key = load_key_file(key_path)
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    logtool_main(
                        ["generate-key", "--key-file", str(key_path)]
                    )
            with redirect_stdout(io.StringIO()):
                result = logtool_main(
                    [
                        "generate-key",
                        "--key-file",
                        str(key_path),
                        "--force",
                    ]
                )
            second_key = load_key_file(key_path)

        self.assertEqual(result, 0)
        self.assertIn(str(key_path), output.getvalue())
        self.assertNotEqual(first_key, second_key)


if __name__ == "__main__":
    unittest.main()
