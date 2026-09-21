import binascii
import io
import logging
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from future_war_agent.logtool import main as logtool_main
from future_war_agent.seclog import (
    MARKER,
    EncryptingHandler,
    configure,
    decrypt,
    decrypt_lines,
    encrypt,
    is_encrypted,
)


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
                self.assertEqual(encrypt(plaintext), ciphertext)
                self.assertEqual(decrypt(ciphertext), plaintext)

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
        self.assertEqual(decrypt("plain error"), "plain error")

    def test_decrypt_rejects_malformed_base64(self) -> None:
        with self.assertRaises(binascii.Error):
            decrypt("ENC1:not-base64")

    def test_decrypt_lines_supports_mixed_logs_and_crlf(self) -> None:
        encrypted = encrypt("routine")

        self.assertEqual(
            decrypt_lines([encrypted + "\n", "plain error\r\n"]),
            "routine\nplain error",
        )

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
        configure(level=logging.INFO, stream=stream)
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
        configure(stream=first_stream)
        configure(stream=second_stream)

        logging.getLogger(f"{__name__}.repeat").info("one record")

        self.assertEqual(first_stream.getvalue(), "")
        self.assertEqual(len(second_stream.getvalue().splitlines()), 1)

    def test_concurrent_records_remain_complete_lines(self) -> None:
        stream = io.StringIO()
        configure(stream=stream)
        logger = logging.getLogger(f"{__name__}.concurrent")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda value: logger.info("record-%03d", value), range(100)))

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 100)
        self.assertTrue(all(line.startswith(MARKER) for line in lines))
        plaintext = [decrypt(line) for line in lines]
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

        handler = EncryptingHandler(BrokenStream())
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
            path.write_text(
                encrypt("secret turn") + "\nplain error\n",
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                result = logtool_main(["decrypt", str(path)])

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "secret turn\nplain error\n")

    def test_logtool_accepts_an_explicit_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent.log"
            path.write_text(encrypt("secret", "other-key"), encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                result = logtool_main(
                    ["decrypt", str(path), "--key", "other-key"]
                )

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "secret")


if __name__ == "__main__":
    unittest.main()
