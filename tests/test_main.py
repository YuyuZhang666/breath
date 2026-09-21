import logging
import tempfile
import unittest
from pathlib import Path

from future_war_agent.seclog import KeyFileError
from main import main, parse_port


TEST_KEY = "test-log-key-0123456789-abcdefghijklmnop"


class MainTests(unittest.TestCase):
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

    def test_parse_port_accepts_valid_port(self) -> None:
        self.assertEqual(parse_port(["18080"]), 18080)

    def test_parse_port_rejects_invalid_arguments(self) -> None:
        for argv in ([], ["1", "2"], ["abc"], ["0"], ["65536"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parse_port(argv)

    def test_main_passes_port_to_runner(self) -> None:
        seen: list[int] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            key_file = Path(temp_dir) / "log_secret.key"
            key_file.write_text(TEST_KEY + "\n", encoding="utf-8")
            result = main(
                ["18080"],
                runner=seen.append,
                key_file=key_file,
            )

        self.assertEqual(result, 0)
        self.assertEqual(seen, [18080])

    def test_main_refuses_to_start_without_a_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.key"
            with self.assertRaises(KeyFileError):
                main(["18080"], runner=lambda port: None, key_file=missing)


if __name__ == "__main__":
    unittest.main()
