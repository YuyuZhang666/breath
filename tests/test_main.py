import logging
import unittest

from main import main, parse_port


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

        self.assertEqual(main(["18080"], runner=seen.append), 0)
        self.assertEqual(seen, [18080])


if __name__ == "__main__":
    unittest.main()
