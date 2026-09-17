import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from future_war_agent.fallback import safe_payload
from future_war_agent.server import create_server


FIXTURE = Path(__file__).parent / "fixtures" / "request.json"
PHASE3_DAY = Path(__file__).parent / "fixtures" / "phase3_day_request.json"
PHASE3_NIGHT = Path(__file__).parent / "fixtures" / "phase3_night_request.json"


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(0, host="127.0.0.1")
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post_raw(self, body: bytes) -> tuple[int, bytes]:
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return response.status, response.read()

    def post(self, body: bytes) -> tuple[int, dict[str, object]]:
        status, raw = self.post_raw(body)
        return status, json.loads(raw.decode("utf-8"))

    def test_phase3_consecutive_requests_are_nonempty_and_cached(self) -> None:
        day_status, day_raw = self.post_raw(PHASE3_DAY.read_bytes())
        night_status, first_night_raw = self.post_raw(PHASE3_NIGHT.read_bytes())
        duplicate_status, duplicate_night_raw = self.post_raw(
            PHASE3_NIGHT.read_bytes()
        )
        day = json.loads(day_raw)
        night = json.loads(first_night_raw)

        self.assertEqual((day_status, night_status, duplicate_status), (200, 200, 200))
        self.assertEqual(set(day), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertEqual(set(night), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertTrue(night["roleCommandMap"])
        self.assertEqual(first_night_raw, duplicate_night_raw)

    def test_valid_request_returns_complete_safe_response(self) -> None:
        status, payload = self.post(FIXTURE.read_bytes())

        self.assertEqual(status, 200)
        self.assertEqual(payload, safe_payload())

    def test_malformed_json_returns_safe_response(self) -> None:
        status, payload = self.post(b"{not json")

        self.assertEqual(status, 200)
        self.assertEqual(payload, safe_payload())

    def test_repeated_requests_remain_independent(self) -> None:
        first = self.post(FIXTURE.read_bytes())
        second = self.post(FIXTURE.read_bytes())

        self.assertEqual(first, second)

    def test_controller_exception_returns_safe_response(self) -> None:
        def broken_controller(_: object) -> dict[str, object]:
            raise RuntimeError("private failure detail")

        server = create_server(
            0,
            controller=broken_controller,
            host="127.0.0.1",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            request = Request(url, data=FIXTURE.read_bytes(), method="POST")
            with urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload, safe_payload())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
