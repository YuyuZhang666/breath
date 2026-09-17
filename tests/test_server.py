import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from future_war_agent.fallback import safe_payload
from future_war_agent.server import create_server


FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


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

    def post(self, body: bytes) -> tuple[int, dict[str, object]]:
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

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
