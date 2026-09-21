import json
import socket
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from future_war_agent.fallback import safe_payload
from future_war_agent.server import _handler_for, create_server


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

    def test_request_start_precedes_body_read_and_reaches_controller(self) -> None:
        body = FIXTURE.read_bytes()
        body_read_at: list[float] = []
        controller_starts: list[float] = []

        class RecordingBody(BytesIO):
            def read(self, size: int = -1) -> bytes:
                body_read_at.append(time.monotonic())
                return super().read(size)

        def controller(
            _: object,
            *,
            request_started_at: float,
        ) -> dict[str, object]:
            controller_starts.append(request_started_at)
            return safe_payload()

        request = SimpleNamespace(
            headers={"Content-Length": str(len(body))},
            rfile=RecordingBody(body),
            wfile=BytesIO(),
            send_response=lambda *_: None,
            send_header=lambda *_: None,
            end_headers=lambda: None,
        )

        _handler_for(controller).do_POST(request)

        self.assertEqual(len(controller_starts), 1)
        self.assertEqual(len(body_read_at), 1)
        self.assertLessEqual(controller_starts[0], body_read_at[0])

    def test_client_disconnect_during_response_is_contained(self) -> None:
        class DisconnectingWriter:
            def write(self, _: bytes) -> None:
                raise BrokenPipeError("client disconnected")

        request = SimpleNamespace(
            headers={"Content-Length": "2"},
            rfile=BytesIO(b"{}"),
            wfile=DisconnectingWriter(),
            send_response=lambda *_: None,
            send_header=lambda *_: None,
            end_headers=lambda: None,
        )

        _handler_for(lambda _: safe_payload()).do_POST(request)

    def test_oversized_body_is_rejected_before_controller_runs(self) -> None:
        received: list[object] = []

        def recording_controller(payload: object) -> dict[str, object]:
            received.append(payload)
            return {"unexpected": True}

        server = create_server(
            0,
            controller=recording_controller,
            host="127.0.0.1",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        body = b'{"padding":"' + (b"x" * 1_048_576) + b'"}'
        try:
            request = Request(url, data=body, method="POST")
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(payload, safe_payload())
        self.assertEqual(received, [])

    def test_oversized_declared_body_returns_without_waiting_for_body(self) -> None:
        received: list[object] = []
        server = create_server(
            0,
            controller=lambda payload: received.append(payload) or {},
            host="127.0.0.1",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = socket.create_connection(
            ("127.0.0.1", server.server_port), timeout=2
        )
        client.settimeout(2)
        try:
            started = time.monotonic()
            client.sendall(
                b"POST / HTTP/1.0\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 1048577\r\n"
                b"\r\n"
            )
            chunks: list[bytes] = []
            while chunk := client.recv(4096):
                chunks.append(chunk)
            response = b"".join(chunks)
            elapsed = time.monotonic() - started
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertIn(b" 200 ", response.partition(b"\r\n")[0])
        self.assertIn(b'"roleCommandMap":{}', response)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
