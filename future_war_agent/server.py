import json
import logging
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from future_war_agent.controller import handle_payload
from future_war_agent.fallback import safe_payload


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Controller = Callable[[object], dict[str, object]]


def _encode_response(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _handler_for(controller: Controller) -> type[BaseHTTPRequestHandler]:
    class AgentRequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ValueError("Content-Length is required")
                content_length = int(raw_length)
                if content_length < 0:
                    raise ValueError("Content-Length cannot be negative")

                request_body = self.rfile.read(content_length)
                request_payload: Any = json.loads(request_body.decode("utf-8"))
                response_body = _encode_response(controller(request_payload))
            except Exception:
                LOGGER.exception("request handling failed")
                response_body = _encode_response(safe_payload())

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return AgentRequestHandler


def create_server(
    port: int,
    controller: Controller = handle_payload,
    host: str = "0.0.0.0",
) -> ThreadingHTTPServer:
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 <= port <= 65535
    ):
        raise ValueError("port must be an integer from 0 through 65535")
    return ThreadingHTTPServer((host, port), _handler_for(controller))


def serve(port: int) -> None:
    with create_server(port) as server:
        LOGGER.info("agent server listening on port %d", port)
        server.serve_forever()
