import json
import logging
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import Parameter, signature
from time import monotonic
from typing import Any

from future_war_agent.controller import handle_payload
from future_war_agent.fallback import safe_payload


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Controller = Callable[..., dict[str, object]]
MAX_REQUEST_BYTES = 1_048_576


def _encode_response(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _handler_for(controller: Controller) -> type[BaseHTTPRequestHandler]:
    class AgentRequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request_started_at = monotonic()
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ValueError("Content-Length is required")
                content_length = int(raw_length)
                if content_length < 0:
                    raise ValueError("Content-Length cannot be negative")
                if content_length > MAX_REQUEST_BYTES:
                    self.close_connection = True
                    _drain_available_request_body(
                        self.rfile,
                        self.connection,
                        content_length,
                    )
                    raise ValueError("request body exceeds size limit")

                request_body = self.rfile.read(content_length)
                request_payload: Any = json.loads(request_body.decode("utf-8"))
                if _accepts_keyword(controller, "request_started_at"):
                    response_payload = controller(
                        request_payload,
                        request_started_at=request_started_at,
                    )
                else:
                    response_payload = controller(request_payload)
                response_body = _encode_response(response_payload)
            except Exception:
                LOGGER.exception("request handling failed")
                response_body = _encode_response(safe_payload())

            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                LOGGER.info("client disconnected before response completed")

        def log_message(self, format: str, *args: object) -> None:
            return

    return AgentRequestHandler


def _accepts_keyword(function: Controller, name: str) -> bool:
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _drain_available_request_body(
    stream: Any,
    connection: Any,
    length: int,
) -> None:
    deadline = monotonic() + 0.05
    original_timeout = connection.gettimeout()
    remaining = length
    try:
        while remaining:
            time_left = deadline - monotonic()
            if time_left <= 0:
                return
            connection.settimeout(time_left)
            chunk = stream.read(min(remaining, 65_536))
            if not chunk:
                return
            remaining -= len(chunk)
    except (OSError, TimeoutError):
        return
    finally:
        connection.settimeout(original_timeout)


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
