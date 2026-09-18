import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from future_war_agent.controller import handle_payload
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.server import create_server
from future_war_agent.strategy.engine import StrategyEngine


FIXTURE = Path(__file__).parent / "fixtures" / "interface_request.json"


class PrematchAcceptanceTests(unittest.TestCase):
    def test_supplied_interface_shape_returns_stable_nonempty_response(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        observed = parse_observation(raw)
        engine = StrategyEngine()
        server = create_server(
            0,
            controller=lambda payload: handle_payload(payload, engine.plan),
            host="127.0.0.1",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            request = Request(url, data=FIXTURE.read_bytes(), method="POST")
            with urlopen(request, timeout=5) as response:
                first = response.read()
            duplicate = Request(url, data=FIXTURE.read_bytes(), method="POST")
            with urlopen(duplicate, timeout=5) as response:
                second = response.read()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        payload = json.loads(first.decode("utf-8"))
        own_ids = {str(unit.unit_id) for unit in observed.our.units}
        self.assertEqual(first, second)
        self.assertEqual(set(payload), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertTrue(payload["roleCommandMap"])
        self.assertLessEqual(set(payload["roleCommandMap"]), own_ids)
        self.assertIsNone(observed.robots[0].target_team)
        weapon = next(unit for unit in observed.our.units if unit.role_type == "gatling")
        self.assertNotIn("cooldown", weapon.provided_fields)


if __name__ == "__main__":
    unittest.main()
