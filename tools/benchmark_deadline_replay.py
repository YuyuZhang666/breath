import copy
import hashlib
import json
import sys
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_war_agent.controller import handle_payload
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.telemetry import TelemetryRecorder


FIXTURES = ROOT / 'tests' / 'fixtures'


def main() -> None:
    day = json.loads(
        (FIXTURES / 'phase3_day_request.json').read_text(encoding='utf-8')
    )
    night = json.loads(
        (FIXTURES / 'phase3_night_request.json').read_text(encoding='utf-8')
    )
    telemetry = TelemetryRecorder(max_samples=256)
    engine = StrategyEngine(telemetry=telemetry)
    digest = hashlib.sha256()

    for round_no in range(1, 131):
        payload = copy.deepcopy(day if round_no <= 70 else night)
        payload['roundNo'] = round_no
        response = handle_payload(
            payload,
            planner=engine.plan,
            telemetry=telemetry,
        )
        digest.update(
            json.dumps(
                response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        )

    samples = telemetry.snapshot()
    latencies = sorted(sample.request_total_ms for sample in samples)
    p99_index = ceil(0.99 * len(latencies)) - 1
    result = {
        'rounds': len(samples),
        'median_request_ms': median(latencies),
        'p99_request_ms': latencies[p99_index],
        'max_request_ms': max(latencies),
        'forecast_modes': dict(sorted(Counter(
            sample.forecast_mode for sample in samples
        ).items())),
        'fallback_reasons': dict(sorted(Counter(
            sample.fallback_reason for sample in samples
        ).items())),
        'watchdog_count': sum(sample.watchdog_hit for sample in samples),
        'timeout_prevented_count': sum(
            sample.timeout_prevented for sample in samples
        ),
        'response_digest': digest.hexdigest(),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
