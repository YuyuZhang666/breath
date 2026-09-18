# Phase 7A Deterministic Replay Evaluation Design

## 1. Objective

Phase 7A builds an offline evaluation subsystem for comparing strategy
variants against recorded request sequences. It reuses production parsing,
planning, validation, and serialization, but never runs in the HTTP request
path and never changes online policy state.

The subsystem answers three questions with reproducible evidence:

1. Does a strategy emit the same legal responses for the same replay?
2. How did it perform on league points, survival, score, and task activity?
3. Which variant ranks best on the same bounded replay corpus?

## 2. Replay format and bounds

A replay file is UTF-8 JSON with a top-level `cases` array. Each case contains
a unique nonblank name, an outcome (`win`, `draw`, `loss`, or `unknown`), and a
nonempty `turns` array of raw protocol request objects.

Loading is offline and strict. Invalid JSON, duplicate names, unknown outcomes,
non-object turns, more than 256 cases, or more than 1,000 turns per case raises
a contextual `ReplayFormatError`. Evaluation never converts corrupt corpus data
into a safe empty response because that would hide defects.

## 3. Runner pipeline

Each `(variant, case)` pair creates a fresh planner instance so strategy
sessions and match memory cannot leak between cases or variants. Every turn is
processed through:

```text
raw payload
-> parse_observation
-> variant planner
-> validate_decision
-> decision_to_payload
-> canonical JSON bytes and SHA-256 digest
-> metrics accumulator
```

Failures are wrapped with variant, case, and one-based turn context while
preserving the original exception as the cause.

## 4. Results and metrics

`ReplayResult` is immutable and contains:

- ordered response digests and optional profile labels;
- per-turn elapsed nanoseconds from an injectable monotonic clock;
- league points using 3/1/0, with `unknown` contributing zero but remaining
  distinguishable;
- initial/final score and score gain;
- final station survival and minimum observed station health;
- command, prompt, task-accept, and answer-submit counts;
- stable action-kind counts and profile-switch count;
- median and nearest-rank p99 latency.

Timing is diagnostic, not part of response determinism. Tests use a fake clock;
real CLI runs use `perf_counter_ns`.

## 5. Variant comparison

A `ReplayVariant` provides a unique name, a zero-argument planner factory, and
an optional profile reader. All variants run every case. Aggregates sum league
points and score gains, count station survivals and task submissions, and take
the maximum case p99 latency.

Ranking is deterministic:

1. higher total league points;
2. more surviving cases;
3. higher total score gain;
4. lower maximum p99 latency;
5. lexical variant name.

The evaluator does not tune parameters or declare statistical significance in
Phase 7A. Those belong to later offline calibration work.

## 6. Production-engine diagnostics

`StrategyEngine.current_profile(team_id)` exposes only the current immutable
profile under the existing engine lock. It returns `None` for missing teams.
This is a read-only diagnostic boundary for replay traces; it does not expose
or mutate session internals.

## 7. CLI and reporting

`python -m future_war_agent.evaluation REPLAY.json` evaluates the production
`StrategyEngine` variant and writes a stable JSON report to stdout. Reports use
plain JSON values, stable ordering, and contain no raw prompts, answers, command
output, or full observations.

## 8. Acceptance criteria

- identical replay and planner produce identical response digests;
- state never leaks between cases or variants;
- production validation filters illegal planner commands before scoring;
- 3/1/0 points, survival, score, task, profile, and latency metrics are correct;
- variant ordering follows the documented stable ranking;
- malformed or oversized replay files fail with contextual errors;
- a real Phase 3 day/night/duplicate replay evaluates successfully;
- online controller behavior and the complete existing suite remain unchanged;
- Python 3.11-compatible compile, diff, and independent review gates pass.
