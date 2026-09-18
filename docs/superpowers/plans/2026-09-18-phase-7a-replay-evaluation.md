# Phase 7A Replay Evaluation Implementation Plan

**Goal:** Build a deterministic offline replay and A/B evaluation framework on
top of the production protocol and strategy engine.

**Architecture:** Add an isolated `future_war_agent.evaluation` package with
immutable models, strict loading, a production-boundary runner, aggregation,
stable reporting, and a stdout CLI. Add one read-only profile accessor to the
strategy engine.

## Engineering rules

- Use TDD for every behavior and observe the expected failure first.
- Create a fresh planner for every variant/case pair.
- Never add replay work to the HTTP request path.
- Bound corpus size before evaluation and retain no raw response content in
  reports.
- Use only Python 3.11-compatible standard-library features.

### Task 1: Replay contracts and strict loader

- Create `evaluation/models.py`, `evaluation/errors.py`, and
  `evaluation/loader.py`.
- Define outcomes, cases, variants, immutable results, and corpus caps.
- Test valid parsing, duplicate names, invalid outcomes/turns, empty cases, and
  case/turn limits.
- Commit the contracts and loader.

### Task 2: Deterministic replay runner

- Create `evaluation/runner.py`.
- Process each turn through production parse, planner, validation, serializer,
  canonical JSON, and SHA-256.
- Accumulate score, station, action, prompt, task, profile, and injected-clock
  latency metrics.
- Wrap errors with variant/case/turn context.
- Test response determinism, validation filtering, metric math, state isolation,
  profile switches, median, and nearest-rank p99.
- Commit the runner.

### Task 3: Variant aggregation and ranking

- Evaluate all cases for all variants with independent planners.
- Aggregate 3/1/0 points, survival, score gain, submissions, and maximum p99.
- Rank with the exact deterministic tuple from the design.
- Test ties, unknown outcomes, and iteration-order independence.
- Commit comparison support.

### Task 4: Production diagnostics, report, and CLI

- Add locked `StrategyEngine.current_profile`.
- Create stable plain-JSON report conversion and `evaluation/__main__.py`.
- Keep stdout machine-readable and errors on stderr with nonzero exit status.
- Test profile access, stable report keys, and CLI success/failure.
- Commit the production adapter and CLI.

### Task 5: Acceptance, review, and local merge

- Run a real day/night/duplicate production-engine replay.
- Verify identical response digests across repeated evaluations.
- Run the full test suite, `compileall`, Python 3.11 syntax scan, and diff check.
- Request independent code review and fix all Critical/Important findings.
- Fast-forward merge the reviewed branch to local `master` and rerun the full
  suite on the merged state. Do not push remotely unless explicitly requested.
