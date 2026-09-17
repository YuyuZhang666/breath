# Phase 6 Opponent Belief and Match Memory Implementation Plan

**Goal:** Add bounded opponent beliefs and cross-half process memory without
weakening deterministic fallbacks or request latency.

## Engineering rules

- Implement every behavior test-first.
- Use immutable records and deterministic stable ordering.
- Keep memory process-local, team-scoped, locked by `StrategyEngine`, and
  bounded independently of match duration.
- Use Python 3.11-compatible standard-library code only.

### Task 1: Opponent belief

- Define immutable tracks and belief configuration.
- Add visible update, deterministic decay, contradiction replacement, and a
  64-track cap.
- Test confidence, bounds, stable ordering, and unknown role safety.

### Task 2: Normalized match memory

- Define canonical orientation and normalized static zone facts.
- Add bounded wave summaries and immutable `MatchMemory` updates.
- Test mirrored coordinates, missing stations, cross-half reuse, and caps.

### Task 3: Independent memory store

- Add a team-keyed `MatchMemoryStore` separate from `SessionStore`.
- Preserve memory across round gaps, rollbacks, map/station signature changes,
  and short-lived session resets.
- Keep duplicate observations mutation-free.

### Task 4: Locked engine integration

- Inject/update the memory store inside the existing engine lock.
- Record visible evidence before planning and fresh certificates after search.
- Make memory failures non-fatal and expose read-only memory for tests/future
  policy consumers.

### Task 5: Acceptance, review, and merge

- Run real-engine side-swap and discontinuity tests.
- Verify complete tests, compileall, diff hygiene, deterministic duplicates,
  and no disk files.
- Obtain independent review, fix all Critical/Important findings, fast-forward
  merge to local `master`, and retest the merged state.
