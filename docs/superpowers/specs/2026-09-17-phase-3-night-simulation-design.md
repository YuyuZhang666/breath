# Phase 3 Night Simulation Design

**Date:** 2026-09-17

**Status:** Approved design, pending implementation plan

**Target runtime:** Python 3.11
**Dependency policy:** Standard library only unless a later requirement makes one of the approved runtime packages necessary

## 1. Purpose

Phase 3 adds a deterministic short-horizon simulator, survival prover, and night joint-action optimizer on top of the Phase 2 deterministic policy.

The new layer answers one narrow question: given the currently visible robots that are targeting our team, which legal night action is most robust over the next few turns?

Phase 3 is safety-gated. It may replace the Phase 2 night decision only when the current state is supported, observation history is continuous, the simulation completes inside its fixed work budget, and all produced actions pass the existing legality checks. Otherwise, the entire Phase 3 result is discarded and Phase 2 is used unchanged.

## 2. Goals

1. Simulate the supported night combat rules deterministically for four plausible robot-behavior scenarios.
2. Generate level-aware Gatling, railgun, and rocket attack candidates from the request data.
3. Reuse the Phase 2 joint-action legality machinery rather than create a second legality system.
4. Evaluate at most 64 root joint actions over a horizon of up to six turns.
5. Produce a survival certificate and a stable, lexicographically ranked action.
6. Learn only short-lived, process-local robot-model weights from consecutive observations.
7. Preserve deterministic output, bounded work, and full fallback to Phase 2.

## 3. Non-goals

Phase 3 does not:

- predict hidden robots or the next night's wave;
- predict or choose pulse summons;
- use bombs, stun items, repairs, or upgrades;
- persist learning to disk or share it across matches, processes, or halves;
- use MCTS, beam search over every future turn, neural inference, or PSRO;
- change the HTTP request or response schema;
- make daytime decisions differently from Phase 2;
- infer unknown rule fields or return partially evaluated actions after a timeout.

Task-agent evolution, enemy belief, pulse-summon planning, item tactics, and cross-match learning remain later phases.

## 4. High-level architecture

```text
Observation
    |
    v
StrategyEngine
    |-- SessionStore (one short-lived session per teamId)
    |-- ObservationReconciler
    |-- Phase 2 planner (always available)
    |
    `-- supported, continuous night state
            |
            v
        NightCandidateGenerator
            |
            v
        Phase 2 legal joint enumeration
            |
            v
        ScenarioGenerator (four robot models)
            |
            v
        Deterministic rollout (up to six turns)
            |
            v
        SurvivalCertificate + robust score
            |
            v
        selected legal action
```

The existing stateless `plan_turn()` remains the Phase 2 baseline. A new process-level `StrategyEngine` owns session state and delegates to that baseline for day turns and every fallback case. Tests instantiate fresh engines to avoid state leakage.

The production controller holds one engine instance. The complete observe/reconcile/plan/cache operation is protected by a lock because the HTTP server may dispatch concurrent requests.

## 5. Session and observation lifecycle

### 5.1 Session key and contents

Sessions are keyed by `teamId` and contain only:

- the last accepted observation summary;
- the last round and phase;
- the last selected action;
- an observation fingerprint and cached response;
- four robot-model weights;
- the map and station identity signature needed to detect a reset.

No session data is written to disk.

### 5.2 Duplicate and revised observations

- The same round with the same canonical observation fingerprint returns the cached response.
- It does not update model weights or advance cooldown projections a second time.
- The same round with a different fingerprint is replanned from the revised observation, but is not treated as a new transition for model calibration.
- Canonical fingerprints use stable field ordering and only protocol data relevant to planning.

### 5.3 Continuity and reset rules

A session is reset when:

- the observed round moves backwards;
- `teamId` changes;
- the map dimensions or static map signature changes;
- the controlled station identity or position changes incompatibly.

A gap of more than one round is treated as discontinuous evidence. Calibration returns to uniform weights, and the first unsupported/discontinuous night request uses Phase 2. Phase 3 becomes eligible after a subsequent consecutive observation. This also handles a process starting midway through a night. In the normal match flow, daytime observations establish continuity before the first night turn.

## 6. Phase 3 eligibility gate

Phase 3 runs only when all of the following hold:

1. The request is a night turn.
2. A preceding, consecutive observation exists.
3. Every controlled role, relevant building, weapon, and threatening robot has the fields required by the simulator.
4. Every simulated weapon is one of Gatling, railgun, or rocket.
5. Visible threatening robots have an unambiguous `targetTeam` equal to `observation.our.team_type`. The protocol's team type identifies combat allegiance; `teamId` is used only as the session key.
6. Positions are valid integer grid cells and the current observation is internally consistent.
7. The fixed candidate and rollout budgets can be applied.

Only currently visible, alive robots whose `targetTeam` points to us are simulated. Other robots are outside the Phase 3 threat set; hidden robots and future spawns are never invented.

Failure of any gate condition calls Phase 2 without attempting a partial Phase 3 decision.

### 6.1 Required-field presence

The current parser intentionally supplies backward-compatible defaults for absent optional integer fields. In particular, a missing weapon `cooldown` currently becomes zero, which Phase 3 must not mistake for an explicitly ready weapon.

`UnitState` therefore gains parser-populated combat-field presence metadata, with a backward-compatible empty default for direct constructors. Phase 3 requires every simulated weapon to have explicitly supplied `attackPower`, `attackRange`, `level`, and `cooldown` fields. The values must also satisfy the normal weapon invariants. Phase 1 and Phase 2 continue to see their existing parsed values and behavior; only the Phase 3 eligibility gate consults the presence metadata.

## 7. Simulation state and turn kernel

`SimState` is separate from the protocol `Observation`. It is an immutable or copy-on-write planning value expressed in integer grid cells. It contains:

- current round and phase;
- controlled roles and their positions/health;
- station, wall, and weapon positions/health;
- weapon type, level, attack power, attack range, and cooldown;
- visible threatening robots, including type, health, position, and target team;
- a pending damage ledger;
- projected structure losses and remaining threat.

One simulated turn uses this fixed order:

1. Resolve selected weapon attacks into the pending damage ledger.
2. Resolve controlled-role movement and collision according to existing legal movement rules.
3. Let each surviving robot choose an action under the active scenario model.
4. Add robot attack damage to the pending ledger.
5. Apply all pending damage simultaneously.
6. Remove dead units and structures.
7. Advance cooldowns and the simulated round.

Weapon fire therefore occurs before robot movement, while all damage is committed together at the end of the turn.

The simulator uses request values such as `attackPower`, `attackRange`, `level`, and `cooldown` as authoritative. Known static robot constants belong in the central rules configuration rather than being scattered through the simulator.

## 8. Weapon candidate generation and damage rules

Each controlled weapon contributes no more than three attack candidates plus a wait candidate. Candidate sorting and truncation use stable keys so identical requests always produce identical candidate sets.

The complete legal root joint-action list is capped at 64 after combining weapon decisions and the Phase 2 role/controller movement assignments. The cap applies to complete joint actions, not merely to raw target tuples.

### 8.1 Gatling

- Available only at night and only when controlled by an adjacent role.
- The number of target cells equals the current weapon level.
- Target cells must be in range and form one 90-degree cone: every pair of shot directions differs by at most 90 degrees.
- Each target defines a ray.
- Each bullet damages only the nearest live robot intersected by its ray.
- Damage uses the request's `attackPower`.
- If a legal level-sized target tuple cannot be formed, no attack candidate is emitted for that weapon.

### 8.2 Railgun

- Uses exactly one target cell in range.
- The request's `attackPower` is the initial penetrating energy.
- Live robots are processed in ray order.
- Each robot takes `min(remaining_energy, robot_health)` damage and that amount is removed from the remaining energy.
- Resolution stops when no energy remains.

### 8.3 Rocket

- May fire only when cooldown is zero.
- The number of target cells equals the current weapon level.
- Each center cell receives full request `attackPower` damage.
- Adjacent cells receive half damage according to the integer rounding rule in the central rules configuration.
- Overlapping blast areas stack.
- Firing starts the configured three-round cooldown; later simulated turns decrement it deterministically.
- If a legal level-sized target tuple cannot be formed, no attack candidate is emitted.

### 8.4 Ray discretization uncertainty

The protocol rules do not fully identify how a continuous ray maps to grid cells. Phase 3 implements three deterministic interpretations: Bresenham, supercover, and cell-center intersection.

The four-scenario rollout budget remains unchanged. Ray interpretations do not create twelve rollout states. Instead, every ray-based attack is resolved under all three interpretations inside the weapon kernel before the scenario state advances:

- damage credited to each robot is the minimum damage produced for that robot by the three interpretations;
- the resulting conservative damage ledger is the only ledger applied to the scenario state;
- median geometric damage may be used only while stably pre-sorting raw candidates before truncation, never in the survival certificate;
- an attack that cannot be represented legally under every required interpretation is not generated.

This makes geometry ambiguity explicit without allowing it to multiply the search tree or runtime unpredictably.

## 9. Legal joint-action reuse

The existing Phase 2 night legality code is extended rather than bypassed:

- railgun attacks require exactly one target;
- Gatling and rocket attacks require exactly `weapon.level` targets;
- Gatling candidates must satisfy the cone rule;
- role-to-weapon control remains one adjacent role per weapon and one weapon per role;
- conflicting role movement and occupation remain subject to the Phase 2 joint solver;
- every selected Phase 3 response still passes the outer Phase 1 response validator.

Phase 2's original level-one/fallback behavior remains unchanged when Phase 3 is ineligible.

## 10. Robot behavior scenarios

The scenario generator creates exactly four deterministic robot policies:

1. **Station shortest path:** move along a stable shortest path to the station; attack when the path is blocked by an attackable structure.
2. **Main-path blocker:** prefer destroying our building that blocks the currently best route to the station.
3. **Low-health reachable structure:** prefer an attackable, reachable controlled structure with the lowest health, using distance and stable identity as tie-breakers.
4. **Maximum station progress:** choose the legal action that maximizes the reduction in path distance to the station, again with stable tie-breakers.

All path ties use a fixed coordinate/direction order. Scenario policies contain no randomness.

Initial scenario weights are equal. The observation reconciler compares each model's one-step prediction with the next real observation using:

- robot position error;
- alive/dead mismatch;
- controlled-building health deltas.

The comparison produces deterministic non-negative loss values. Weights are updated by a fixed likelihood transform, clamped so every model retains at least a small probability, then normalized. The minimum retained probability is 5 percent. With no usable evidence, weights remain unchanged.

Calibration affects scoring only; it never changes the rule kernel or legality checks.

## 11. Search and future policy

Search is bounded by:

```text
at most 64 root joint actions
    x exactly 4 robot scenarios
    x min(6, remaining night turns) rollout steps
```

The desired horizon is four to six turns. Near the end of a night, fewer than four turns may remain, so the horizon is limited to the actual remaining turns.

Branching happens only at the root. After the candidate root action, every rollout follows a fixed future policy:

- roles continue toward their already assigned legal weapon-control cells;
- a ready weapon chooses the stable attack candidate with the greatest immediate threat reduction;
- unavailable weapons wait;
- robot behavior follows the scenario's fixed policy.

This future policy is not returned to the server. It exists only to value the root action, including rocket cooldown consequences, without an exponentially growing tree.

Candidate generation, scenario order, pathfinding ties, and score ties all use stable ordering. Wall-clock time never decides which partially explored action wins.

## 12. Survival certificate and ranking

Each root action produces a `SurvivalCertificate` containing:

- classification: `SAFE`, `MARGINAL`, `UNSAFE`, or `UNKNOWN`;
- weighted station-survival probability;
- expected station health;
- lower-tail (`p10`) station health;
- worst-case station health;
- projected wall and weapon losses;
- surviving key-weapon count;
- remaining robot threat;
- scenario-level diagnostics for tests and internal logging.

Classification is deterministic:

- `SAFE`: the station survives every valid scenario;
- `MARGINAL`: the station survives at least one but not every valid scenario;
- `UNSAFE`: the station is lost in every valid scenario;
- `UNKNOWN`: no valid certificate can be produced because the state or computation is unsupported.

Root actions are ranked lexicographically by:

1. weighted station-survival probability;
2. `p10` station health;
3. worst-case station health;
4. surviving key-weapon count;
5. expected station health;
6. lower remaining robot threat;
7. greater immediate kill value;
8. stable serialized action ordering.

If every evaluated action is `UNSAFE`, Phase 3 still returns the least-bad fully evaluated action using the same ordering. Phase 2 fallback is reserved for unsupported input, discontinuity, internal failure, or watchdog/budget failure; it is not triggered merely because the position is losing.

## 13. Work budget and watchdog

The fixed scenario, candidate, and horizon caps are the primary runtime controls. A hard watchdog of approximately 800 milliseconds protects the service from unexpected pathological work.

If the watchdog is reached:

- evaluation stops;
- all partial Phase 3 scores are discarded;
- Phase 2 plans again from the original observation;
- no partially best action is returned.

The watchdog is a failure boundary, not a normal search termination condition. Tests use an injected clock/deadline abstraction so timeout behavior is deterministic and does not require sleeping.

## 14. Error handling and fallback boundary

Phase 3 catches expected parsing, support, geometry, simulation, and deadline failures at its integration boundary. Each such failure records an internal diagnostic reason and delegates to Phase 2.

The outward request/response protocol is unchanged. Internal survival certificates and fallback reasons are not added to the server response.

If Phase 2 also fails or produces an invalid response, the existing Phase 1 validation and safe-action boundary remains responsible for the final fallback. Phase 3 does not weaken that boundary.

## 15. Proposed module boundaries

The implementation plan may refine filenames, but responsibilities should remain separated:

- `protocol/models.py` and `protocol/parser.py`: preserve explicit presence of weapon combat fields without changing Phase 1/2 default-value behavior.
- `strategy/engine.py`: stateful `StrategyEngine`, locking, cache, eligibility, and fallback.
- `strategy/session.py`: per-team session records, fingerprints, continuity, and reset rules.
- `strategy/reconcile.py`: one-step scenario prediction comparison and weight updates.
- `strategy/simulation/state.py`: protocol-to-`SimState` conversion and immutable state values.
- `strategy/simulation/geometry.py`: ray and cone geometry.
- `strategy/simulation/weapons.py`: candidate generation and weapon damage kernels.
- `strategy/simulation/robots.py`: four deterministic robot policies.
- `strategy/simulation/kernel.py`: turn ordering and state transitions.
- `strategy/simulation/certificate.py`: rollout metrics, classification, and ranking.
- `strategy/simulation/search.py`: fixed-budget root enumeration and rollout orchestration.

The Phase 2 joint solver should expose a reusable legal-joint enumeration interface while retaining its current public behavior.

## 16. Verification strategy

### 16.1 Geometry and weapons

- Gatling level-sized target count, range, 90-degree cone, and nearest-hit behavior.
- Railgun ray order, penetration, exact energy depletion, and zero-energy stop.
- Rocket target count, center/adjacent damage, overlap stacking, cooldown start/decrement, and not-ready behavior.
- Bresenham, supercover, and cell-center ray edge cases.
- Conservative geometry folding.

### 16.2 Turn kernel

- Weapon attacks occur before robot movement.
- Damage is applied simultaneously at turn end.
- Dead units cannot act on later turns.
- Movement and collision follow existing rules.
- Cooldowns advance exactly once per simulated turn.

### 16.3 Scenarios and reconciliation

- Every robot model makes stable choices under ties.
- One-step predictions update weights toward better-fitting models.
- Weight floors and normalization hold.
- Missing evidence leaves weights unchanged.
- Duplicate requests do not calibrate twice.
- Round gaps, rollback, map changes, and station changes reset or suspend continuity correctly.

### 16.4 Search and certificates

- No more than 64 root actions, four scenarios, and six turns are evaluated.
- Ranking follows the specified lexicographic order.
- `SAFE`, `MARGINAL`, `UNSAFE`, and `UNKNOWN` classifications are correct.
- An all-unsafe position selects the deterministic least-bad action.
- Rocket cooldown can change the preferred root action.
- Stable request data produces byte-equivalent responses.

### 16.5 Integration and fallback

- Phase 3 multi-target actions pass the Phase 1 validator.
- Missing fields, unsupported weapons, discontinuity, simulator exceptions, and watchdog expiry all produce a fresh Phase 2 result.
- Parser-presence metadata distinguishes an explicitly supplied zero cooldown from a missing cooldown.
- Robot `targetTeam` is compared with `our.team_type`, while sessions remain isolated by `our.team_id`.
- A timed-out partial result is never returned.
- Day requests still use Phase 2 while recording continuity.
- Different `teamId` sessions do not contaminate each other.
- Concurrent requests remain deterministic and race-free.
- All existing Phase 1 and Phase 2 tests continue to pass.
- An end-to-end HTTP probe verifies a supported night request, duplicate request caching, and a forced fallback.

## 17. Acceptance criteria

Phase 3 is complete when:

1. All supported night decisions pass through the bounded simulator and certificate ranking.
2. Unsupported or failed simulations return the Phase 2 result with no protocol difference.
3. The same observation and session history always produce the same action.
4. Root actions, scenario count, and horizon never exceed their specified caps.
5. Session reset, duplicate, and concurrency behavior is covered by tests.
6. Weapon behavior is covered for all visible supported levels using request-provided combat values.
7. Existing tests plus the new Phase 3 suite pass under Python 3.11.
8. Source compilation succeeds and the HTTP deterministic/fallback probes pass.
