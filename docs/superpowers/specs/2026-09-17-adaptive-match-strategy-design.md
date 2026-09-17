# Adaptive Match Strategy Design

**Date:** 2026-09-17
**Status:** Approved direction, written design pending user review
**Target runtime:** Python 3.11
**Dependency policy:** Standard library first; approved runtime packages are used only when a later phase has a demonstrated need

## 1. Purpose

This document defines the match-level architecture for maximizing the chance of winning the two-half game and, ultimately, league points. It consolidates the two supplied strategy analyses while separating confirmed rules from hypotheses that require organizer confirmation or live evidence.

The strategy is not a fixed "turtle" script and not an unconstrained online learner. It is a layered adaptive system:

- immutable protocol, legality, deadline, and fallback safeguards;
- deterministic tactical planners for movement, construction, and night combat;
- an event-driven strategic director that selects a pre-tested profile;
- bounded opponent beliefs and match memory;
- offline self-play and replay used to improve the profile library, never placed on the response-critical path.

## 2. Match objective

The primary objective is match win probability, not raw score and not station health in isolation.

The official outcome rules imply:

1. If stations are destroyed in different rounds, the station destroyed first loses the half.
2. If both stations are destroyed in the same round, score decides unless scores are equal.
3. If neither station is destroyed, score decides.
4. If each team wins one half, combined score across both halves decides the match.
5. League results award 3/1/0 points for win/draw/loss.

Station survival is therefore a hard constraint while immediate destruction risk exists, but task score, robot kill ownership, and the two-half aggregate become decisive once survival is sufficiently secured.

The strategic objective is expressed as:

```text
maximize expected match points
subject to a configured upper bound on first-station-loss risk
and invariant legality, deadline, and fallback constraints
```

## 3. Confirmed strategic facts

The implementation may rely on these documented facts:

- the map is reused for both halves and the teams switch sides;
- our units share vision, enemy stations and walls are globally visible, and other enemy units require vision;
- robots are globally visible;
- station survival score totals 550 if the station survives every day;
- task score rewards speed and awards partial credit by pass rate;
- remaining robots are removed on the first turn of the next day;
- summon orders add robots to the opponent's next night and at most ten may be used per day;
- upgrades restore the upgraded building to full health;
- build actions are daytime-only, while supported item use is not generally phase-limited;
- each role has one action per round, so movement, item use, task work, and weapon control compete for the same role action;
- a failed execution of a structurally valid command is skipped but does not count as a team exception.

## 4. Unconfirmed mechanics and feature flags

The following claims must not become unconditional production logic:

- whether rocket damage can affect enemy roles, weapons, walls, or stations;
- whether build regions are team-exclusive and whether enemy construction can occupy or overwrite them;
- whether an existing weapon can be overwritten directly by a new build;
- whether repeated stun effects stack or refresh;
- whether abandoning a task guarantees a different task after refresh;
- whether a failed treasure summon consumes its items;
- the exact mapping of a continuous weapon ray to grid cells where the rules remain ambiguous.

Each uncertain mechanic is represented by a named `RuleFeatureFlags` value. Defaults use only the conservative documented behavior. A flag may be enabled only by organizer confirmation, a controlled integration probe, or unambiguous observed evidence. Tactical safety must never depend on an experimental flag.

## 5. Layered architecture

```text
Raw request
    |
    v
Phase 1 protocol + legality + safe response
    |
    v
StrategyEngine (locked observe/reconcile/plan/cache lifecycle)
    |
    +--> StrategySession: current-half continuity and tactical cache
    +--> MatchMemory: bounded map/opponent knowledge across halves
    |
    v
FeatureExtractor + EventDetector
    |
    v
OpponentBelief + ThreatProfile
    |
    v
StrategicDirector selects a pre-tested StrategyProfile
    |
    +--> Phase 2 day/spatial/joint-action foundation
    +--> Phase 3 robot-wave night simulator
    +--> later economy, item, task, opponent-pressure, and summon planners
    |
    v
Phase 1 validation + serialization + safe fallback
```

The tactical planners remain deterministic functions of an observation, an immutable profile, and bounded session evidence. They do not own opponent learning or mutate global strategy.

## 6. Strategy profiles

The strategic director selects one of five profiles:

### 6.1 `SURVIVE`

Used when the station, a necessary controller, or a key weapon has material loss risk. It prioritizes worst-case survival, emergency positioning, repair/upgrade value, and threat removal. Score efficiency is a tie-breaker only.

### 6.2 `ECONOMY`

Used when the next modeled defense window is secure but the planned construction, upgrade, item, or task budget is underfunded. It prioritizes mining, selling, news-aware inventory timing, and resource reservation without violating twilight recall.

### 6.3 `SCORE`

Used after survival constraints are satisfied. It prioritizes task speed, partial-answer deadlines, robot kill ownership, treasure opportunities, and low-risk score denial.

### 6.4 `PRESSURE`

Used only when a visible or high-confidence opponent weakness produces positive expected match value after accounting for the score donated by summoned robots and the cost of interrupting our own defense/economy.

### 6.5 `DESPERATION`

Used when conservative play has a very low chance to save the half or match. It deliberately accepts higher variance through aggressive summons, score races, or experimentally enabled attacks. Legality, deadline, and process-safety invariants still apply.

Profiles contain priorities, budgets, safety thresholds, and planner options. They do not contain executable callbacks or mutable state.

## 7. Adaptation cadence

Adaptation is event-driven rather than a full strategic replan every round.

### 7.1 Every round

- parse and validate current facts;
- update visible robot threat and immediate survival risk;
- react to cooldowns, deaths, damage, movement conflicts, and legal kill opportunities;
- run the bounded tactical planner under the currently selected profile.

### 7.2 On material events or a small fixed interval

- update opponent belief confidence;
- reassess score-versus-survival mode;
- detect wave anomalies against the learned natural-wave distribution;
- reconsider emergency reserves, scouting, and task interruption.

### 7.3 At commitment boundaries

Construction, weapon composition, major upgrades, large summon purchases, and other sunk-cost decisions are reconsidered only at meaningful windows such as day start, phase transition, task completion, or high-confidence opponent change.

This cadence prevents strategy oscillation and preserves response time.

## 8. Opponent belief

Opponent state is a collection of observations with confidence and last-seen time, not a permanently trusted snapshot.

The initial belief features include:

- global station health/level and wall geometry when exposed by the request;
- visible weapon type, level, position, cooldown, and adjacent controllers;
- visible role locations and behavior;
- changes in wall geometry over time;
- natural-wave baselines and statistically unusual next-night additions;
- task, economy, or pressure behavior that can be inferred without inventing hidden facts.

Beliefs decay when the underlying unit leaves vision. A later contradictory observation replaces the old claim. Opponent weapon positions are never treated as permanent because weapons may be destroyed or rebuilt.

The first implementation uses deterministic confidence rules. Statistical or learned models may replace their internals later without changing the tactical planner interfaces.

## 9. Anti-oscillation and deception resistance

A profile transition requires all of:

1. evidence above the transition's confidence threshold;
2. expected benefit greater than travel, inventory, sunk-gold, and task-interruption costs;
3. no active minimum-hold period, unless an immediate station-loss override applies.

Hard facts such as current health or visible units can trigger immediately. Hidden-state inferences require repeated evidence. A profile switch records its reason and supporting evidence for replay tests.

## 10. State and memory boundaries

### 10.1 `StrategySession`

`StrategySession` is process-local and current-half scoped. It owns duplicate-request caching, round continuity, the last accepted observation, the selected action, and short-horizon model calibration. It is protected by the strategy engine lock.

### 10.2 `MatchMemory`

`MatchMemory` is logically distinct from the session and may survive a half transition. It stores only bounded, versioned summaries:

- side-normalized static map information;
- observed mine, task, shop, and treasure-related facts;
- natural-wave summaries;
- opponent observations with confidence and last-seen time;
- strategy outcomes needed for the second half.

Coordinates are normalized to an own-station-relative orientation so the same policy works after sides switch. Disk persistence is introduced only after process-lifecycle behavior is confirmed. If later enabled, snapshots must be atomic, versioned, size-bounded, and optional; corrupt or missing memory always falls back to an empty memory.

## 11. Performance and failure budget

The service has a five-second response limit, but normal planning targets a substantially smaller budget:

```text
parse and validate                 target < 20 ms
feature and belief update          target < 30 ms
profile selection                  target < 10 ms
tactical simulation/search         target < 600 ms
final validation/serialization     target < 40 ms
reserved watchdog margin           at least 100 ms within an 800 ms planner budget
```

These are design targets, not measured guarantees. Phase 3 benchmarks establish the first real baseline.

Every expensive planner is an atomic anytime boundary: it either completes its declared deterministic work set or discards the partial result and invokes the lower-level fallback. Wall-clock arrival order never selects between partially explored actions.

Belief updates and profile selection must be linear in the bounded observation/memory size. Offline self-play, model fitting, and large searches never execute in the HTTP request path.

## 12. Phase 1 and Phase 2 impact

Phase 1 remains the protocol and safety foundation. Parser, action types, validation, serialization, HTTP exception handling, and safe response semantics are retained. The default controller integration later delegates to the locked `StrategyEngine`; protocol code does not become strategy-aware.

Phase 2 retains its world grid, pathfinding, defensive geometry, candidate generation, and joint-action solver. Two policy choices are later extracted from static configuration:

- weapon composition becomes a `BuildPlan` selected by the strategic director rather than a game-rule constant;
- day-job priorities and spending permission become `StrategicIntent` inputs rather than universal ordering.

The existing mixed loadout and day priorities remain deterministic fallback defaults. No unverified strategy document directly replaces them.

## 13. Phase roadmap

1. **Phase 3 — Robot-wave night simulation:** correct combat kernel, weapon geometry, short-lived scenario calibration, bounded search, and a robot-wave-scoped certificate.
2. **Phase 4 — Strategic Director:** hard-fact event detection, profile selection, build/economy reserves, upgrades, repairs, medicine, bombs, and stun tactics.
3. **Phase 5 — Task Agent:** task valuation, SOP reuse, sandbox interaction, deadline-aware partial submission, news, and treasure reasoning.
4. **Phase 6 — Opponent belief and match memory:** confidence decay, scouting, counter-pressure, side-normalized cross-half summaries, and confirmed counter-battery mechanics.
5. **Phase 7 — Summon and self-play optimization:** opponent-aware summon timing, replay evaluation, profile calibration, and offline self-play/PSRO experiments.

Each phase gets its own design, implementation plan, tests, and acceptance gate. Later phases compose Phase 2/3 interfaces rather than bypassing their legality and fallback boundaries.

## 14. Verification strategy

The adaptive architecture is verified with deterministic replays rather than only isolated unit tests:

- identical observations and memory produce byte-equivalent responses;
- noisy or low-confidence observations do not cause profile oscillation;
- hard station-loss evidence overrides a hold period immediately;
- stale enemy weapon knowledge decays and is invalidated by contradictory evidence;
- switching profiles accounts for travel and sunk-cost penalties;
- side-normalized facts map correctly across both halves;
- corrupt, absent, or unsupported memory falls back safely;
- fixed computation budgets and watchdog behavior hold under adversarial observations;
- each experimental rule flag is disabled by default and has isolated tests;
- Phase 1 safe responses and all existing Phase 2 behavior continue to pass.

## 15. Acceptance criteria

This architecture is successfully landed when:

1. Phase 3 exposes deterministic robot-wave metrics without claiming total match safety.
2. Strategy profiles are immutable inputs to planners, and tactical code contains no opponent-learning side effects.
3. Strategy session and match memory have separate lifecycles and fallback semantics.
4. Unconfirmed mechanics are isolated behind disabled feature flags.
5. Response-critical work has explicit deterministic caps and measured latency tests.
6. Phase 1/2 remain valid fallbacks while later phases incrementally add adaptive behavior.
