# Phase 4 Strategic Director Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an event-driven strategic director and immutable planner inputs while preserving Phase 1 safety and Phase 2/3 deterministic fallbacks.

**Architecture:** New policy contracts and a pure director feed an intent into the existing locked `StrategyEngine`. Phase 2 consumes build, priority, reserve, and medicine policy inputs; Phase 3 consumes the intent's immutable night objective and is bypassed only for emergency item use it cannot simulate.

**Tech Stack:** Python 3.11, standard library, `dataclasses`, `enum`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-4-strategic-director-design.md`

## Global Constraints

- Preserve Phase 1 validation, serialization, HTTP exception handling, and safe response behavior.
- Preserve byte-equivalent Phase 2 decisions when the default intent is used.
- Every uncertain mechanic is disabled by default.
- No production code is added before a failing behavior test.
- Full tests use `python -m unittest discover -s tests`.
- Compilation uses `python -m compileall -q main.py future_war_agent tests`.

---

### Task 1: Immutable policy contracts

**Files:**
- Create: `future_war_agent/strategy/policy.py`
- Test: `tests/test_strategy_policy.py`

**Interfaces:**
- Produces `StrategyProfile`, `RuleFeatureFlags`, `BuildPlan`, `DayPriorities`, `ItemPolicy`, `StrategicIntent`, and immutable default/profile intent factories.

- [ ] Write tests showing invalid reserves, thresholds, loadouts, wall caps, and hold periods are rejected; defaults are conservative and immutable.
- [ ] Run `python -m unittest tests.test_strategy_policy` and confirm the import fails.
- [ ] Implement the frozen contracts and validation.
- [ ] Re-run the focused tests and commit the contracts.

### Task 2: Parameterize Phase 2 without changing defaults

**Files:**
- Modify: `future_war_agent/strategy/layout.py`
- Modify: `future_war_agent/strategy/jobs.py`
- Modify: `future_war_agent/strategy/joint.py`
- Modify: `future_war_agent/strategy/night.py`
- Modify: `future_war_agent/strategy/planner.py`
- Test: `tests/test_layout.py`
- Test: `tests/test_jobs.py`
- Test: `tests/test_joint.py`
- Test: `tests/test_night.py`
- Test: `tests/test_strategy_planner.py`

**Interfaces:**
- Consumes `StrategicIntent.build_plan`, `day_priorities`, `gold_reserve`, and `item_policy`.
- Produces `plan_turn(observation, rules=DEFAULT_RULES, intent=DEFAULT_STRATEGIC_INTENT)`.

- [ ] Add tests proving default decisions are unchanged, custom loadouts and wall caps are honored, and gold reserves constrain complete legal joints.
- [ ] Run the focused suites and confirm failures identify missing intent parameters.
- [ ] Thread intent through layout, jobs, joint enumeration, night candidates, and planner composition.
- [ ] Add medicine-use and medicine-purchase tests with literal expected actions; run them red.
- [ ] Implement conservative medicine candidates and direct legality checks.
- [ ] Re-run all Phase 2 focused suites and commit the compatibility layer.

### Task 3: Feature extraction and event-driven director

**Files:**
- Create: `future_war_agent/strategy/features.py`
- Create: `future_war_agent/strategy/director.py`
- Test: `tests/test_strategy_features.py`
- Test: `tests/test_strategy_director.py`

**Interfaces:**
- Produces `StrategyFeatures`, `DirectorState`, `DirectorDecision`, `extract_features`, and `StrategicDirector.select`.
- Consumes current observation, previous observation/features/state, and the most recent robot-wave certificate.

- [ ] Write feature tests for station deltas, threats, defense completeness, and certificate projection; run them red.
- [ ] Implement pure feature extraction and run the tests green.
- [ ] Write director tests for all profiles, material-event reuse, hold periods, and immediate survival override; run them red.
- [ ] Implement deterministic profile selection and intent factories.
- [ ] Re-run both focused suites and commit the director.

### Task 4: Locked engine integration and atomic fallback

**Files:**
- Modify: `future_war_agent/strategy/session.py`
- Modify: `future_war_agent/strategy/engine.py`
- Test: `tests/test_strategy_session.py`
- Test: `tests/test_strategy_engine.py`
- Test: `tests/test_controller.py`
- Test: `tests/test_server.py`

**Interfaces:**
- `StrategySession` stores features, director state, and selected intent.
- `StrategyEngine` invokes director under its lock, passes intent to Phase 2 and Phase 3, and caches the complete result.

- [ ] Add tests for session state, intent-aware Phase 2 calls, profile objective injection, emergency medicine Phase 3 bypass, duplicate caching, and director exception fallback; run them red.
- [ ] Implement engine/session integration without changing duplicate or discontinuity semantics.
- [ ] Run the focused engine, controller, and HTTP tests green.
- [ ] Commit production integration.

### Task 5: Phase 4 acceptance

**Files:**
- Create: `tests/test_phase4_acceptance.py`
- Modify: `docs/superpowers/specs/2026-09-17-phase-1-foundation-design.md`
- Modify: `docs/superpowers/specs/2026-09-17-phase-2-deterministic-policy-design.md`

**Interfaces:**
- Acceptance tests exercise real `StrategyEngine`, validator, serializer, and HTTP boundaries.

- [ ] Add deterministic replay tests for economy-to-score transition, danger override, default-disabled experimental flags, and medicine behavior; observe failures before final integration fixes.
- [ ] Add concise documentation notes clarifying process-level strategy state and Phase 2's fallback-default role.
- [ ] Run focused acceptance, full unittest discovery, compileall, `git diff --check`, and a placeholder/syntax scan.
- [ ] Request independent code review, fix all Critical and Important findings with regression tests, and repeat verification.
- [ ] Commit final review fixes and merge only after the merged `master` suite passes.
