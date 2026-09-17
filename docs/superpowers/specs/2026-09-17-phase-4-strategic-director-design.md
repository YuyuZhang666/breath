# Phase 4 Strategic Director Design

**Date:** 2026-09-17
**Status:** Approved for uninterrupted implementation
**Target runtime:** Python 3.11
**Dependencies:** standard library and existing project modules only

## 1. Purpose

Phase 4 turns the fixed Phase 2 policy and the Phase 3 robot-wave search into
profile-driven planners without weakening Phase 1 legality, serialization, or
fallback behavior. The strategic director reacts to hard facts and material
events, selects one immutable intent, and keeps expensive or uncertain tactics
behind conservative feature flags.

## 2. Scope

Phase 4 includes:

- immutable strategy profiles, build plans, day priorities, item policy, and
  rule feature flags;
- backward-compatible parameterization of Phase 2 construction and day jobs;
- a deterministic feature extractor and event-driven strategic director;
- anti-oscillation through minimum hold periods and immediate survival
  overrides;
- profile-specific resource reserves and Phase 3 `NightObjective` values;
- conservative purchase and use of the confirmed `Medicine` item;
- configurable but default-disabled bomb, stun, repair, upgrade, and offensive
  mechanics whose exact rules remain unconfirmed;
- session persistence for director state, intent, features, and transition
  reason;
- atomic fallback to the default Phase 2 intent when Phase 4 fails.

Phase 4 does not solve tasks, infer hidden opponent state, persist cross-half
memory, issue treasure summons, purchase robot summons, or perform offline
learning. Those remain Phases 5-7.

## 3. Invariants

1. Phase 1 validation and safe response remain the final authority.
2. Default `BuildPlan` and `StrategicIntent` reproduce Phase 2 behavior.
3. Tactical planners receive immutable inputs and never mutate director state.
4. Duplicate observations return the cached byte-equivalent decision.
5. Low-confidence or absent data never enables an experimental mechanic.
6. A station-loss risk overrides profile hold periods immediately.
7. Planner work remains deterministic and bounded under the existing watchdog.

## 4. Contracts

`StrategyProfile` has five values: `SURVIVE`, `ECONOMY`, `SCORE`, `PRESSURE`,
and `DESPERATION`.

`RuleFeatureFlags` contains one boolean per uncertain rule. Every field defaults
to `False`. Phase 4 only consumes the medicine behavior unconditionally because
the protocol fixture and supported action schema name `Medicine` explicitly.

`BuildPlan` owns the ordered weapon loadout, wall-site cap, entrance policy, and
whether missing weapons or walls may be built. `RulesConfig` continues to own
game mechanics such as costs and geometry; it no longer owns strategic loadout
selection.

`DayPriorities` owns recall, emergency item, weapon build, wall build, purchase,
sale, collection, scouting, and preposition weights.

`StrategicIntent` owns the selected profile, build plan, day priorities, minimum
gold reserve, allowed planner families, medicine thresholds and stock target,
the immutable `NightObjective`, and the transition reason.

## 5. Features and events

`StrategyFeatures` is derived only from the current observation and the prior
current-half session:

- our and visible enemy station health and levels;
- station health loss since the prior accepted observation;
- living personal-role, weapon, wall, and robot counts;
- targeted robot attack power and distance to our station;
- current gold, score, day, and phase;
- defensive build completeness;
- the most recent Phase 3 wave classification and secured bit.

Material events are station damage, living-role or weapon loss, phase/day
transition, a new Phase 3 certificate, defense completion change, or an
interval expiry. The director reuses the previous intent between material
events.

## 6. Profile selection

The first matching rule wins:

1. `DESPERATION` when the station is alive but visible one-turn threat can
   destroy it and the configured desperate-health threshold is met.
2. `SURVIVE` on station damage, an unsecured/marginal/unsafe certificate,
   missing required controllers, or material visible station threat.
3. `PRESSURE` only when a visible enemy station is critically weak, our defense
   is complete, no visible threat targets us, and the pressure feature is
   explicitly enabled.
4. `ECONOMY` while the configured defense or reserve is incomplete.
5. `SCORE` once survival and reserve constraints are satisfied.

Non-emergency transitions obey a minimum hold period. `SURVIVE` and
`DESPERATION` can interrupt it. Every state records a stable reason string.

## 7. Phase 2 adaptation

`plan_turn` accepts an optional `StrategicIntent`. Layout construction consumes
its `BuildPlan`; day job generation consumes its priorities and reserve; joint
validation prevents all selected actions from spending reserved gold.

Medicine use is a high-priority personal action when a living worker or pioneer
holds `Medicine` and is at or below the intent threshold. Medicine purchase is
considered only during day, adjacent to a weapon shop, while stock is below the
target and the purchase leaves the gold reserve intact.

When an emergency medicine action exists at night, Phase 3 is skipped for that
turn and the intent-aware deterministic night planner is used. This avoids
pretending the Phase 3 simulator models healing.

## 8. Strategy engine lifecycle

The locked engine performs:

```text
fingerprint and continuity
-> duplicate cache check
-> feature extraction
-> event/profile selection
-> emergency-item gate
-> Phase 3 search or intent-aware Phase 2 planner
-> immutable session update
```

Director failure is logged and replaced by `DEFAULT_STRATEGIC_INTENT`. Phase 3
failure retains its existing atomic Phase 2 fallback. No partially computed
director or search result enters the session.

## 9. Performance

Feature extraction and profile selection are linear in the bounded observation
and target less than 10 ms together. Phase 3 keeps its existing work cap. The
full response remains inside the existing five-second external limit and the
800 ms planner watchdog design target.

## 10. Verification

Tests prove:

- default intent preserves Phase 2 layout, jobs, and decisions;
- custom build plans alter only requested construction choices;
- reserves prevent joint overspending;
- hard danger selects `SURVIVE` or `DESPERATION` immediately;
- safe funded states select `SCORE`, while incomplete states select `ECONOMY`;
- hold periods stop oscillation but never suppress survival overrides;
- duplicate observations do not rerun the director or planners;
- medicine use and purchase obey inventory, health, phase, adjacency, and
  reserve constraints;
- experimental tactics remain disabled by default;
- Phase 3 objectives follow the selected intent;
- Phase 4 failure returns the deterministic Phase 2 default;
- full unit, HTTP, compilation, determinism, and latency suites pass.

## 11. Acceptance criteria

Phase 4 is complete when all five profiles and their immutable intents exist,
hard-fact selection is event-driven and stable, Phase 2 is parameterized without
changing its default behavior, confirmed medicine tactics are safe, uncertain
mechanics are default-disabled, Phase 3 receives profile objectives, and every
existing fallback and test remains valid.
