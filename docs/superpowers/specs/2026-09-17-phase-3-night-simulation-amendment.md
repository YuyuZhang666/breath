# Phase 3 Night Simulation Design Amendment

**Date:** 2026-09-17
**Status:** Written amendment pending user review
**Amends:** [Phase 3 Night Simulation Design](./2026-09-17-phase-3-night-simulation-design.md)
**Related architecture:** [Adaptive Match Strategy Design](./2026-09-17-adaptive-match-strategy-design.md)

## 1. Authority and intent

This amendment incorporates the subsequent official-rule review and the approved adaptive-strategy direction. Where it conflicts with the original Phase 3 design, this amendment is authoritative. Original sections not changed here remain valid.

The existing Phase 3 implementation plan predates this amendment and is stale. It must be revised with the writing-plans workflow after the user approves this document.

Phase 3 remains intentionally narrow: it evaluates currently visible robots whose `targetTeam` points to us. It does not implement economy, tasks, item usage, enemy weapon attacks, summon purchasing, opponent archetype selection, or cross-half learning.

## 2. Safety claim and objective input

Rename the Phase 3 certificate to `RobotWaveSafetyCertificate`. Its classifications are:

- `WAVE_SAFE`: the station survives every valid visible-robot scenario;
- `WAVE_MARGINAL`: the station survives at least one but not every valid scenario;
- `WAVE_UNSAFE`: the station is lost in every valid scenario;
- `UNKNOWN`: no valid certificate can be produced.

No classification claims safety from enemy fire, hidden roles, hidden weapons, future natural waves, or future summons.

The night searcher accepts an immutable `NightObjective`. Phase 3 provides one deterministic default. The objective may specify safety-reserve and score-mode preferences, but it cannot change legality, geometry, the turn kernel, scenario transitions, work caps, or fallback behavior. A later strategic director may select among pre-tested objectives without entering the simulator.

## 3. Simulation state corrections

`SimState` contains:

- controlled roles with ID, type, position, health, and controller assignment;
- station, walls, and weapons with ID, type, position, health, level, and current cooldown where applicable;
- visible threatening robots with ID, type, position, health, `abnormalState`, and target team;
- official robot combat specifications derived from robot type;
- pending damage, projected losses, kill ownership, and remaining threat;
- current round, remaining night turns, and immutable objective.

Robot combat values are centralized. The documented small, medium, large, and boss attack powers apply, and their documented attack range is three. Unknown robot types make the state unsupported rather than receiving invented values.

An observed dizzy robot waits for the current simulated turn. Because the protocol does not expose remaining stun duration, it is treated as active from the next simulated turn. This is conservative for our survival.

## 4. Corrected turn kernel

The original kernel incorrectly serialized controlled-role movement before robot intent. The corrected kernel is:

1. Snapshot the start-of-turn state.
2. Resolve selected weapon attacks into a pending damage ledger. Damage is not committed yet, so a robot killed by pending weapon damage may still act this turn.
3. From the same pre-movement snapshot, form controlled-role movement intents and each robot's move-or-attack intent under the active scenario.
4. Resolve controlled-role and robot movement together using the documented destination-block, destination-contest, and position-swap collision rules.
5. For robots that selected attack, add damage against the blocking attackable role or building to the pending ledger. A robot may attack a blocking role or building within its attack range; adjacency is not required.
6. Apply all pending damage simultaneously at turn end.
7. Remove dead roles, robots, and structures. A dead controller cannot control a weapon on later simulated turns.
8. Advance weapon cooldowns and the simulated round exactly once.

Weapons attack before robot movement, but all attack damage is committed at the end of the turn. Scenario intent and collision logic must therefore be deterministic and independent of iteration order.

When a rollout reaches the end of the night, remaining robots produce no later survival damage because the rules remove them on the first turn of the next day. They also produce no unearned kill score.

## 5. Weapon semantics corrections

The request remains authoritative for current level, attack range, and cooldown. Documented weapon-specific damage semantics are authoritative:

### 5.1 Gatling

- target count equals level;
- targets must be in range and within one legal 90-degree cone;
- each target defines one bullet ray;
- the bullet hits the nearest live robot on that ray;
- each bullet deals the documented fixed 10 damage;
- current request `attackPower`, when present, is checked for compatibility rather than used to invent a different Gatling rule.

### 5.2 Railgun

- exactly one target is supplied;
- request `attackPower` represents current penetrating energy;
- robots are processed in ray order;
- each receives `min(remaining_energy, current_health)` and consumes that energy;
- processing stops when energy is exhausted or the endpoint is reached.

### 5.3 Rocket

- target count equals level;
- the documented center damage is fixed at 20 and adjacent splash damage is 10;
- overlapping blast areas stack;
- firing starts the documented three-round cooldown;
- Phase 3 applies rockets only to modeled robots and makes no claim about whether rockets can damage enemy roles or buildings.

The conservative three-interpretation ray policy from the original design remains. Any request value that contradicts a required documented invariant makes Phase 3 unsupported and triggers atomic Phase 2 fallback.

## 6. Robot scenario corrections

The four deterministic scenario families remain, but every scenario must account for roles as attackable blockers and range-three attacks:

1. **Station shortest path:** follow a stable shortest route to a station footprint cell; attack the blocking role or building when a documented attack is available.
2. **Main-path blocker:** prefer the attackable role or building whose removal most improves the best station route.
3. **Low-health blocker:** prefer the lowest-health attackable blocking role or building among legal attacks, with path impact, distance, and stable identity as tie-breakers.
4. **Maximum station progress:** select the legal move or blocker attack that maximizes expected station-path progress, with stable tie-breakers.

Scenarios do not assume robots attack only adjacent structures. They do not invent hidden robots or extra summoned robots. Observed summon surges are naturally included because those robots are present in the request; pre-night summon stress testing belongs to a later strategic phase.

## 7. Certificate metrics

Each `RobotWaveSafetyCertificate` contains:

- wave classification;
- weighted station-survival probability;
- expected, lower-tail, and worst-case station health;
- surviving controlled-role count and minimum controlled-role health;
- surviving controller and key-weapon counts;
- projected wall and weapon losses;
- projected robot kill score owned by us;
- remaining robot threat and its maximum modeled one-turn damage;
- scenario diagnostics for tests and internal logs.

Kill score is credited only when the modeled damage ledger establishes that our action kills the robot. Phase 3 does not infer kill ownership from an enemy or unmodeled damage source.

## 8. Survival-to-score ranking

Ranking uses two bands.

### 8.1 Survival band

A root is *secured* only if every valid scenario satisfies all objective reserves:

- station survives above the required health reserve;
- every required controller survives;
- every required key weapon survives;
- remaining modeled one-turn robot damage does not exceed the configured post-horizon reserve, unless the horizon ends with the night.

If no root is secured, rank all fully evaluated roots lexicographically by:

1. weighted station-survival probability;
2. lower-tail station health;
3. worst-case station health;
4. surviving required-controller count;
5. surviving key-weapon count;
6. minimum controlled-role health;
7. expected station health;
8. lower remaining robot threat;
9. projected owned kill score;
10. stable serialized action order.

An all-unsafe position still returns the least-bad fully evaluated root.

### 8.2 Score band

If at least one root is secured, discard unsecured roots and rank only secured roots by:

1. greater projected owned robot kill score;
2. lower remaining robot threat;
3. greater worst-case station health;
4. greater minimum controlled-role health;
5. greater surviving wall and non-key-weapon value;
6. stable serialized action order.

This implements “survive first, score when safe” without allowing raw HP-per-point efficiency to override immediate station or controller safety.

## 9. Session and future memory boundary

Phase 3 `StrategySession` remains process-local and current-half scoped. It owns continuity, duplicate caching, previous observation/action, and short-lived robot-scenario weights.

A later `MatchMemory` is logically separate. A Phase 3 rollback, gap, team change, map change, or station change resets the tactical session as specified in the original design, but must not be defined as deleting future cross-half memory. Phase 3 itself does not create disk state.

## 10. Module-boundary additions

The revised implementation plan must include:

- `strategy/simulation/objective.py`: immutable `NightObjective` and deterministic default;
- robot combat specifications in central rules or a focused simulation rules module;
- role health, abnormal robot state, robot attack range, and kill ownership in simulation state;
- combined role/robot intent and collision resolution in the turn kernel;
- `RobotWaveSafetyCertificate` naming and two-band ranking.

The Phase 1 parser, validator, serializer, and safe payload remain unchanged except for already-planned explicit-field presence metadata. Phase 2 world, pathfinding, controller assignment, and legal joint enumeration remain reusable fallbacks.

## 11. Required test corrections

The revised plan must add focused tests proving:

- every documented robot type attacks at range three;
- robots can attack blocking roles as well as buildings;
- a role move and robot move are resolved from simultaneous intents;
- weapon damage is committed at turn end, so a lethally hit robot still acts that turn;
- a dead controller becomes unavailable on later rollout turns;
- an observed dizzy robot waits once and then conservatively becomes active;
- Gatling uses fixed 10 damage and rocket uses fixed 20/10 damage;
- night-end robot removal prevents invented next-day damage and score;
- certificate names and diagnostics make the robot-wave-only scope explicit;
- unsecured roots use survival ranking;
- secured roots use kill-score ranking without violating configured reserves;
- enemy fire, hidden units, items, and future summons are not silently simulated;
- the 64-root, four-scenario, six-turn, and approximately 800 ms caps still hold;
- unsupported values or deadline expiry atomically return a fresh Phase 2 result;
- all existing 72 Phase 1/2 tests continue to pass.

## 12. Revised Phase 3 acceptance criteria

Phase 3 is complete when:

1. The corrected simultaneous intent/damage kernel matches the documented timing and collision rules.
2. Robot range, blocker targeting, abnormal-state handling, and weapon-specific damage semantics have direct tests.
3. Every completed search produces a `RobotWaveSafetyCertificate` whose scope cannot be mistaken for total match safety.
4. Survival-to-score ranking switches bands only when every configured reserve is satisfied.
5. A later strategic director can provide an immutable night objective without changing the simulator's rule kernel.
6. Unsupported input, discontinuity, deadline expiry, and internal simulation failure atomically invoke Phase 2.
7. Determinism, concurrency, duplicate handling, fixed work caps, Python 3.11 compatibility, compilation, HTTP probes, and the complete regression suite pass.
