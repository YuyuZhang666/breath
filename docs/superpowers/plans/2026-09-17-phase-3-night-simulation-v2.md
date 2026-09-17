# Phase 3 Robot-Wave Night Simulation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, stateful, safety-gated night simulator that models the documented visible-robot rules, evaluates at most 64 legal roots across four robot scenarios, switches from survival to kill-score utility only after explicit reserves are satisfied, and falls back atomically to Phase 2.

**Architecture:** Keep Phase 1 as the protocol/validation/fallback boundary and Phase 2 as the stateless spatial and legal-action fallback. Add focused simulation modules, an immutable `NightObjective`, a robot-wave-scoped certificate, short-lived scenario calibration, and one locked process-level `StrategyEngine`. Enemy weapon fire, items, economy, task solving, opponent archetypes, summons, and cross-half memory remain outside this implementation.

**Tech Stack:** Python 3.11-compatible standard-library code, frozen dataclasses, `fractions.Fraction`, `hashlib`, `itertools`, `threading.RLock`, `time.monotonic`, `unittest`, and the existing HTTP/decision stack.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-3-night-simulation-design.md` plus the authoritative `docs/superpowers/specs/2026-09-17-phase-3-night-simulation-amendment.md`. Architectural context is in `docs/superpowers/specs/2026-09-17-adaptive-match-strategy-design.md`.

## Global Constraints

- This plan supersedes `docs/superpowers/plans/2026-09-17-phase-3-night-simulation.md`. Do not execute tasks from both plans.
- Target syntax and behavior are Python 3.11. Local verification may use the configured `python` executable; final deployment verification must also run in the organizer's Python 3.11 environment.
- Runtime code uses only the Python standard library.
- The HTTP request/response schema and Phase 1 safe payload do not change.
- Phase 3 simulates only alive, visible robots with an explicit `targetTeam` equal to `observation.our.team_type`.
- A `RobotWaveSafetyCertificate` never claims safety from enemy weapons, hidden enemy roles, future waves, future summons, or experimental rule flags.
- Search evaluates at most 64 roots, exactly four policies per root, and at most six turns per policy.
- Planning has an approximately 800 ms watchdog. Deadline expiry discards all partial Phase 3 work and invokes Phase 2 from the original observation.
- Candidate generation, iteration, path ties, scenario order, certificate math, and score ties are deterministic.
- Every behavior change is developed test-first. Each task ends with focused tests, the complete suite, `git diff --check`, and a focused commit.
- Preserve unrelated user changes. Do not rewrite Phase 1/2 modules beyond the named extension seams.

## File Structure

Create:

- `future_war_agent/strategy/simulation/__init__.py` — public Phase 3 simulation exports.
- `future_war_agent/strategy/simulation/errors.py` — `UnsupportedSimulation` and `DeadlineExceeded`.
- `future_war_agent/strategy/simulation/config.py` — robot specifications, fixed weapon damage, and work caps.
- `future_war_agent/strategy/simulation/objective.py` — immutable `NightObjective`.
- `future_war_agent/strategy/simulation/state.py` — protocol conversion and immutable simulation entities.
- `future_war_agent/strategy/simulation/geometry.py` — cone and conservative ray geometry.
- `future_war_agent/strategy/simulation/weapons.py` — target generation and damage kernels.
- `future_war_agent/strategy/simulation/candidates.py` — Phase 2 assignment reuse and legal root conversion.
- `future_war_agent/strategy/simulation/movement.py` — simultaneous role/robot collision resolution.
- `future_war_agent/strategy/simulation/robots.py` — four deterministic robot policies.
- `future_war_agent/strategy/simulation/kernel.py` — corrected turn transition and damage ledger.
- `future_war_agent/strategy/simulation/future.py` — fixed post-root rollout policy.
- `future_war_agent/strategy/simulation/certificate.py` — wave classification, aggregation, and two-band ranking.
- `future_war_agent/strategy/simulation/search.py` — bounded root/scenario orchestration.
- `future_war_agent/strategy/session.py` — current-half sessions, fingerprints, and cache records.
- `future_war_agent/strategy/reconcile.py` — exact scenario-weight update.
- `future_war_agent/strategy/engine.py` — locked lifecycle and atomic fallback.
- `tests/test_simulation_foundation.py`
- `tests/test_simulation_state.py`
- `tests/test_simulation_geometry.py`
- `tests/test_simulation_weapons.py`
- `tests/test_simulation_candidates.py`
- `tests/test_simulation_movement.py`
- `tests/test_simulation_robots.py`
- `tests/test_simulation_kernel.py`
- `tests/test_simulation_certificate.py`
- `tests/test_simulation_search.py`
- `tests/test_strategy_session.py`
- `tests/test_strategy_reconcile.py`
- `tests/test_strategy_engine.py`
- `tests/fixtures/phase3_day_request.json`
- `tests/fixtures/phase3_night_request.json`

Modify:

- `future_war_agent/protocol/models.py:24-34` — preserve explicitly supplied unit fields.
- `future_war_agent/protocol/parser.py:110-127` — populate field-presence metadata.
- `future_war_agent/strategy/joint.py:14-44, 136-230, 329-365` — multi-target attacks and reusable legal-joint enumeration.
- `future_war_agent/controller.py:1-27` — connect one process-level engine while retaining injection and Phase 1 fallback.
- `tests/strategy_helpers.py:14-60` — build explicit combat fields and abnormal robot states.
- `tests/test_parser.py`, `tests/test_joint.py`, `tests/test_controller.py` — regression and integration coverage.

---

### Task 1: Explicit Field Presence, Phase 3 Configuration, and Objective

**Files:**
- Modify: `future_war_agent/protocol/models.py:24-34`
- Modify: `future_war_agent/protocol/parser.py:110-127`
- Create: `future_war_agent/strategy/simulation/__init__.py`
- Create: `future_war_agent/strategy/simulation/errors.py`
- Create: `future_war_agent/strategy/simulation/config.py`
- Create: `future_war_agent/strategy/simulation/objective.py`
- Modify: `tests/strategy_helpers.py:14-60`
- Modify: `tests/test_parser.py`
- Create: `tests/test_simulation_foundation.py`

**Interfaces:**
- Consumes: existing `UnitState`, `RobotState`, parser helpers, and documented robot/weapon constants.
- Produces: `UnitState.provided_fields`, `RobotSpec`, `Phase3Config`, `NightObjective`, `DEFAULT_PHASE3_CONFIG`, `DEFAULT_NIGHT_OBJECTIVE`, `UnsupportedSimulation`, and `DeadlineExceeded`.

- [ ] **Step 1: Write failing parser-presence and helper tests**

Add to `tests/test_parser.py`:

```python
def test_parser_preserves_explicit_unit_fields(self) -> None:
    raw = deepcopy(self.payload)
    weapon = raw["teamOur"]["roles"][1]
    weapon["attackPower"] = 10
    weapon["attackRange"] = 4
    weapon["level"] = 1
    weapon["cooldown"] = 0
    parsed = parse_observation(raw)
    observed = parsed.our.units[1]
    self.assertEqual(
        observed.provided_fields,
        frozenset({"attackPower", "attackRange", "level", "cooldown"}),
    )

def test_missing_cooldown_is_defaulted_but_not_marked_explicit(self) -> None:
    raw = deepcopy(self.payload)
    weapon = raw["teamOur"]["roles"][1]
    weapon.pop("cooldown", None)
    parsed = parse_observation(raw)
    observed = parsed.our.units[1]
    self.assertEqual(observed.cooldown, 0)
    self.assertNotIn("cooldown", observed.provided_fields)
```

Extend `unit()` with `attack_power` and `provided_fields` and extend `robot()` with `abnormal_state`. Add a helper test proving passed iterables become frozen values.

- [ ] **Step 2: Run the focused tests and witness the failure**

Run:

```powershell
python -m unittest tests.test_parser tests.test_simulation_foundation -v
```

Expected: failure because `provided_fields` and the Phase 3 foundation modules do not exist.

- [ ] **Step 3: Preserve explicit fields without changing Phase 1/2 defaults**

Append this field to `UnitState`:

```python
provided_fields: frozenset[str] = frozenset()
```

In `_parse_unit` pass only the four wire names Phase 3 needs:

```python
provided_fields=frozenset(
    name
    for name in ("attackPower", "attackRange", "level", "cooldown")
    if name in value and value[name] is not None
),
```

Keep `_optional_int` and `_nullable_int` behavior unchanged so every Phase 1/2 regression remains stable.

- [ ] **Step 4: Add exact simulation errors, robot specs, and work caps**

`errors.py`:

```python
class UnsupportedSimulation(RuntimeError):
    """The observation cannot be modeled without inventing rules."""


class DeadlineExceeded(RuntimeError):
    """The atomic Phase 3 work budget expired."""
```

`config.py`:

```python
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RobotSpec:
    attack_power: int
    attack_range: int
    max_health: int
    kill_score: int


ROBOT_SPECS: Mapping[str, RobotSpec] = MappingProxyType({
    "smallRobot": RobotSpec(5, 3, 40, 1),
    "middleRobot": RobotSpec(10, 3, 60, 2),
    "largeRobot": RobotSpec(20, 3, 500, 4),
    "bossRobot": RobotSpec(40, 3, 800, 10),
})


@dataclass(frozen=True, slots=True)
class Phase3Config:
    max_root_actions: int = 64
    scenario_count: int = 4
    max_horizon: int = 6
    watchdog_seconds: float = 0.8
    scenario_weight_floor: Fraction = Fraction(1, 20)
    gatling_damage: int = 10
    rocket_center_damage: int = 20
    rocket_splash_damage: int = 10
    rocket_cooldown_rounds: int = 3
    max_weapon_candidates: int = 3

    def __post_init__(self) -> None:
        if self.max_root_actions != 64:
            raise ValueError("Phase 3 root cap must remain 64")
        if self.scenario_count != 4:
            raise ValueError("Phase 3 requires exactly four scenarios")
        if not 1 <= self.max_horizon <= 6:
            raise ValueError("max_horizon must be between one and six")
        if self.watchdog_seconds <= 0:
            raise ValueError("watchdog_seconds must be positive")


DEFAULT_PHASE3_CONFIG = Phase3Config()
```

- [ ] **Step 5: Add the immutable night objective**

`objective.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NightObjective:
    minimum_station_health: int = 1
    protect_all_controllers: bool = True
    protect_all_weapons: bool = True
    require_post_horizon_buffer: bool = True
    enable_score_band: bool = True

    def __post_init__(self) -> None:
        if self.minimum_station_health < 1:
            raise ValueError("minimum_station_health must be positive")


DEFAULT_NIGHT_OBJECTIVE = NightObjective()
```

Test all defaults and invalid values in `tests/test_simulation_foundation.py`.

- [ ] **Step 6: Run focused and complete tests**

```powershell
python -m unittest tests.test_parser tests.test_simulation_foundation -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all existing 72 tests plus the new tests pass.

- [ ] **Step 7: Commit the foundation**

```powershell
git add future_war_agent/protocol future_war_agent/strategy/simulation tests/strategy_helpers.py tests/test_parser.py tests/test_simulation_foundation.py
git commit -m "feat: define phase 3 simulation foundation"
```

---

### Task 2: Immutable Simulation State and Eligibility Conversion

**Files:**
- Create: `future_war_agent/strategy/simulation/state.py`
- Create: `tests/test_simulation_state.py`

**Interfaces:**
- Consumes: `Observation`, Phase 2 `ControllerAssignment` values, `Phase3Config`, `ROBOT_SPECS`, and `station_footprint`.
- Produces: `SimRole`, `SimStructure`, `SimWeapon`, `SimRobot`, `SimState`, `AssignedStand`, and `build_sim_state(observation, assignments, config)`.

- [ ] **Step 1: Write failing state-conversion tests**

Cover these named cases:

```python
def test_build_state_keeps_role_health_and_assignments(self) -> None:
    state = build_sim_state(self.observed, self.assignments)
    self.assertEqual(state.roles[0].health, 220)
    self.assertEqual(state.roles[0].assigned_weapon_id, 10020)

def test_robot_spec_supplies_range_three_and_kill_score(self) -> None:
    state = build_sim_state(self.observed, self.assignments)
    self.assertEqual(state.robots[0].attack_range, 3)
    self.assertEqual(state.robots[0].kill_score, 2)

def test_unknown_robot_type_is_unsupported(self) -> None:
    observed = replace(
        self.observed,
        robots=(robot(9, 8, 8, role_type="mysteryRobot"),),
    )
    with self.assertRaises(UnsupportedSimulation):
        build_sim_state(observed, self.assignments)
```

Also test: day request, blank team type, missing `targetTeam` on any alive robot, invalid coordinates, missing station, multiple stations, missing explicit weapon fields, incompatible Gatling/Rocket attack power, unknown weapon type, ignored robots targeting the other team, terrain blockers, station footprint, dizzy state, and `remaining_night_turns = 60 - round_in_phase + 1`.

- [ ] **Step 2: Run the state tests and witness the failure**

Run:

```powershell
python -m unittest tests.test_simulation_state -v
```

Expected: import failure for `simulation.state`.

- [ ] **Step 3: Implement focused immutable state types**

Use these public types:

```python
@dataclass(frozen=True, slots=True)
class AssignedStand:
    role_id: int
    weapon_id: int
    stand: Position


@dataclass(frozen=True, slots=True)
class SimRole:
    unit_id: int
    role_type: str
    position: Position
    health: int
    assigned_weapon_id: int | None
    assigned_stand: Position | None


@dataclass(frozen=True, slots=True)
class SimStructure:
    unit_id: int
    role_type: str
    position: Position
    occupied_cells: frozenset[Position]
    health: int
    level: int | None


@dataclass(frozen=True, slots=True)
class SimWeapon:
    unit_id: int
    role_type: str
    position: Position
    health: int
    attack_power: int
    attack_range: int
    level: int
    cooldown: int


@dataclass(frozen=True, slots=True)
class SimRobot:
    robot_id: int
    role_type: str
    position: Position
    health: int
    attack_power: int
    attack_range: int
    kill_score: int
    waits_this_turn: bool


@dataclass(frozen=True, slots=True)
class SimState:
    round_no: int
    width: int
    height: int
    remaining_night_turns: int
    team_type: str
    static_blocked: frozenset[Position]
    roles: tuple[SimRole, ...]
    station: SimStructure
    walls: tuple[SimStructure, ...]
    weapons: tuple[SimWeapon, ...]
    robots: tuple[SimRobot, ...]
    owned_kill_score: int = 0
```

- [ ] **Step 4: Implement strict conversion**

Implement:

```python
def build_sim_state(
    observation: Observation,
    assignments: tuple[AssignedStand, ...],
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> SimState:
```

The body performs checks in this order:

1. Require night phase and non-blank `our.team_type`.
2. Require exactly one living station.
3. Validate every observed coordinate before filtering units.
4. Require every alive robot to have a non-blank `target_team`; retain only robots targeting us.
5. Resolve each retained robot through `ROBOT_SPECS`.
6. Require every living weapon to be Gatling, railgun, or rocket and to contain explicit `attackPower`, `attackRange`, `level`, and `cooldown`.
7. Require positive range/level, non-negative cooldown, Gatling attack power 10, and rocket attack power 20. Railgun energy must be positive.
8. Convert own living workers/pioneers, station, walls, and weapons in ID order.
9. Build `static_blocked` from neutral zones and every visible building footprint; dynamic roles and robots stay out of this set.
10. Mark `waits_this_turn` when `abnormal_state.casefold() == "dizzy"`.

Any failed condition raises `UnsupportedSimulation` with an internal-only reason string.

- [ ] **Step 5: Run focused and complete tests**

```powershell
python -m unittest tests.test_simulation_state -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all pass and direct helper constructors remain backward compatible.

- [ ] **Step 6: Commit immutable state**

```powershell
git add future_war_agent/strategy/simulation/state.py tests/test_simulation_state.py
git commit -m "feat: model immutable robot-wave state"
```

---

### Task 3: Conservative Geometry and Weapon Kernels

**Files:**
- Create: `future_war_agent/strategy/simulation/geometry.py`
- Create: `future_war_agent/strategy/simulation/weapons.py`
- Create: `tests/test_simulation_geometry.py`
- Create: `tests/test_simulation_weapons.py`

**Interfaces:**
- Consumes: `SimState`, `SimWeapon`, `Position`, and `Phase3Config`.
- Produces: `WeaponAttack`, `bresenham_cells`, `supercover_cells`, `center_intersection_cells`, `is_legal_cone`, `generate_weapon_attacks`, and `weapon_damage`.

- [ ] **Step 1: Write failing exact-geometry tests**

Test horizontal, vertical, diagonal, shallow-slope, steep-slope, and endpoint-inclusive rays. Assert each function returns a tuple beginning after the weapon cell and ending at the target cell. Add:

```python
def test_cone_accepts_ninety_degrees_and_rejects_more(self) -> None:
    origin = Position(5, 5)
    self.assertTrue(
        is_legal_cone(origin, (Position(8, 5), Position(5, 8)))
    )
    self.assertFalse(
        is_legal_cone(origin, (Position(8, 5), Position(4, 8)))
    )
```

Implement cone comparison with integer dot products, not floating-point angles: every pair of target direction vectors must have a dot product greater than or equal to zero.

- [ ] **Step 2: Write failing weapon tests**

Cover:

- Gatling level-sized target count, 90-degree cone, fixed 10 damage, and nearest robot under all three ray interpretations.
- Railgun one target, ray order, residual energy, and stop at zero.
- Rocket level-sized distinct targets, fixed 20 center/10 splash, overlapping areas, and cooldown eligibility.
- Range checks and stable truncation to at most three attacks.
- Conservative ray folding: per-robot damage is the minimum across the three interpretations.
- No candidate when there are fewer legal distinct target cells than the weapon level.

- [ ] **Step 3: Run both modules and witness failure**

```powershell
python -m unittest tests.test_simulation_geometry tests.test_simulation_weapons -v
```

Expected: import failures.

- [ ] **Step 4: Implement exact geometry**

Export:

```python
Ray = tuple[Position, ...]


def bresenham_cells(start: Position, end: Position) -> Ray:
def supercover_cells(start: Position, end: Position) -> Ray:
def center_intersection_cells(start: Position, end: Position) -> Ray:
def is_legal_cone(origin: Position, targets: tuple[Position, ...]) -> bool:
```

Use integer arithmetic or `Fraction` for cell-center intersection. Do not use rounded floating-point slopes. Each ray excludes `start` and includes `end`. Deduplicate cells while preserving traversal order.

- [ ] **Step 5: Implement attack values and deterministic target generation**

`weapons.py`:

```python
@dataclass(frozen=True, slots=True)
class WeaponAttack:
    weapon_id: int
    controller_id: int
    targets: tuple[Position, ...]

    @property
    def stable_key(self) -> tuple[object, ...]:
        return (
            self.weapon_id,
            self.controller_id,
            tuple((target.x, target.y) for target in self.targets),
        )


def generate_weapon_attacks(
    state: SimState,
    weapon: SimWeapon,
    controller_id: int,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> tuple[WeaponAttack, ...]:


def weapon_damage(
    state: SimState,
    attack: WeaponAttack,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> Mapping[int, int]:
```

Candidate cells are:

- live robot positions for Gatling and railgun;
- every in-bounds cell within Chebyshev distance one of a live robot for rockets.

Filter cells by weapon range and stable coordinate order. Enumerate distinct target tuples of exact level for Gatling/Rocket and singletons for railgun. Reject illegal cones. Score raw attacks by immediate killed robot score, total threat removed, total conservative damage, then `stable_key`. Return at most `config.max_weapon_candidates`.

`weapon_damage` returns robot ID to damage and never mutates state.

- [ ] **Step 6: Run focused and complete tests**

```powershell
python -m unittest tests.test_simulation_geometry tests.test_simulation_weapons -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all weapon semantics use fixed documented values and all tests pass.

- [ ] **Step 7: Commit geometry and weapons**

```powershell
git add future_war_agent/strategy/simulation/geometry.py future_war_agent/strategy/simulation/weapons.py tests/test_simulation_geometry.py tests/test_simulation_weapons.py
git commit -m "feat: simulate documented weapon geometry"
```

---

### Task 4: Reusable Legal Joints and Phase 3 Root Candidates

**Files:**
- Modify: `future_war_agent/strategy/joint.py`
- Create: `future_war_agent/strategy/simulation/candidates.py`
- Modify: `tests/test_joint.py`
- Create: `tests/test_simulation_candidates.py`

**Interfaces:**
- Consumes: Phase 2 `TacticalCandidate`, `assign_controllers`, `WorldGrid`, `WeaponAttack`, and `SimState`.
- Produces: multi-target `TacticalCandidate.attack`, `enumerate_legal_joints`, `decision_for_joint`, `SimJointAction`, `RoleMove`, and `generate_root_actions`.

- [ ] **Step 1: Write failing Phase 2 compatibility and multi-target tests**

Add tests that:

- existing one-target Phase 2 calls serialize identically;
- `TacticalCandidate.attack(..., targets=(a, b))` preserves both targets;
- a level-two Gatling legal attack is accepted only through an injected attack validator;
- default Phase 2 validation still requires exactly one target;
- legal enumeration returns deterministic tuples and honors `limit=64`.

- [ ] **Step 2: Run the joint tests and witness failure**

```powershell
python -m unittest tests.test_joint tests.test_simulation_candidates -v
```

Expected: signature/import failures.

- [ ] **Step 3: Generalize the joint seam without changing Phase 2 behavior**

Change the constructor to:

```python
@classmethod
def attack(
    cls,
    role_id: int,
    start: Position,
    weapon_id: int,
    targets: Iterable[Position],
    priority: int,
) -> "TacticalCandidate":
    return cls(
        role_id=role_id,
        command_actor_id=weapon_id,
        action=Action.attack(role_id, tuple(targets)),
        job_kind=JobKind.ATTACK,
        start=start,
        priority=priority,
        completes_job=True,
        progress=1,
    )
```

Update Phase 2 call sites to pass `(target,)`.

Add:

```python
AttackValidator = Callable[[UnitState, UnitState, Action], bool]
Joint = tuple[TacticalCandidate, ...]


def enumerate_legal_joints(
    observation: Observation,
    world: WorldGrid,
    choices: Mapping[int, tuple[TacticalCandidate, ...]],
    *,
    limit: int | None = None,
    attack_validator: AttackValidator | None = None,
) -> tuple[Joint, ...]:


def decision_for_joint(joint: Joint) -> Decision:
```

`solve_joint` calls these functions with the default Phase 2 validator. Enumeration walks `product` in stable role-ID/candidate order and truncates only after a complete legal joint is produced.

- [ ] **Step 4: Implement root action conversion**

`candidates.py`:

```python
@dataclass(frozen=True, slots=True)
class RoleMove:
    role_id: int
    target: Position


@dataclass(frozen=True, slots=True)
class SimJointAction:
    role_moves: tuple[RoleMove, ...] = ()
    weapon_attacks: tuple[WeaponAttack, ...] = ()


@dataclass(frozen=True, slots=True)
class RootAction:
    decision: Decision
    simulation_action: SimJointAction
    stable_key: tuple[object, ...]


def generate_root_actions(
    observation: Observation,
    world: WorldGrid,
    state: SimState,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> tuple[RootAction, ...]:
```

Reuse `assign_controllers`. A role not at its assigned stand receives the existing Phase 2 move/wait candidates. An adjacent controller receives every generated weapon attack plus wait. Unassigned roles receive Phase 2 safe-position/wait choices. Pass a Phase 3 attack validator that checks controller adjacency, cooldown, exact target count, range, and cone. Enumerate at most 64 legal roots and convert each joint to a `Decision` and `SimJointAction`.

- [ ] **Step 5: Run focused and complete tests**

```powershell
python -m unittest tests.test_joint tests.test_night tests.test_simulation_candidates -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: Phase 2 behavior stays byte-equivalent and legal Phase 3 roots are deterministic.

- [ ] **Step 6: Commit the reusable joint seam**

```powershell
git add future_war_agent/strategy/joint.py future_war_agent/strategy/night.py future_war_agent/strategy/simulation/candidates.py tests/test_joint.py tests/test_night.py tests/test_simulation_candidates.py
git commit -m "feat: enumerate legal phase 3 roots"
```

---

### Task 5: Robot Policies and Simultaneous Movement

**Files:**
- Create: `future_war_agent/strategy/simulation/movement.py`
- Create: `future_war_agent/strategy/simulation/robots.py`
- Create: `tests/test_simulation_movement.py`
- Create: `tests/test_simulation_robots.py`

**Interfaces:**
- Consumes: immutable `SimState` and root `RoleMove` values.
- Produces: `MoveIntent`, `RobotIntent`, `RobotPolicy`, `ALL_ROBOT_POLICIES`, `choose_robot_intents`, and `resolve_simultaneous_moves`.

- [ ] **Step 1: Write failing collision tests**

Cover same destination, role/robot swap, robot/robot swap, move into a stationary actor, static blocker, independent moves, and deterministic three-actor contention:

```python
def test_role_and_robot_contesting_one_cell_both_stay(self) -> None:
    intents = (
        MoveIntent("role", 1, Position(4, 4), Position(5, 5)),
        MoveIntent("robot", 9, Position(6, 6), Position(5, 5)),
    )
    resolved = resolve_simultaneous_moves(self.state, intents)
    self.assertEqual(resolved[("role", 1)], Position(4, 4))
    self.assertEqual(resolved[("robot", 9)], Position(6, 6))
```

- [ ] **Step 2: Write failing policy tests**

For each policy assert:

- a blocker within range three produces an attack intent;
- the blocker may be a role, wall, weapon, or station;
- a blocker at range four is not attacked;
- a dizzy robot waits;
- stable ties choose the same target/move repeatedly;
- no policy mutates the input state.

- [ ] **Step 3: Run tests and witness failure**

```powershell
python -m unittest tests.test_simulation_movement tests.test_simulation_robots -v
```

Expected: import failures.

- [ ] **Step 4: Implement simultaneous movement**

`movement.py`:

```python
ActorKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class MoveIntent:
    actor_kind: str
    actor_id: int
    start: Position
    target: Position


def resolve_simultaneous_moves(
    state: SimState,
    intents: tuple[MoveIntent, ...],
) -> Mapping[ActorKey, Position]:
```

Build all intent maps before resolving any position. Mark an intent blocked when its target is out of bounds, statically blocked, occupied by a non-moving dynamic actor, contested by another intent, or part of a position swap. Return every moving actor's final position; blocked actors retain `start`.

- [ ] **Step 5: Implement four deterministic robot policies**

`robots.py`:

```python
class RobotPolicy(StrEnum):
    STATION_SHORTEST_PATH = "station_shortest_path"
    MAIN_PATH_BLOCKER = "main_path_blocker"
    LOW_HEALTH_BLOCKER = "low_health_blocker"
    MAXIMUM_STATION_PROGRESS = "maximum_station_progress"


ALL_ROBOT_POLICIES = tuple(RobotPolicy)


@dataclass(frozen=True, slots=True)
class RobotIntent:
    robot_id: int
    move_target: Position | None = None
    attack_target_kind: str | None = None
    attack_target_id: int | None = None


def choose_robot_intents(
    state: SimState,
    policy: RobotPolicy,
) -> tuple[RobotIntent, ...]:
```

Use a stable eight-neighbor BFS over the simulation occupancy. Identify attackable blocking roles/buildings on candidate station routes. A legal attack requires Chebyshev distance no greater than the robot's documented range three. Each policy applies the exact preference stated in the amendment; ties use target kind, target ID, coordinate, then fixed direction order. A robot that attacks does not move.

- [ ] **Step 6: Run focused and complete tests**

```powershell
python -m unittest tests.test_simulation_movement tests.test_simulation_robots -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all pass with no order-dependent collision.

- [ ] **Step 7: Commit robot intent logic**

```powershell
git add future_war_agent/strategy/simulation/movement.py future_war_agent/strategy/simulation/robots.py tests/test_simulation_movement.py tests/test_simulation_robots.py
git commit -m "feat: model simultaneous robot intent"
```

---

### Task 6: Corrected Turn Kernel and Fixed Future Policy

**Files:**
- Create: `future_war_agent/strategy/simulation/kernel.py`
- Create: `future_war_agent/strategy/simulation/future.py`
- Create: `tests/test_simulation_kernel.py`

**Interfaces:**
- Consumes: `SimState`, `SimJointAction`, `RobotPolicy`, weapon kernels, robot intents, movement resolver, config, and objective.
- Produces: `DamageLedger`, `step_simulation`, and `choose_future_action`.

- [ ] **Step 1: Write failing turn-order tests**

Add named tests for:

- a robot receiving lethal weapon damage still attacks this turn;
- weapon and robot damage commit simultaneously;
- a role and robot move from the same snapshot and collide;
- a range-three attack damages a blocking role;
- a dead controller cannot control a weapon next turn;
- a dizzy robot waits on the first step and is active on the second;
- a fired rocket remains at cooldown three after the firing turn, then decrements on later turns;
- non-fired cooldown decrements once;
- remaining night turns decrement once;
- the end-of-night state clears robots without awarding kill score.

- [ ] **Step 2: Run the kernel tests and witness failure**

```powershell
python -m unittest tests.test_simulation_kernel -v
```

Expected: import failure.

- [ ] **Step 3: Implement the damage ledger and exact step order**

`kernel.py`:

```python
@dataclass(slots=True)
class DamageLedger:
    role_damage: dict[int, int]
    structure_damage: dict[int, int]
    weapon_damage: dict[int, int]
    robot_damage_by_us: dict[int, int]

    @classmethod
    def empty(cls) -> "DamageLedger":
        return cls({}, {}, {}, {})


def step_simulation(
    state: SimState,
    action: SimJointAction,
    robot_policy: RobotPolicy,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> SimState:
```

Implement the amended order exactly:

1. Snapshot state and create a ledger.
2. Add weapon damage without applying it.
3. Create role and robot intents from the snapshot.
4. Resolve all movement together.
5. Add robot attack damage against the selected blocking role/building.
6. Apply every ledger entry simultaneously.
7. Remove dead entities and increment `owned_kill_score` only for robots crossing from alive to dead with positive `robot_damage_by_us`.
8. For fired rockets set cooldown to three; for all other weapons decrement positive cooldown once.
9. Clear `waits_this_turn` on every surviving robot so an initially dizzy robot becomes active on the next simulated turn.
10. Advance round and remaining-night count.
11. If no night turns remain, clear robots without changing `owned_kill_score`.

Never mutate an input tuple or dataclass.

- [ ] **Step 4: Implement the fixed future policy**

`future.py`:

```python
def choose_future_action(
    state: SimState,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> SimJointAction:
```

For each living assigned controller, move one stable shortest step toward its assigned stand or control the assigned ready weapon when adjacent. A ready weapon chooses the generated attack with the greatest immediate killed score, removed threat, damage, then stable key. Unassigned roles and unavailable weapons wait. Resolve controller conflicts by ascending role ID and never assign one role twice.

- [ ] **Step 5: Run focused and complete tests**

```powershell
python -m unittest tests.test_simulation_kernel -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all corrected ordering, cooldown, stun, death, and night-end tests pass.

- [ ] **Step 6: Commit the turn kernel**

```powershell
git add future_war_agent/strategy/simulation/kernel.py future_war_agent/strategy/simulation/future.py tests/test_simulation_kernel.py
git commit -m "feat: implement corrected night turn kernel"
```

---

### Task 7: Robot-Wave Certificates and Fixed-Budget Search

**Files:**
- Create: `future_war_agent/strategy/simulation/certificate.py`
- Create: `future_war_agent/strategy/simulation/search.py`
- Create: `tests/test_simulation_certificate.py`
- Create: `tests/test_simulation_search.py`

**Interfaces:**
- Consumes: legal root actions, four policies, fixed future policy, `NightObjective`, `Phase3Config`, exact scenario weights, and an injected monotonic clock.
- Produces: `WaveClassification`, `ScenarioOutcome`, `RobotWaveSafetyCertificate`, `SearchStats`, `SearchResult`, `build_certificate`, and `search_night`.

- [ ] **Step 1: Write failing certificate tests**

Cover exact `Fraction` math, weighted survival, expected health, weighted p10, worst health, controller/weapon loss aggregation, kill score, remaining threat, and classifications:

```python
def test_certificate_name_and_classification_are_wave_scoped(self) -> None:
    certificate = build_certificate(self.safe_outcomes, self.objective)
    self.assertIs(
        certificate.classification,
        WaveClassification.WAVE_SAFE,
    )
    self.assertTrue(certificate.secured)

def test_score_band_is_unavailable_when_one_controller_dies(self) -> None:
    outcomes = replace_one(self.safe_outcomes, controller_losses=1)
    certificate = build_certificate(outcomes, self.objective)
    self.assertFalse(certificate.secured)
```

Add a night-end case where remaining robots do not require a post-horizon buffer.

- [ ] **Step 2: Write failing search and ranking tests**

Prove:

- no more than 64 roots, four scenarios, and six turns are evaluated;
- if no root is secured, station survival and lower-tail health beat kill score;
- if secured roots exist, an unsecured root is discarded and higher owned kill score wins among secured roots;
- stable ties select byte-equivalent decisions;
- all-unsafe states return the least-bad fully evaluated root;
- rocket cooldown changes a future rollout preference;
- a deadline before or during evaluation raises `DeadlineExceeded` and returns no partial result.

- [ ] **Step 3: Run certificate/search tests and witness failure**

```powershell
python -m unittest tests.test_simulation_certificate tests.test_simulation_search -v
```

Expected: import failures.

- [ ] **Step 4: Implement exact outcomes and certificates**

`certificate.py`:

```python
class WaveClassification(StrEnum):
    WAVE_SAFE = "wave_safe"
    WAVE_MARGINAL = "wave_marginal"
    WAVE_UNSAFE = "wave_unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    weight: Fraction
    station_health: int
    controller_losses: int
    key_weapon_losses: int
    minimum_role_health: int
    surviving_asset_health: int
    owned_kill_score: int
    remaining_threat: int
    remaining_one_turn_damage: int
    ended_with_night: bool


@dataclass(frozen=True, slots=True)
class RobotWaveSafetyCertificate:
    classification: WaveClassification
    secured: bool
    station_survival_probability: Fraction
    expected_station_health: Fraction
    p10_station_health: int
    worst_station_health: int
    worst_controller_losses: int
    worst_key_weapon_losses: int
    worst_minimum_role_health: int
    expected_owned_kill_score: Fraction
    expected_remaining_threat: Fraction
    outcomes: tuple[ScenarioOutcome, ...]
```

Require non-empty outcomes with positive weights summing to exactly one. Compute p10 by sorting station health ascending and accumulating weight until `Fraction(1, 10)` is reached. `secured` is true only when every outcome meets the objective's station, controller, weapon, and post-horizon buffer rules.

Export separate `survival_rank_key` and `score_rank_key` functions. Use the exact field order in the amendment and finish both keys with the root stable key.

- [ ] **Step 5: Implement bounded search**

`search.py`:

```python
ScenarioWeights = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class SearchStats:
    roots_generated: int
    roots_evaluated: int
    scenarios_per_root: int
    maximum_steps: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    decision: Decision
    simulation_action: SimJointAction
    certificate: RobotWaveSafetyCertificate
    stats: SearchStats


def search_night(
    observation: Observation,
    scenario_weights: ScenarioWeights,
    *,
    objective: NightObjective = DEFAULT_NIGHT_OBJECTIVE,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    clock: Callable[[], float] = monotonic,
    deadline: float | None = None,
) -> SearchResult:
```

When `deadline` is `None`, compute `effective_deadline = clock() + config.watchdog_seconds`; otherwise use the supplied deadline unchanged.

Algorithm:

1. Validate exactly four positive weights summing to one.
2. Build Phase 2 world/assignments, immutable state, and at most 64 roots.
3. Set horizon to `min(config.max_horizon, state.remaining_night_turns)`.
4. Check the deadline before each root, scenario, and simulated turn.
5. For every root and every policy, apply the root once and then `choose_future_action` for remaining steps.
6. Convert final states into exact `ScenarioOutcome` values and build one certificate per root.
7. If any certificate is secured and score mode is enabled, compare only secured roots with `score_rank_key`; otherwise compare every completed root with `survival_rank_key`.
8. Return the winning decision, its `SimJointAction`, certificate, and exact stats.
9. If the deadline is crossed at any point, raise `DeadlineExceeded`. Do not return the best completed prefix.

- [ ] **Step 6: Run focused and complete tests**

```powershell
python -m unittest tests.test_simulation_certificate tests.test_simulation_search -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: caps, two-band ranking, exact math, and atomic deadline behavior all pass.

- [ ] **Step 7: Commit certificates and search**

```powershell
git add future_war_agent/strategy/simulation/certificate.py future_war_agent/strategy/simulation/search.py tests/test_simulation_certificate.py tests/test_simulation_search.py
git commit -m "feat: rank bounded robot-wave searches"
```

---

### Task 8: Session Fingerprints and Scenario Reconciliation

**Files:**
- Create: `future_war_agent/strategy/session.py`
- Create: `future_war_agent/strategy/reconcile.py`
- Create: `tests/test_strategy_session.py`
- Create: `tests/test_strategy_reconcile.py`

**Interfaces:**
- Consumes: observations, decisions, optional certificates, Phase 3 config, and one-step simulation predictions.
- Produces: `observation_fingerprint`, `static_signature`, `StrategySession`, `SessionStore`, `uniform_scenario_weights`, and `reconcile_scenario_weights`.

- [ ] **Step 1: Write failing session lifecycle tests**

Prove:

- canonical fingerprints ignore mapping order but change on planning-relevant values;
- duplicate same-round/same-fingerprint requests return the same cached decision object;
- same-round revisions keep weights and do not reconcile;
- round gaps, rollback, map signature changes, and station identity changes reset weights;
- different team IDs are isolated;
- blank team IDs do not create sessions;
- the current certificate can be retained for future internal consumers without entering the response schema.

- [ ] **Step 2: Write failing exact reconciliation tests**

Use four injected predicted next states. Verify robot position error, alive/dead penalty, role/building/weapon health-delta error, missing-evidence no-op, exact `Fraction` normalization, and a five-percent floor.

- [ ] **Step 3: Run tests and witness failure**

```powershell
python -m unittest tests.test_strategy_session tests.test_strategy_reconcile -v
```

Expected: import failures.

- [ ] **Step 4: Implement frozen sessions and canonical hashes**

`session.py`:

```python
@dataclass(frozen=True, slots=True)
class StrategySession:
    team_id: str
    last_round: int
    fingerprint: str
    signature: str
    observation: Observation
    decision: Decision
    simulation_action: SimJointAction | None
    certificate: RobotWaveSafetyCertificate | None
    scenario_weights: ScenarioWeights


class SessionStore:
    def __init__(self) -> None:
        self._by_team: dict[str, StrategySession] = {}

    def get(self, team_id: str) -> StrategySession | None:
        return self._by_team.get(team_id)

    def put(self, session: StrategySession) -> None:
        self._by_team[session.team_id] = session
```

Fingerprint a stable tuple of round/time, map/zones, our team score/gold/tasks/units, visible enemy units, robots, shops, action results, news, and errors with SHA-256. The static signature covers dimensions, sorted zone positions/types, and station ID/position. Do not serialize mutable raw request dictionaries.

- [ ] **Step 5: Implement exact scenario weights**

`reconcile.py`:

```python
def uniform_scenario_weights() -> ScenarioWeights:
    return (
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
    )


def reconcile_scenario_weights(
    previous: StrategySession,
    current: Observation,
    *,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> ScenarioWeights:
```

For every policy, predict one turn from `previous.observation` and `previous.decision` through the same production kernel. Loss is:

- Chebyshev position error for robots alive in both states;
- 20 for each alive/dead robot mismatch;
- absolute health-delta error for each comparable controlled role, station, wall, and weapon.

If `previous.simulation_action` is `None`, no comparable evidence exists, or conversion is unsupported, return old weights unchanged. Otherwise apply the stored simulation action to each policy, compute `raw_i = old_i / (1 + loss_i)`, reserve the exact floor for every model, distribute the remaining mass proportionally, and assert the result sums exactly to one.

- [ ] **Step 6: Run focused and complete tests**

```powershell
python -m unittest tests.test_strategy_session tests.test_strategy_reconcile -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: deterministic hashes, isolation, resets, evidence handling, and exact fractions pass.

- [ ] **Step 7: Commit state reconciliation**

```powershell
git add future_war_agent/strategy/session.py future_war_agent/strategy/reconcile.py tests/test_strategy_session.py tests/test_strategy_reconcile.py
git commit -m "feat: reconcile current-half robot scenarios"
```

---

### Task 9: Locked Strategy Engine and Production Integration

**Files:**
- Create: `future_war_agent/strategy/engine.py`
- Modify: `future_war_agent/controller.py`
- Create: `tests/test_strategy_engine.py`
- Modify: `tests/test_controller.py`
- Create: `tests/fixtures/phase3_day_request.json`
- Create: `tests/fixtures/phase3_night_request.json`

**Interfaces:**
- Consumes: Phase 2 planner, searcher, objective provider, session store, reconciler, injected clock, and existing controller validation.
- Produces: `StrategyEngine.plan(observation) -> Decision` and the module-level production engine.

- [ ] **Step 1: Write failing lifecycle and cache tests**

Test:

- first mid-night observation falls back to Phase 2;
- a consecutive day-to-night request may use Phase 3 immediately;
- duplicate night requests perform one search and return the cached object;
- same-round revisions replan without reconciliation;
- gap, rollback, signature change, and blank team ID use Phase 2;
- two team IDs never share state;
- the default objective provider is called only for an eligible Phase 3 search.

- [ ] **Step 2: Write failing atomic fallback and concurrency tests**

Inject searchers that raise `UnsupportedSimulation`, `DeadlineExceeded`, and `RuntimeError`. Each must call Phase 2 once from the original observation and store the decision actually returned. Use `ThreadPoolExecutor(max_workers=4)` to prove four duplicate calls execute one search.

- [ ] **Step 3: Run engine tests and witness failure**

```powershell
python -m unittest tests.test_strategy_engine tests.test_controller -v
```

Expected: import/integration failures.

- [ ] **Step 4: Implement the locked engine**

`engine.py`:

```python
Phase2Planner = Callable[[Observation], Decision]
NightSearcher = Callable[..., SearchResult]
ObjectiveProvider = Callable[[Observation], NightObjective]


class StrategyEngine:
    def __init__(
        self,
        *,
        config: Phase3Config = DEFAULT_PHASE3_CONFIG,
        phase2_planner: Phase2Planner = plan_turn,
        night_searcher: NightSearcher = search_night,
        objective_provider: ObjectiveProvider = (
            lambda observation: DEFAULT_NIGHT_OBJECTIVE
        ),
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._config = config
        self._phase2_planner = phase2_planner
        self._night_searcher = night_searcher
        self._objective_provider = objective_provider
        self._clock = clock
        self._sessions = SessionStore()
        self._lock = RLock()

    def plan(self, observation: Observation) -> Decision:
        with self._lock:
            return self._plan_locked(observation)
```

`_plan_locked` performs this order:

1. Blank team ID calls Phase 2 without creating a session.
2. Same round and same fingerprint returns the cached decision.
3. Detect rollback, gap greater than one, static-signature change, and same-round revision.
4. Reconcile only a truly consecutive new observation.
5. Day or discontinuous night calls Phase 2.
6. Continuous night computes `deadline = clock() + config.watchdog_seconds`, resolves the immutable objective, and calls the searcher.
7. Expected Phase 3 failures and unexpected Phase 3 exceptions are logged internally, discard all Phase 3 output, and call Phase 2 once.
8. Store current observation, exact returned decision, successful `simulation_action` or `None`, optional certificate, signature, fingerprint, and weights.

Do not catch a Phase 2 exception in the engine; allow the existing controller boundary to produce `safe_payload()`.

- [ ] **Step 5: Add complete consecutive fixtures**

Create a 15x15 day round 70 fixture and matching night round 71 fixture. Both contain explicit arrays/maps, the same team ID/type, one station, three living roles, three supported weapons with all four combat fields, walls, and deterministic zones. The night fixture contains at least two robots targeting our team, including one blocker in range three, and one robot with `abnormalState: "dizzy"`.

- [ ] **Step 6: Connect the process-level engine**

Modify `controller.py`:

```python
from future_war_agent.strategy.engine import StrategyEngine


DEFAULT_STRATEGY_ENGINE = StrategyEngine()


def default_planner(observation: Observation) -> Decision:
    return DEFAULT_STRATEGY_ENGINE.plan(observation)
```

Keep planner injection, `parse_observation`, `validate_decision`, serialization, exception logging, and `safe_payload` unchanged.

- [ ] **Step 7: Add HTTP/controller integration tests**

Use a fresh engine through `handle_payload(..., planner=engine.plan)`. Prove:

- day then night returns the unchanged three-field schema and a legal non-empty night command map;
- duplicate night payloads are byte-equivalent;
- removing explicit cooldown forces a Phase 2 response;
- a planner exception still returns the Phase 1 safe payload;
- all multi-target Phase 3 actions survive the Phase 1 validator.

- [ ] **Step 8: Run focused and complete tests**

```powershell
python -m unittest tests.test_strategy_engine tests.test_controller tests.test_server -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: lifecycle, fallback, duplicate, concurrency, schema, and every accumulated test pass.

- [ ] **Step 9: Commit production integration**

```powershell
git add future_war_agent/strategy/engine.py future_war_agent/controller.py tests/test_strategy_engine.py tests/test_controller.py tests/fixtures/phase3_day_request.json tests/fixtures/phase3_night_request.json
git commit -m "feat: integrate robot-wave strategy engine"
```

---

### Task 10: Phase 3 Acceptance and Performance Verification

**Files:**
- Verify: `main.py`, all files under `future_war_agent/`, and all files under `tests/`.
- Modify only when a newly added focused failing test demonstrates an acceptance defect.

**Interfaces:**
- Consumes: the complete Phase 3 service and fixtures.
- Produces: fresh evidence for every amended acceptance criterion without implementing Phase 4 behavior.

- [ ] **Step 1: Run the complete test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: more than 72 tests, zero failures, and zero errors. Logged exceptions are acceptable only inside tests that intentionally exercise safe fallback.

- [ ] **Step 2: Compile and audit Python 3.11 compatibility**

```powershell
python -m compileall -q main.py future_war_agent tests
rg -n "match\\s|case\\s|type\\s+[A-Za-z_].*=|except\\*" future_war_agent tests
```

Expected: compilation succeeds. Any syntax audit hit is reviewed against Python 3.11; do not introduce Python 3.12-only `type Alias = ...` syntax. Repeat the full suite with the organizer's Python 3.11 interpreter before deployment.

- [ ] **Step 3: Audit dependencies and placeholders**

```powershell
rg -n "^(import|from) " future_war_agent
rg -n "TODO|TBD|PLACEHOLDER|NotImplementedError|pass$" future_war_agent tests
```

Expected: imports are standard-library or internal modules, and no Phase 3 implementation placeholder remains.

- [ ] **Step 4: Prove deterministic caps and timeout behavior**

```powershell
python -m unittest tests.test_simulation_search -v
python -m unittest tests.test_strategy_engine -v
```

Expected: at most 64 roots, exactly four scenarios, at most six turns, atomic timeout failure, and one search for concurrent duplicates.

- [ ] **Step 5: Measure local latency without turning timing into correctness**

Add a test helper that measures 20 fresh supported searches, each using a new engine primed by the day fixture, followed by 100 duplicate night calls on one primed engine. Use `time.perf_counter`, report fresh-search and cached median/p99 separately, and assert only that no individual call exceeds the external five-second limit. Record the approximately 800 ms planner target as a benchmark result, not a flaky unit-test threshold.

Run:

```powershell
python -m unittest tests.test_strategy_engine.StrategyEngineTests.test_supported_night_latency_stays_below_external_limit -v
```

Expected: pass with measured values printed. Duplicate calls should be much faster because they use the session cache.

- [ ] **Step 6: Run consecutive real HTTP requests**

Start:

```powershell
python main.py 18080
```

In a second terminal post the complete day fixture, night fixture, and the same night fixture again to `http://127.0.0.1:18080/`. Expected:

- all return HTTP 200;
- every body contains exactly `roleCommandMap`, `prompt`, and `executeCmd`;
- the two night responses are byte-equivalent and non-empty;
- the server remains alive after the requests.

Stop the server with Ctrl+C.

- [ ] **Step 7: Trace every amended requirement to a named test**

Record the exact test name beside each item:

- range-three robot attack;
- role/building blocker targeting;
- simultaneous role/robot intent;
- lethal weapon hit still permits same-turn robot action;
- later-turn controller death effect;
- one-turn conservative dizzy behavior;
- Gatling 10 and rocket 20/10;
- night-end clearing without score;
- wave-scoped certificate naming;
- unsecured survival ranking;
- secured score ranking;
- no enemy fire/items/future summons in state;
- deterministic caps and 800 ms watchdog;
- atomic Phase 2 fallback;
- duplicate/concurrency/session isolation.

If any item lacks a named test, write that one failing test, witness failure, make the minimum correction, and repeat Steps 1 through 6.

- [ ] **Step 8: Check repository integrity**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -15
```

Expected: no whitespace errors, no uncommitted files, and Phase 3 commits appear after `04e563a` on `codex/phase3-night-simulation`.

- [ ] **Step 9: Commit only if acceptance required a correction**

If Step 7 required a code/test correction:

```powershell
git add future_war_agent tests
git commit -m "fix: satisfy amended phase 3 acceptance"
```

If no correction was required, do not create an empty commit.

## Requirement Traceability

- Original design sections 5-6 and amendment sections 2, 9: Tasks 1, 2, 8, and 9.
- Original design sections 7-9 and amendment sections 3-5: Tasks 2 through 6.
- Original design section 10 and amendment section 6: Tasks 5, 6, and 8.
- Original design sections 11-13 and amendment sections 7-8: Task 7.
- Original design sections 14-15 and amendment section 10: Tasks 8 and 9.
- Amendment sections 11-12: Tasks 1 through 10, with final evidence in Task 10.
- Adaptive architecture boundaries for immutable objectives, latency, session separation, and later profiles: Tasks 1, 7, 8, and 9.
