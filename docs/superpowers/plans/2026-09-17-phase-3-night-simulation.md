# Phase 3 Night Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, stateful, safety-gated night simulator that ranks at most 64 legal root actions across four robot scenarios and falls back atomically to the Phase 2 planner whenever the state or computation is unsupported.

**Architecture:** Preserve the stateless Phase 2 `plan_turn()` as the fallback, then add a process-local `StrategyEngine` that owns per-team continuity, duplicate caching, and scenario weights. Supported night requests flow through immutable simulation state, conservative weapon geometry, four deterministic robot policies, a fixed future policy, and a lexicographic survival certificate; all errors and deadline failures discard the whole Phase 3 result.

**Tech Stack:** Python 3.11 standard library, frozen dataclasses, `fractions.Fraction`, `hashlib`, `itertools`, `threading.RLock`, `time.monotonic`, `unittest`, and the existing `ThreadingHTTPServer` integration.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-3-night-simulation-design.md`

## Global Constraints

- Target runtime is Python 3.11; use only the standard library even though additional packages are available in the approved environment.
- Work directly in the current checkout on `codex/phase3-night-simulation`; do not create a worktree.
- Keep `future_war_agent.strategy.planner.plan_turn()` stateless and behavior-compatible as the complete Phase 2 fallback.
- Phase 3 runs only for a supported night request with a preceding consecutive observation; day, first-seen, discontinuous, malformed, timed-out, and failed requests use Phase 2.
- Simulate only visible, living robots whose `targetTeam` equals `Observation.our.team_type`; use `Observation.our.team_id` only for session isolation.
- Root search is capped at 64 legal joint actions, exactly four robot policies, and `min(6, remaining night turns)` rollout steps.
- The approximately 800 ms watchdog is an atomic failure boundary: discard partial Phase 3 work and invoke Phase 2 from the original observation.
- Preserve deterministic ordering for candidates, paths, scenarios, scores, serialized actions, and tests; do not use randomness or wall time for normal winner selection.
- Keep the outward request and response schemas unchanged; certificates and fallback reasons remain internal.
- Use TDD for every implementation task: write a focused failing test, witness the expected failure, implement the minimum behavior, run focused and full regression tests, then commit.
- Never weaken the Phase 1 validator, serializer, or safe-payload exception boundary.

## File Structure

- Modify `future_war_agent/protocol/models.py` and `future_war_agent/protocol/parser.py` to preserve which optional combat fields were explicitly observed.
- Modify `future_war_agent/strategy/rules.py` for robot damage and simulator constants.
- Create `future_war_agent/strategy/simulation/` with focused `state.py`, `geometry.py`, `weapons.py`, `robots.py`, `kernel.py`, `certificate.py`, and `search.py` modules.
- Modify `future_war_agent/strategy/joint.py` only to expose stable legal-joint enumeration and strict level-aware attack legality.
- Create `future_war_agent/strategy/session.py`, `reconcile.py`, and `engine.py` for state that must span requests.
- Modify `future_war_agent/controller.py` only at the default-planner integration point.
- Add focused tests beside the existing `unittest` suite; shared simulation builders live in `tests/simulation_helpers.py`.

---

### Task 1: Preserve Combat-Field Presence and Centralize Simulation Rules

**Files:**
- Modify: `future_war_agent/protocol/models.py`
- Modify: `future_war_agent/protocol/parser.py`
- Modify: `future_war_agent/strategy/rules.py`
- Modify: `tests/strategy_helpers.py`
- Modify: `tests/test_parser.py`
- Create: `tests/test_simulation_rules.py`

**Interfaces:**
- Consumes: raw unit dictionaries already parsed by `_parse_unit`.
- Produces: `UnitState.provided_fields: frozenset[str]`, `RulesConfig.robot_attack_power(role_type)`, `rocket_cooldown_rounds`, `rocket_splash_divisor`, `scenario_weight_floor_numerator`, `scenario_weight_floor_denominator`, `phase3_root_limit`, `phase3_horizon`, and `phase3_watchdog_seconds`.

- [ ] **Step 1: Write failing parser-presence tests**

Add tests that parse the existing request fixture and a copy with `cooldown` removed from weapon `10020`:

```python
def test_unit_records_explicit_combat_fields(self) -> None:
    observed = parse_observation(self.payload)
    weapon = next(unit for unit in observed.our.units if unit.unit_id == 10020)

    self.assertTrue(
        {"attackPower", "attackRange", "level", "cooldown"}
        <= weapon.provided_fields
    )

def test_missing_cooldown_keeps_default_but_not_presence(self) -> None:
    payload = copy.deepcopy(self.payload)
    weapon = next(
        unit for unit in payload["teamOur"]["roles"] if unit["id"] == 10020
    )
    weapon.pop("cooldown")

    observed = parse_observation(payload)
    parsed = next(unit for unit in observed.our.units if unit.unit_id == 10020)

    self.assertEqual(parsed.cooldown, 0)
    self.assertNotIn("cooldown", parsed.provided_fields)
```

Import `copy` at the top of `tests/test_parser.py`.

- [ ] **Step 2: Run the parser tests and witness the missing attribute**

```powershell
python -m unittest tests.test_parser -v
```

Expected: the new tests fail because `UnitState` has no `provided_fields`.

- [ ] **Step 3: Add backward-compatible presence metadata**

Change the model and parser with this exact shape:

```python
# protocol/models.py
@dataclass(frozen=True, slots=True)
class UnitState:
    unit_id: int
    position: Position
    role_type: str
    health: int
    attack_power: int = 0
    attack_range: int = 0
    backpack_capacity: int = 0
    backpack: tuple[str, ...] = ()
    level: int | None = None
    cooldown: int = 0
    provided_fields: frozenset[str] = frozenset()
```

In `_parse_unit`, pass `provided_fields=frozenset(value)`. Do not make the optional combat fields required at parse time; Phase 1 and Phase 2 retain their current default behavior.

- [ ] **Step 4: Extend shared test builders**

Add `attack_power` and optional `provided_fields` parameters to `tests.strategy_helpers.unit`:

```python
def unit(
    unit_id: int,
    x: int,
    y: int,
    role_type: str,
    *,
    health: int = 100,
    attack_power: int = 0,
    attack_range: int = 0,
    backpack_capacity: int = 100,
    backpack: Iterable[str] = (),
    level: int | None = None,
    cooldown: int = 0,
    provided_fields: Iterable[str] = (),
) -> UnitState:
    return UnitState(
        unit_id=unit_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        attack_power=attack_power,
        attack_range=attack_range,
        backpack_capacity=backpack_capacity,
        backpack=tuple(backpack),
        level=level,
        cooldown=cooldown,
        provided_fields=frozenset(provided_fields),
    )
```

Also add keyword parameters `team_type: str = "challenger"` and `team_id: str = "team"` to `tests.strategy_helpers.observation`, and pass them into `OurTeamState`. Later session tests use these parameters to prove team isolation.

- [ ] **Step 5: Write failing simulator-rule tests**

Create `tests/test_simulation_rules.py`:

```python
import unittest

from future_war_agent.strategy.rules import DEFAULT_RULES, RulesConfig


class SimulationRulesTests(unittest.TestCase):
    def test_robot_damage_is_centralized(self) -> None:
        self.assertEqual(DEFAULT_RULES.robot_attack_power("smallRobot"), 5)
        self.assertEqual(DEFAULT_RULES.robot_attack_power("middleRobot"), 10)
        self.assertEqual(DEFAULT_RULES.robot_attack_power("largeRobot"), 20)
        self.assertEqual(DEFAULT_RULES.robot_attack_power("bossRobot"), 40)

    def test_unknown_robot_type_is_unsupported(self) -> None:
        with self.assertRaises(ValueError):
            DEFAULT_RULES.robot_attack_power("futureRobot")

    def test_phase3_limits_are_positive(self) -> None:
        rules = RulesConfig()
        self.assertEqual(rules.rocket_cooldown_rounds, 3)
        self.assertEqual(rules.phase3_root_limit, 64)
        self.assertEqual(rules.phase3_horizon, 6)
        self.assertEqual(rules.phase3_watchdog_seconds, 0.8)
```

- [ ] **Step 6: Implement simulator rules**

Add immutable fields to `RulesConfig`:

```python
robot_attack_powers: tuple[tuple[str, int], ...] = (
    ("smallRobot", 5),
    ("middleRobot", 10),
    ("largeRobot", 20),
    ("bossRobot", 40),
)
rocket_cooldown_rounds: int = 3
rocket_splash_divisor: int = 2
scenario_weight_floor_numerator: int = 1
scenario_weight_floor_denominator: int = 20
phase3_root_limit: int = 64
phase3_horizon: int = 6
phase3_watchdog_seconds: float = 0.8

def robot_attack_power(self, role_type: str) -> int:
    powers = dict(self.robot_attack_powers)
    if role_type not in powers:
        raise ValueError(f"unsupported robot type: {role_type}")
    return powers[role_type]
```

Extend `__post_init__` to reject non-positive limits, cooldown, splash divisor, watchdog, and an invalid weight-floor fraction.

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m unittest tests.test_parser tests.test_simulation_rules -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: the two presence tests, three rule tests, and all 72 pre-Phase-3 tests pass.

- [ ] **Step 8: Commit protocol support metadata**

```powershell
git add future_war_agent/protocol/models.py future_war_agent/protocol/parser.py future_war_agent/strategy/rules.py tests/strategy_helpers.py tests/test_parser.py tests/test_simulation_rules.py
git commit -m "feat: preserve phase 3 combat metadata"
```

---

### Task 2: Immutable Simulation State and Conservative Geometry

**Files:**
- Create: `future_war_agent/strategy/simulation/__init__.py`
- Create: `future_war_agent/strategy/simulation/state.py`
- Create: `future_war_agent/strategy/simulation/geometry.py`
- Create: `tests/simulation_helpers.py`
- Create: `tests/test_simulation_state.py`
- Create: `tests/test_simulation_geometry.py`

**Interfaces:**
- Consumes: `Observation`, `Position`, `RulesConfig`, and `UnitState.provided_fields`.
- Produces: `UnsupportedSimulation`, `SimUnit`, `SimRobot`, `SimState`, `sim_state_from_observation`, `GeometryMode`, `ray_cells`, `conservative_ray_hits`, and `within_ninety_degree_cone`.

- [ ] **Step 1: Write failing state-conversion tests**

Create `tests/test_simulation_state.py` with cases for a supported night state, a missing explicit cooldown, a missing robot target team, an explicitly different target team that is ignored, an unknown robot type, and an out-of-bounds entity. Use round 71 for night and give weapons this explicit field set:

```python
COMBAT_FIELDS = frozenset({"attackPower", "attackRange", "level", "cooldown"})

def test_builds_supported_state_with_only_our_threats(self) -> None:
    observed = observation(
        round_no=71,
        our_units=(
            unit(1, 1, 1, "worker"),
            unit(2, 5, 5, "station", level=1),
            unit(
                3, 4, 5, "gatling", attack_power=10, attack_range=6,
                level=2, cooldown=0, provided_fields=COMBAT_FIELDS,
            ),
        ),
        robots=(
            robot(10, 9, 5, target_team="challenger"),
            robot(11, 9, 6, target_team="other"),
        ),
    )

    state = sim_state_from_observation(observed)

    self.assertEqual(tuple(item.robot_id for item in state.robots), (10,))
    self.assertEqual(state.station_id, 2)
    self.assertEqual(state.entity(3).level, 2)

def test_missing_explicit_weapon_field_is_unsupported(self) -> None:
    observed = observation(
        round_no=71,
        our_units=(
            unit(2, 5, 5, "station", level=1),
            unit(
                3, 4, 5, "rocket", attack_power=20, attack_range=6,
                level=1, cooldown=0,
                provided_fields={"attackPower", "attackRange", "level"},
            ),
        ),
    )

    with self.assertRaises(UnsupportedSimulation):
        sim_state_from_observation(observed)
```

The remaining tests assert `UnsupportedSimulation` with a reason string containing `targetTeam`, `robot type`, or `bounds`.

- [ ] **Step 2: Run and witness the missing simulation package**

```powershell
python -m unittest tests.test_simulation_state -v
```

Expected: import failure for `future_war_agent.strategy.simulation.state`.

- [ ] **Step 3: Implement immutable simulation values**

Use these public shapes in `state.py`:

```python
class UnsupportedSimulation(ValueError):
    """The current observation cannot be simulated safely."""


@dataclass(frozen=True, slots=True)
class SimUnit:
    unit_id: int
    position: Position
    role_type: str
    health: int
    attack_power: int
    attack_range: int
    level: int | None
    cooldown: int


@dataclass(frozen=True, slots=True)
class SimRobot:
    robot_id: int
    position: Position
    role_type: str
    health: int


@dataclass(frozen=True, slots=True)
class SimState:
    round_no: int
    width: int
    height: int
    team_type: str
    station_id: int
    roles: tuple[SimUnit, ...]
    structures: tuple[SimUnit, ...]
    robots: tuple[SimRobot, ...]

    def entity(self, unit_id: int) -> SimUnit | None:
        return next(
            (item for item in self.roles + self.structures if item.unit_id == unit_id),
            None,
        )

    def robot(self, robot_id: int) -> SimRobot | None:
        return next((item for item in self.robots if item.robot_id == robot_id), None)

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height
```

`sim_state_from_observation` must sort every tuple by ID, require exactly one living station, validate every entity position, require explicit combat fields and valid positive combat values for all living weapons, include robots whose non-null `targetTeam` equals our `team_type`, ignore robots explicitly targeting another team, and reject a living robot with missing `targetTeam` or an unknown type. Add pure replacement helpers with these signatures:

```python
def with_units(
    self,
    *,
    roles: tuple[SimUnit, ...] | None = None,
    structures: tuple[SimUnit, ...] | None = None,
) -> "SimState":
    return replace(
        self,
        roles=self.roles if roles is None else tuple(sorted(roles, key=lambda x: x.unit_id)),
        structures=(
            self.structures
            if structures is None
            else tuple(sorted(structures, key=lambda x: x.unit_id))
        ),
    )

def with_robots(self, robots: tuple[SimRobot, ...]) -> "SimState":
    return replace(
        self,
        robots=tuple(sorted(robots, key=lambda x: x.robot_id)),
    )
```

Do not expose mutable lists or dictionaries.

- [ ] **Step 4: Add reusable simulation builders**

Create `tests/simulation_helpers.py` with `sim_unit`, `sim_robot`, and `sim_state` builders. The default state is a 15x15 night board, station ID 100 at `(5, 5)`, one worker, no weapons, and no robots. Builders return the frozen production dataclasses and accept explicit overrides for every field.

- [ ] **Step 5: Write failing geometry tests**

Create `tests/test_simulation_geometry.py`:

```python
class SimulationGeometryTests(unittest.TestCase):
    def test_axis_aligned_ray_extends_to_board_edge(self) -> None:
        cells = ray_cells(
            Position(2, 2), Position(4, 2), 7, 7, GeometryMode.BRESENHAM
        )
        self.assertEqual(cells, tuple(Position(x, 2) for x in range(3, 7)))

    def test_supercover_contains_bresenham_cells(self) -> None:
        basic = set(ray_cells(
            Position(1, 1), Position(4, 3), 8, 8, GeometryMode.BRESENHAM
        ))
        cover = set(ray_cells(
            Position(1, 1), Position(4, 3), 8, 8, GeometryMode.SUPERCOVER
        ))
        self.assertLessEqual(basic, cover)

    def test_conservative_hits_are_the_intersection(self) -> None:
        occupied = {Position(4, 2): 10, Position(4, 3): 11}
        hits = conservative_ray_hits(
            Position(1, 1), Position(4, 2), 8, 8, occupied
        )
        expected = set.intersection(
            *(
                set(occupied).intersection(
                    ray_cells(Position(1, 1), Position(4, 2), 8, 8, mode)
                )
                for mode in GeometryMode
            )
        )
        self.assertEqual(set(hits), expected)

    def test_cone_accepts_boundary_and_rejects_wider_pair(self) -> None:
        origin = Position(5, 5)
        self.assertTrue(within_ninety_degree_cone(
            origin, (Position(7, 5), Position(5, 7))
        ))
        self.assertFalse(within_ninety_degree_cone(
            origin, (Position(7, 5), Position(4, 7))
        ))
```

- [ ] **Step 6: Implement the three geometry modes**

Define:

```python
class GeometryMode(StrEnum):
    BRESENHAM = "bresenham"
    SUPERCOVER = "supercover"
    CELL_CENTER = "cell_center"
```

`ray_cells(origin, aim, width, height, mode)` must reject `origin == aim`, reduce the direction by `math.gcd`, extend the ray to the last in-bounds cell, then apply the selected deterministic grid traversal. Return unique cells in increasing distance from the origin and exclude the origin. `conservative_ray_hits` returns only occupied cells present in every mode, sorted by squared distance and entity ID. Implement cone checking without floating point by requiring every pair of direction vectors to have non-negative dot product; this includes the 90-degree boundary.

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m unittest tests.test_simulation_state tests.test_simulation_geometry -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: state and geometry tests pass without changing any Phase 2 decision test.

- [ ] **Step 8: Commit state and geometry**

```powershell
git add future_war_agent/strategy/simulation tests/simulation_helpers.py tests/test_simulation_state.py tests/test_simulation_geometry.py
git commit -m "feat: add immutable night simulation state"
```

---

### Task 3: Level-Aware Weapon Kernels and Reusable Joint Enumeration

**Files:**
- Create: `future_war_agent/strategy/simulation/weapons.py`
- Modify: `future_war_agent/strategy/joint.py`
- Modify: `future_war_agent/strategy/night.py`
- Modify: `future_war_agent/decision/validator.py`
- Create: `tests/test_simulation_weapons.py`
- Modify: `tests/test_joint.py`
- Modify: `tests/test_validator.py`

**Interfaces:**
- Consumes: `SimState`, conservative geometry, `TacticalCandidate`, and Phase 2 controller assignments.
- Produces: `WeaponAttack`, `weapon_attacks`, `weapon_damage`, `TacticalCandidate.attack_many`, and `enumerate_valid_joints(..., limit: int | None)`.

- [ ] **Step 1: Write failing weapon-kernel tests**

Create `tests/test_simulation_weapons.py` with these named cases:

```python
def test_gatling_requires_level_sized_cone_and_hits_nearest_robot(self) -> None:
    state = sim_state(
        structures=(sim_unit(
            20, 3, 3, "gatling", attack_power=10, attack_range=8, level=2
        ),),
        robots=(sim_robot(1, 5, 3, health=30), sim_robot(2, 7, 3, health=30),
                sim_robot(3, 3, 6, health=30)),
    )
    attacks = weapon_attacks(state, 20)
    self.assertTrue(all(len(item.targets) == 2 for item in attacks))
    damage = weapon_damage(state, attacks[0])
    self.assertNotIn((2, 10), damage.robot_damage)

def test_railgun_spends_energy_in_ray_order(self) -> None:
    state = sim_state(
        structures=(sim_unit(
            20, 2, 2, "railgun", attack_power=25, attack_range=9, level=3
        ),),
        robots=(sim_robot(1, 4, 2, health=10), sim_robot(2, 6, 2, health=30)),
    )
    attack = WeaponAttack(20, (Position(6, 2),))
    self.assertEqual(weapon_damage(state, attack).robot_damage, ((1, 10), (2, 15)))

def test_rocket_stacks_center_and_adjacent_damage(self) -> None:
    state = sim_state(
        structures=(sim_unit(
            20, 2, 2, "rocket", attack_power=20, attack_range=9, level=2
        ),),
        robots=(sim_robot(1, 5, 5, health=50), sim_robot(2, 6, 5, health=50)),
    )
    attack = WeaponAttack(20, (Position(5, 5), Position(6, 5)))
    self.assertEqual(weapon_damage(state, attack).robot_damage, ((1, 30), (2, 30)))

def test_cooldown_weapon_has_no_attack_candidates(self) -> None:
    state = sim_state(
        structures=(sim_unit(
            20, 2, 2, "rocket", attack_power=20, attack_range=9,
            level=1, cooldown=1,
        ),),
        robots=(sim_robot(1, 5, 5),),
    )
    self.assertEqual(weapon_attacks(state, 20), ())
```

Also test: target tuples are distinct, in range, stably ordered, limited to three, Gatling rejects a cone wider than 90 degrees, and a conservative ray never credits damage that only one geometry mode sees.

- [ ] **Step 2: Run and witness the missing weapons module**

```powershell
python -m unittest tests.test_simulation_weapons -v
```

Expected: import failure for `simulation.weapons`.

- [ ] **Step 3: Implement weapon values and damage ledgers**

Use these public values:

```python
@dataclass(frozen=True, slots=True)
class WeaponAttack:
    weapon_id: int
    targets: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class WeaponDamage:
    robot_damage: tuple[tuple[int, int], ...]
    immediate_kills: int
    total_damage: int
```

`weapon_attacks(state, weapon_id, limit=3)` generates target tuples from living threatening robot positions, applies range/count/cone/cooldown rules, deduplicates tuples, computes conservative immediate damage, and sorts by `(-immediate_kills, -total_damage, target coordinates)` before truncation. Railgun always has one target regardless of level; Gatling and rocket require exactly the positive integer `level` and produce no attack if too few distinct robot positions exist.

`weapon_damage` implements:

- Gatling: one conservative ray per target and only the nearest robot on that ray receives `attack_power`.
- Railgun: one conservative ray, ordered nearest first, with `min(remaining_energy, current_health)` damage and exact energy depletion.
- Rocket: full power at each center, `attack_power // rocket_splash_divisor` at Chebyshev distance one, and additive overlap.

Return sorted immutable `(robot_id, damage)` tuples and cap credited damage at each robot's current health only when computing `immediate_kills` and `total_damage`; retain stacked ledger damage for end-of-turn simultaneous resolution.

- [ ] **Step 4: Write failing multi-target joint tests**

Add tests that construct an adjacent controller and a level-two Gatling candidate:

```python
def test_level_two_gatling_joint_accepts_two_targets(self) -> None:
    world, observed = self._night_world_with_weapon("gatling", level=2)
    role = world.friendly_roles[0]
    weapon = world.weapons[0]
    candidate = TacticalCandidate.attack_many(
        role.unit_id,
        role.position,
        weapon.unit_id,
        (Position(7, 5), Position(5, 7)),
        priority=500,
    )
    self.assertTrue(is_valid_joint(observed, world, (candidate,)))

def test_joint_enumerator_is_stable_and_capped(self) -> None:
    choices = {
        10010: (
            candidate(10010, Position(1, 1), Position(1, 2)),
            candidate(10010, Position(1, 1), None),
        ),
        10011: (
            candidate(10011, Position(2, 1), Position(2, 2)),
            candidate(10011, Position(2, 1), None),
        ),
    }
    first = enumerate_valid_joints(
        self.observed, self.world, choices, limit=2
    )
    second = enumerate_valid_joints(
        self.observed, self.world, choices, limit=2
    )
    self.assertEqual(first, second)
    self.assertLessEqual(len(first), 2)
```

Define the referenced helper in `JointSolverTests`:

```python
def _night_world_with_weapon(
    self, role_type: str, *, level: int
) -> tuple[WorldGrid, Observation]:
    observed = observation(
        round_no=71,
        our_units=(
            unit(1, 4, 5, "worker"),
            unit(2, 1, 1, "station", level=1),
            unit(
                3, 5, 5, role_type, attack_power=20,
                attack_range=6, level=level, cooldown=0,
            ),
        ),
    )
    return WorldGrid.from_observation(observed), observed
```

Add negative tests for a one-target level-two Gatling, a two-target railgun, a wide Gatling cone, out-of-range targets, cooldown, and duplicate target cells.

- [ ] **Step 5: Extend joint candidates without changing Phase 2 behavior**

Add:

```python
@classmethod
def attack_many(
    cls,
    role_id: int,
    start: Position,
    weapon_id: int,
    targets: tuple[Position, ...],
    priority: int,
) -> "TacticalCandidate":
    return cls(
        role_id=role_id,
        command_actor_id=weapon_id,
        action=Action.attack(role_id, targets),
        job_kind=JobKind.ATTACK,
        start=start,
        priority=priority,
        completes_job=True,
        progress=1,
    )
```

Keep `attack(...)` as a compatibility wrapper that calls `attack_many(..., (target,), ...)`.

Extract enumeration from `solve_joint`:

```python
def enumerate_valid_joints(
    observation: Observation,
    world: WorldGrid,
    choices: Mapping[int, tuple[TacticalCandidate, ...]],
    *,
    limit: int | None = None,
) -> tuple[tuple[TacticalCandidate, ...], ...]:
    role_ids = tuple(sorted(choices))
    if not role_ids or any(not choices[role_id] for role_id in role_ids):
        return ()
    result: list[tuple[TacticalCandidate, ...]] = []
    for combination in product(*(choices[role_id] for role_id in role_ids)):
        joint = tuple(combination)
        if is_valid_joint(observation, world, joint):
            result.append(joint)
            if limit is not None and len(result) >= limit:
                break
    return tuple(result)
```

Make `solve_joint` score this returned tuple. Extend `_direct_action_is_legal` with strict target count, uniqueness, in-bounds/range, positive level, zero cooldown, and Gatling cone checks. Keep Phase 2's level-one `night.py` generation unchanged.

- [ ] **Step 6: Strengthen the outer validator for supported weapon semantics**

Add validator tests for target count, cooldown, range, duplicate targets, and Gatling cone. Update only the attack branch so a retained attack must have a living personal controller, exact Chebyshev adjacency of one, legal target count for the weapon type, unique in-bounds targets within `attack_range`, zero cooldown, and a valid Gatling cone. This is a Phase 1 safety check; it must not generate candidates or import simulation state.

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m unittest tests.test_simulation_weapons tests.test_joint tests.test_night tests.test_validator -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: weapon, joint, validator, and every original Phase 2 night test pass.

- [ ] **Step 8: Commit weapon and joint support**

```powershell
git add future_war_agent/strategy/simulation/weapons.py future_war_agent/strategy/joint.py future_war_agent/strategy/night.py future_war_agent/decision/validator.py tests/test_simulation_weapons.py tests/test_joint.py tests/test_validator.py
git commit -m "feat: support level-aware night attacks"
```

---

### Task 4: Four Deterministic Robot Scenario Policies

**Files:**
- Create: `future_war_agent/strategy/simulation/robots.py`
- Create: `tests/test_simulation_robots.py`

**Interfaces:**
- Consumes: immutable `SimState`, `SimRobot`, station footprint rules, and centralized robot damage.
- Produces: `RobotPolicy`, `RobotIntent`, `robot_intent`, and `ALL_ROBOT_POLICIES` in a fixed order.

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_simulation_robots.py` with one behavior test and one tie-break test per policy:

```python
def test_station_shortest_path_moves_toward_station(self) -> None:
    state = sim_state(robots=(sim_robot(1, 10, 5),))
    intent = robot_intent(state, 1, RobotPolicy.STATION_SHORTEST)
    self.assertEqual(intent.move_target, Position(9, 5))

def test_station_shortest_path_attacks_blocking_structure(self) -> None:
    state = sim_state(
        structures=(
            sim_unit(100, 5, 5, "station"),
            sim_unit(101, 8, 5, "wall", health=30),
        ),
        robots=(sim_robot(1, 9, 5),),
    )
    intent = robot_intent(state, 1, RobotPolicy.STATION_SHORTEST)
    self.assertEqual(intent.attack_unit_id, 101)

def test_blocker_policy_prefers_main_route_blocker(self) -> None:
    state = self._state_with_main_and_side_walls()
    intent = robot_intent(state, 1, RobotPolicy.MAIN_PATH_BLOCKER)
    self.assertEqual(intent.goal_unit_id, 101)

def test_low_health_policy_prefers_reachable_lowest_health(self) -> None:
    state = self._state_with_structure_healths((101, 20), (102, 5))
    intent = robot_intent(state, 1, RobotPolicy.LOW_HEALTH_STRUCTURE)
    self.assertEqual(intent.goal_unit_id, 102)

def test_max_progress_uses_stable_coordinate_tie_break(self) -> None:
    state = sim_state(robots=(sim_robot(1, 9, 9),))
    first = robot_intent(state, 1, RobotPolicy.MAX_STATION_PROGRESS)
    second = robot_intent(state, 1, RobotPolicy.MAX_STATION_PROGRESS)
    self.assertEqual(first, second)
```

Define the two fixture helpers in the same test class:

```python
def _state_with_main_and_side_walls(self) -> SimState:
    return sim_state(
        structures=(
            sim_unit(100, 5, 5, "station"),
            sim_unit(101, 8, 5, "wall", health=30),
            sim_unit(102, 8, 7, "wall", health=30),
        ),
        robots=(sim_robot(1, 9, 5),),
    )

def _state_with_structure_healths(
    self, *healths: tuple[int, int]
) -> SimState:
    positions = {101: Position(8, 5), 102: Position(8, 7)}
    structures = (sim_unit(100, 5, 5, "station"),) + tuple(
        sim_unit(unit_id, positions[unit_id].x, positions[unit_id].y,
                 "wall", health=health)
        for unit_id, health in healths
    )
    return sim_state(
        structures=structures,
        robots=(sim_robot(1, 9, 6),),
    )
```

Also assert every policy returns `WAIT` for a dead or missing robot and raises `UnsupportedSimulation` for an unknown robot type before damage lookup.

- [ ] **Step 2: Run and witness the missing robots module**

```powershell
python -m unittest tests.test_simulation_robots -v
```

Expected: import failure for `simulation.robots`.

- [ ] **Step 3: Implement public policy and intent values**

Use:

```python
class RobotPolicy(StrEnum):
    STATION_SHORTEST = "station_shortest"
    MAIN_PATH_BLOCKER = "main_path_blocker"
    LOW_HEALTH_STRUCTURE = "low_health_structure"
    MAX_STATION_PROGRESS = "max_station_progress"


ALL_ROBOT_POLICIES = (
    RobotPolicy.STATION_SHORTEST,
    RobotPolicy.MAIN_PATH_BLOCKER,
    RobotPolicy.LOW_HEALTH_STRUCTURE,
    RobotPolicy.MAX_STATION_PROGRESS,
)


class RobotIntentKind(StrEnum):
    MOVE = "move"
    ATTACK = "attack"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class RobotIntent:
    robot_id: int
    kind: RobotIntentKind
    move_target: Position | None = None
    attack_unit_id: int | None = None
    goal_unit_id: int | None = None
```

- [ ] **Step 4: Implement deterministic path and target selection**

Inside `robots.py`, implement an eight-direction BFS with neighbors sorted by `(x, y)`. For route selection, treat controlled structures as destructible path cells but treat other robots and board obstacles as blocked; if the selected next route cell contains a controlled structure, the robot attacks it instead of moving into it. When approaching a specifically chosen structure, stop at an adjacent cell. A robot attacks only a living structure at Chebyshev distance one; otherwise it moves one legal cell.

Implement the policies exactly as follows:

- `STATION_SHORTEST`: choose the shortest path to any station footprint cell. If the first blocked step contains a controlled structure, attack it; otherwise move along the first stable shortest step.
- `MAIN_PATH_BLOCKER`: for each living non-station structure, compare station path length with that structure removed. Prefer the structure whose removal produces the largest path improvement, then shorter robot-to-structure distance, then lower unit ID. Approach or attack it; if none helps, use `STATION_SHORTEST`.
- `LOW_HEALTH_STRUCTURE`: choose a reachable controlled structure by `(health, approach distance, unit_id)`, then approach or attack it.
- `MAX_STATION_PROGRESS`: choose the legal neighbor minimizing `(distance_to_station, x, y)` when it strictly improves distance. If no move improves distance, attack the adjacent structure with `(health, unit_id)` minimum or wait.

Call `rules.robot_attack_power(robot.role_type)` before returning an active intent so an unsupported robot type cannot enter the kernel.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_simulation_robots -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all four policy families and the existing suite pass deterministically.

- [ ] **Step 6: Commit robot scenarios**

```powershell
git add future_war_agent/strategy/simulation/robots.py tests/test_simulation_robots.py
git commit -m "feat: model deterministic robot scenarios"
```

---

### Task 5: Ordered Turn Kernel and Fixed Future Policy Primitives

**Files:**
- Create: `future_war_agent/strategy/simulation/kernel.py`
- Create: `tests/test_simulation_kernel.py`

**Interfaces:**
- Consumes: `SimState`, a root or future `Decision`, `RobotPolicy`, weapon damage ledgers, and `RulesConfig`.
- Produces: `TurnResult`, `step_simulation`, and deterministic role/robot collision resolution.

- [ ] **Step 1: Write failing turn-order tests**

Create `tests/test_simulation_kernel.py`:

```python
def test_weapon_damage_and_robot_damage_apply_simultaneously(self) -> None:
    state = self._adjacent_robot_and_ready_gatling(
        robot_health=10, station_health=40, robot_type="smallRobot"
    )
    decision = Decision(commands={
        20: Action.attack(1, (Position(7, 5),)),
    })

    result = step_simulation(state, decision, RobotPolicy.STATION_SHORTEST)

    self.assertIsNone(result.state.robot(30))
    self.assertEqual(result.state.entity(100).health, 35)

def test_role_moves_before_robot_policy_is_resolved(self) -> None:
    state = self._state_with_movable_controller()
    decision = Decision(commands={1: Action.move(Position(3, 4))})
    result = step_simulation(state, decision, RobotPolicy.MAX_STATION_PROGRESS)
    self.assertEqual(result.state.entity(1).position, Position(3, 4))

def test_fired_rocket_starts_full_three_round_cooldown(self) -> None:
    state = self._ready_rocket_state()
    decision = Decision(commands={
        20: Action.attack(1, (Position(7, 5),)),
    })
    first = step_simulation(state, decision, RobotPolicy.STATION_SHORTEST)
    second = step_simulation(first.state, Decision(), RobotPolicy.STATION_SHORTEST)
    self.assertEqual(first.state.entity(20).cooldown, 3)
    self.assertEqual(second.state.entity(20).cooldown, 2)

def test_conflicting_robot_moves_are_blocked_stably(self) -> None:
    state = self._two_robots_with_same_destination()
    result = step_simulation(state, Decision(), RobotPolicy.MAX_STATION_PROGRESS)
    self.assertEqual(result.state.robot(1).position, state.robot(1).position)
    self.assertEqual(result.state.robot(2).position, state.robot(2).position)
```

Define the kernel fixtures exactly:

```python
def _adjacent_robot_and_ready_gatling(
    self, *, robot_health: int, station_health: int, robot_type: str
) -> SimState:
    return sim_state(
        roles=(sim_unit(1, 5, 4, "worker"),),
        structures=(
            sim_unit(100, 5, 5, "station", health=station_health),
            sim_unit(
                20, 6, 4, "gatling", attack_power=10,
                attack_range=6, level=1, cooldown=0,
            ),
        ),
        robots=(sim_robot(
            30, 7, 5, role_type=robot_type, health=robot_health
        ),),
    )

def _state_with_movable_controller(self) -> SimState:
    return sim_state(
        roles=(sim_unit(1, 2, 4, "worker"),),
        structures=(sim_unit(100, 5, 5, "station"),),
        robots=(sim_robot(30, 12, 12),),
    )

def _ready_rocket_state(self) -> SimState:
    return sim_state(
        roles=(sim_unit(1, 5, 4, "worker"),),
        structures=(
            sim_unit(100, 5, 5, "station"),
            sim_unit(
                20, 6, 4, "rocket", attack_power=20,
                attack_range=6, level=1, cooldown=0,
            ),
        ),
        robots=(sim_robot(30, 7, 5),),
    )

def _two_robots_with_same_destination(self) -> SimState:
    return sim_state(
        structures=(sim_unit(100, 5, 5, "station"),),
        robots=(sim_robot(1, 7, 4), sim_robot(2, 7, 6)),
    )
```

Also test: dead entities disappear after the ledger is committed, a cooldown weapon command is rejected with `UnsupportedSimulation`, non-rocket cooldowns decrement but never become negative, role moves cannot overlap or swap, and `round_no` advances exactly once.

- [ ] **Step 2: Run and witness the missing kernel module**

```powershell
python -m unittest tests.test_simulation_kernel -v
```

Expected: import failure for `simulation.kernel`.

- [ ] **Step 3: Implement turn result and validation**

Use:

```python
@dataclass(frozen=True, slots=True)
class TurnResult:
    state: SimState
    destroyed_unit_ids: tuple[int, ...]
    destroyed_robot_ids: tuple[int, ...]
    station_damage: int
    immediate_robot_kills: int
```

Before applying a decision, validate that every command actor exists and is alive, attack commands identify an adjacent living personal controller, and every attack satisfies the same type/count/range/cooldown/cone rules as root generation. Raise `UnsupportedSimulation` instead of silently accepting an impossible simulated action.

- [ ] **Step 4: Implement the seven-stage kernel**

`step_simulation(state, decision, policy, rules=DEFAULT_RULES)` performs:

1. Build robot damage from all valid weapon attacks without mutating health.
2. Collect role moves, reject out-of-bounds/blocked/swap/duplicate destinations, and apply the remaining moves in role-ID order.
3. Compute one `RobotIntent` for every robot alive at turn start, using the post-role-move state. A robot pending lethal weapon damage still acts this turn.
4. Resolve robot move conflicts: if two robots choose one destination, neither moves; occupied destinations and swaps are blocked. Add attacks to the structure damage ledger using centralized robot power.
5. Apply both damage ledgers simultaneously and remove entities whose resulting health is at most zero.
6. Set a fired rocket to `rocket_cooldown_rounds`; decrement positive cooldowns only on weapons that did not fire this turn.
7. Increment `round_no` once and return stable sorted destruction IDs and metrics.

Use local dictionaries only inside the pure function and construct a new frozen `SimState` at the end.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_simulation_kernel -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all turn-order, simultaneous-damage, collision, and cooldown cases pass.

- [ ] **Step 6: Commit the turn kernel**

```powershell
git add future_war_agent/strategy/simulation/kernel.py tests/test_simulation_kernel.py
git commit -m "feat: add deterministic combat turn kernel"
```

---

### Task 6: Survival Certificates and Fixed-Budget Root Search

**Files:**
- Create: `future_war_agent/strategy/simulation/certificate.py`
- Create: `future_war_agent/strategy/simulation/search.py`
- Create: `tests/test_survival_certificate.py`
- Create: `tests/test_simulation_search.py`

**Interfaces:**
- Consumes: Phase 2 controller assignment and joint enumeration, `SimState`, four robot policies, exact scenario weights, an absolute deadline, and an injected clock.
- Produces: `ScenarioWeights`, `uniform_scenario_weights`, `SafetyClass`, `ScenarioOutcome`, `SurvivalCertificate`, `SearchStats`, `SearchResult`, `DeadlineExceeded`, and `search_night`.

- [ ] **Step 1: Write failing certificate tests**

Create `tests/test_survival_certificate.py`:

```python
def test_safe_requires_survival_in_every_scenario(self) -> None:
    outcomes = (
        ScenarioOutcome(Fraction(1, 2), 20, 2, 0, 12, 1),
        ScenarioOutcome(Fraction(1, 2), 10, 1, 1, 18, 0),
    )
    certificate = build_certificate(outcomes)
    self.assertEqual(certificate.safety, SafetyClass.SAFE)
    self.assertEqual(certificate.survival_probability, Fraction(1, 1))
    self.assertEqual(certificate.worst_station_health, 10)

def test_marginal_and_p10_use_exact_weights(self) -> None:
    outcomes = (
        ScenarioOutcome(Fraction(9, 10), 30, 2, 0, 5, 2),
        ScenarioOutcome(Fraction(1, 10), 0, 0, 3, 40, 0),
    )
    certificate = build_certificate(outcomes)
    self.assertEqual(certificate.safety, SafetyClass.MARGINAL)
    self.assertEqual(certificate.survival_probability, Fraction(9, 10))
    self.assertEqual(certificate.p10_station_health, 0)

def test_all_lost_is_unsafe(self) -> None:
    outcomes = tuple(
        ScenarioOutcome(Fraction(1, 4), 0, 0, 2, 20, 0)
        for _ in range(4)
    )
    self.assertEqual(build_certificate(outcomes).safety, SafetyClass.UNSAFE)

def test_no_valid_outcomes_is_unknown(self) -> None:
    certificate = build_certificate(())
    self.assertEqual(certificate.safety, SafetyClass.UNKNOWN)
    self.assertEqual(certificate.survival_probability, Fraction())
```

Add a test that `rank_key(action_key)` orders by survival probability, p10 health, worst health, surviving weapons, expected health, lower threat, greater immediate kills, then lexicographically smaller stable action key.

- [ ] **Step 2: Implement exact certificate aggregation**

Use `Fraction` throughout:

```python
class SafetyClass(StrEnum):
    SAFE = "safe"
    MARGINAL = "marginal"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    weight: Fraction
    station_health: int
    surviving_weapons: int
    structure_losses: int
    remaining_threat: int
    immediate_kills: int


@dataclass(frozen=True, slots=True)
class SurvivalCertificate:
    safety: SafetyClass
    survival_probability: Fraction
    expected_station_health: Fraction
    p10_station_health: int
    worst_station_health: int
    surviving_weapons: int
    structure_losses: int
    remaining_threat: int
    immediate_kills: int
    outcomes: tuple[ScenarioOutcome, ...]


ScenarioWeights = tuple[Fraction, Fraction, Fraction, Fraction]


def uniform_scenario_weights() -> ScenarioWeights:
    return (Fraction(1, 4),) * 4
```

For no outcomes, return an UNKNOWN certificate with zero metrics and an empty outcome tuple; search treats it as unsupported rather than ranking it. Otherwise normalize weights exactly, calculate weighted expected health, and implement p10 as the first station health whose ascending cumulative weight reaches `1/10`. Aggregate surviving weapons with the minimum, structure losses and remaining threat with the maximum, and immediate kills with the weighted floor. `rank_key` returns an ascending tuple of negated desired metrics followed by the stable action key.

- [ ] **Step 3: Write failing bounded-search tests**

Create `tests/test_simulation_search.py` with dependency-injected clock and small fixture states:

```python
def test_search_never_exceeds_fixed_budget(self) -> None:
    result = search_night(
        self.supported_observation,
        uniform_scenario_weights(),
        clock=lambda: 0.0,
        deadline=1.0,
    )
    self.assertLessEqual(result.stats.root_actions, 64)
    self.assertEqual(result.stats.scenarios_per_root, 4)
    self.assertLessEqual(result.stats.max_steps_per_scenario, 6)

def test_search_prefers_action_with_better_lower_tail_station_health(self) -> None:
    result = search_night(
        self.rocket_tradeoff_observation,
        uniform_scenario_weights(),
        clock=lambda: 0.0,
        deadline=1.0,
    )
    self.assertEqual(result.decision.commands[20].kind, ActionKind.ATTACK)
    self.assertGreater(result.certificate.p10_station_health, 0)

def test_all_unsafe_returns_least_bad_fully_evaluated_action(self) -> None:
    result = search_night(
        self.losing_observation,
        uniform_scenario_weights(),
        clock=lambda: 0.0,
        deadline=1.0,
    )
    self.assertEqual(result.certificate.safety, SafetyClass.UNSAFE)
    self.assertIsInstance(result.decision, Decision)

def test_deadline_discards_search_instead_of_returning_partial_best(self) -> None:
    ticks = iter((0.0, 0.1, 0.2, 1.1))
    with self.assertRaises(DeadlineExceeded):
        search_night(
            self.supported_observation,
            uniform_scenario_weights(),
            clock=lambda: next(ticks, 1.1),
            deadline=1.0,
        )
```

Wrap the tests in `SimulationSearchTests(unittest.TestCase)` and define exact observations:

```python
COMBAT_FIELDS = frozenset({"attackPower", "attackRange", "level", "cooldown"})

def setUp(self) -> None:
    self.supported_observation = self._observed("gatling", station_health=100)
    self.rocket_tradeoff_observation = self._observed(
        "rocket", station_health=30
    )
    self.losing_observation = self._observed(
        "gatling", station_health=5, robot_type="bossRobot"
    )

def _observed(
    self,
    weapon_type: str,
    *,
    station_health: int,
    robot_type: str = "smallRobot",
) -> Observation:
    return observation(
        round_no=71,
        our_units=(
            unit(1, 5, 4, "worker"),
            unit(100, 5, 5, "station", health=station_health, level=1),
            unit(
                20, 6, 4, weapon_type,
                attack_power=20 if weapon_type == "rocket" else 10,
                attack_range=8, level=1, cooldown=0,
                provided_fields=COMBAT_FIELDS,
            ),
        ),
        robots=(
            robot(
                30, 8, 5, role_type=robot_type,
                health=40, target_team="challenger",
            ),
            robot(
                31, 8, 6, role_type="smallRobot",
                health=40, target_team="challenger",
            ),
        ),
    )
```

Also test deterministic replay, horizon truncation at night round 60, level-aware root attacks, duplicate-free roots, and a stable empty `Decision` root when no command is legal.

- [ ] **Step 4: Implement root generation and fixed future policy**

In `search.py`, expose:

```python
class DeadlineExceeded(RuntimeError):
    """The complete Phase 3 search did not finish before its deadline."""


@dataclass(frozen=True, slots=True)
class SearchStats:
    root_actions: int
    scenarios_per_root: int
    max_steps_per_scenario: int
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    decision: Decision
    certificate: SurvivalCertificate
    stats: SearchStats
```

Build Phase 3 role choices from `assign_controllers`: a role away from its unique stand gets the same deterministic movement candidates as Phase 2; an adjacent assigned controller gets up to three `weapon_attacks` converted to `attack_many` plus wait; unassigned roles use Phase 2 safe candidates. Call `enumerate_valid_joints(..., limit=rules.phase3_root_limit)` and convert joints to decisions. Preserve product order and add `Decision()` if enumeration is empty.

After a root decision, future turns use one fixed policy: roles continue toward their original assigned stands; an adjacent controller fires the first ranked ready attack; every other actor waits. Recompute weapon attacks from the current `SimState` so deaths and rocket cooldown affect later turns. Do not branch future decisions.

- [ ] **Step 5: Implement four-scenario rollout and selection**

`search_night(observation, scenario_weights, *, rules=DEFAULT_RULES, clock=monotonic, deadline)` must:

1. Check `clock() >= deadline` before root generation, every root, every scenario, and every simulated step; raise `DeadlineExceeded` immediately.
2. Convert the observation to `SimState` and compute horizon `min(rules.phase3_horizon, NIGHT_ROUNDS - observation.time.round_in_phase + 1)`.
3. Evaluate every root under `ALL_ROBOT_POLICIES` in its fixed order using the matching exact weight.
4. Derive final station health, surviving weapons, structure losses, remaining threat as the sum of living robot health, and root-turn immediate kills.
5. Build a certificate and select the minimum `certificate.rank_key(stable_decision_key(decision))`.
6. Return no value until every planned root/scenario/step completes; any exception propagates to the engine boundary.

`stable_decision_key` is a tuple sorted by actor ID containing action kind, controller ID or `-1`, target coordinate tuples, name or empty string, and quantity or zero. It must not call JSON serialization or depend on dictionary insertion order.

- [ ] **Step 6: Run focused and full tests**

```powershell
python -m unittest tests.test_survival_certificate tests.test_simulation_search -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: certificate ordering, budget counters, deadline behavior, deterministic replay, and all accumulated tests pass.

- [ ] **Step 7: Commit certificate and search**

```powershell
git add future_war_agent/strategy/simulation/certificate.py future_war_agent/strategy/simulation/search.py tests/test_survival_certificate.py tests/test_simulation_search.py
git commit -m "feat: rank bounded night survival rollouts"
```

---

### Task 7: Session Fingerprints and Exact Scenario-Weight Reconciliation

**Files:**
- Create: `future_war_agent/strategy/session.py`
- Create: `future_war_agent/strategy/reconcile.py`
- Create: `tests/test_strategy_session.py`
- Create: `tests/test_strategy_reconcile.py`

**Interfaces:**
- Consumes: consecutive observations, the previously returned decision, four robot policies, `ScenarioWeights`, `uniform_scenario_weights`, and `RulesConfig`.
- Produces: `observation_fingerprint`, `map_signature`, `StrategySession`, `SessionStore`, and `reconcile_scenario_weights`.

- [ ] **Step 1: Write failing session tests**

Create `tests/test_strategy_session.py`:

```python
def test_fingerprint_is_stable_and_changes_with_relevant_health(self) -> None:
    first = observation_fingerprint(self.observed)
    second = observation_fingerprint(self.observed)
    changed = observation_fingerprint(self._with_station_health(99))
    self.assertEqual(first, second)
    self.assertNotEqual(first, changed)

def test_map_signature_ignores_dynamic_robot_positions(self) -> None:
    moved = self._with_robot_position(Position(9, 9))
    self.assertEqual(map_signature(self.observed), map_signature(moved))

def test_store_isolates_team_ids(self) -> None:
    store = SessionStore()
    store.put("a", self._session("a"))
    store.put("b", self._session("b"))
    self.assertNotEqual(store.get("a"), store.get("b"))

def test_reset_removes_only_selected_team(self) -> None:
    store = SessionStore()
    store.put("a", self._session("a"))
    store.put("b", self._session("b"))
    store.reset("a")
    self.assertIsNone(store.get("a"))
    self.assertIsNotNone(store.get("b"))
```

Define the immutable fixture helpers in `StrategySessionTests`:

```python
def setUp(self) -> None:
    self.observed = observation(
        round_no=71,
        our_units=(unit(100, 5, 5, "station", health=100, level=1),),
        robots=(robot(1, 9, 5, target_team="challenger"),),
    )

def _with_station_health(self, health: int) -> Observation:
    units = tuple(
        replace(unit, health=health) if unit.role_type == "station" else unit
        for unit in self.observed.our.units
    )
    return replace(self.observed, our=replace(self.observed.our, units=units))

def _with_robot_position(self, position: Position) -> Observation:
    robots = tuple(
        replace(robot, position=position)
        if robot.robot_id == self.observed.robots[0].robot_id
        else robot
        for robot in self.observed.robots
    )
    return replace(self.observed, robots=robots)

def _session(self, team_id: str) -> StrategySession:
    observed = replace(
        self.observed,
        our=replace(self.observed.our, team_id=team_id),
    )
    return StrategySession(
        team_id=team_id,
        last_observation=observed,
        last_fingerprint=observation_fingerprint(observed),
        last_decision=Decision(),
        scenario_weights=uniform_scenario_weights(),
        static_signature=map_signature(observed),
    )
```

Also verify that fingerprinting sorts `last_action_results`, units, robots, zones, and commands independently of source insertion order.

- [ ] **Step 2: Implement immutable session records**

Use:

```python
@dataclass(frozen=True, slots=True)
class StrategySession:
    team_id: str
    last_observation: Observation
    last_fingerprint: str
    last_decision: Decision
    scenario_weights: ScenarioWeights
    static_signature: str


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, StrategySession] = {}

    def get(self, team_id: str) -> StrategySession | None:
        return self._sessions.get(team_id)

    def put(self, team_id: str, session: StrategySession) -> None:
        self._sessions[team_id] = session

    def reset(self, team_id: str) -> None:
        self._sessions.pop(team_id, None)
```

Build canonical nested tuples explicitly from protocol fields, sort all ID-keyed collections and mapping entries, then hash `repr(tuple_value).encode("utf-8")` with SHA-256. `map_signature` includes dimensions, sorted zones, station ID/position, and `team_type`; it excludes dynamic health, roles, robots, cooldowns, and round number.

- [ ] **Step 3: Write failing reconciliation tests**

Create `tests/test_strategy_reconcile.py`:

```python
def test_matching_policy_gains_weight_without_eliminating_others(self) -> None:
    updated = reconcile_scenario_weights(
        self.previous,
        self.current_matching_station_shortest,
        Decision(),
        uniform_scenario_weights(),
    )
    self.assertGreater(updated[0], Fraction(1, 4))
    self.assertTrue(all(weight >= Fraction(1, 20) for weight in updated))
    self.assertEqual(sum(updated, Fraction()), Fraction(1, 1))

def test_no_shared_evidence_leaves_weights_unchanged(self) -> None:
    weights = (
        Fraction(2, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)
    )
    self.assertEqual(
        reconcile_scenario_weights(
            self.previous_without_robots,
            self.current_without_robots,
            Decision(),
            weights,
        ),
        weights,
    )

def test_unsupported_prediction_leaves_weights_unchanged(self) -> None:
    weights = uniform_scenario_weights()
    self.assertEqual(
        reconcile_scenario_weights(
            self.previous_missing_weapon_fields,
            self.current,
            Decision(),
            weights,
        ),
        weights,
    )
```

Wrap the tests in `StrategyReconcileTests(unittest.TestCase)` and define:

```python
def setUp(self) -> None:
    self.previous = observation(
        round_no=71,
        our_units=(unit(100, 5, 5, "station", health=100, level=1),),
        robots=(robot(1, 9, 5, target_team="challenger"),),
    )
    self.current_matching_station_shortest = observation(
        round_no=72,
        our_units=(unit(100, 5, 5, "station", health=100, level=1),),
        robots=(robot(1, 8, 5, target_team="challenger"),),
    )
    self.previous_without_robots = replace(self.previous, robots=())
    self.current_without_robots = replace(
        self.current_matching_station_shortest, robots=()
    )
    self.current = self.current_matching_station_shortest
    self.previous_missing_weapon_fields = observation(
        round_no=71,
        our_units=(
            unit(100, 5, 5, "station", health=100, level=1),
            unit(
                20, 5, 4, "gatling", attack_power=10,
                attack_range=6, level=1, cooldown=0,
                provided_fields={"attackPower", "attackRange", "level"},
            ),
        ),
        robots=(robot(1, 9, 5, target_team="challenger"),),
    )
```

Add cases for alive/dead mismatch and controlled-structure health-delta mismatch.

- [ ] **Step 4: Implement exact deterministic calibration**

`uniform_scenario_weights()` returns four `Fraction(1, 4)` values in `ALL_ROBOT_POLICIES` order.

For each policy, predict one turn from the previous observation and previous decision using `step_simulation`. Compare only IDs present in either the predicted or current relevant sets:

- add Chebyshev position error for each robot alive in both states;
- add 20 for each robot alive/dead mismatch;
- add absolute error between predicted and observed health delta for each controlled structure present in both observations.

If there is no comparable robot or structure evidence, or conversion/simulation is unsupported, return the original weights. Otherwise compute `raw_i = old_i / (1 + loss_i)`, then enforce the exact floor without iterative floating-point clipping:

```python
floor = Fraction(
    rules.scenario_weight_floor_numerator,
    rules.scenario_weight_floor_denominator,
)
free_mass = Fraction(1, 1) - floor * len(ALL_ROBOT_POLICIES)
total_raw = sum(raw, Fraction())
updated = tuple(floor + free_mass * value / total_raw for value in raw)
```

Require the result to sum to exactly one.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_strategy_session tests.test_strategy_reconcile -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: fingerprints, reset isolation, exact fractions, evidence handling, and all accumulated tests pass.

- [ ] **Step 6: Commit session and reconciliation**

```powershell
git add future_war_agent/strategy/session.py future_war_agent/strategy/reconcile.py tests/test_strategy_session.py tests/test_strategy_reconcile.py
git commit -m "feat: calibrate process-local robot scenarios"
```

---

### Task 8: Stateful Strategy Engine, Atomic Fallback, and Production Integration

**Files:**
- Create: `future_war_agent/strategy/engine.py`
- Modify: `future_war_agent/controller.py`
- Create: `tests/test_strategy_engine.py`
- Modify: `tests/test_controller.py`
- Create: `tests/fixtures/phase3_day_request.json`
- Create: `tests/fixtures/phase3_night_request.json`

**Interfaces:**
- Consumes: stateless Phase 2 planner, session store, reconciler, bounded search, monotonic clock, and existing controller validation.
- Produces: `StrategyEngine.plan(observation) -> Decision` and a process-level production engine used only by default_planner.

- [ ] **Step 1: Write failing lifecycle and cache tests**

Create `tests/test_strategy_engine.py` with injected Phase 2 and Phase 3 callables that record calls:

```python
def test_first_seen_night_uses_phase2_then_consecutive_night_uses_phase3(self) -> None:
    engine, calls = self._recording_engine()
    first = engine.plan(self._night_observation(71))
    second = engine.plan(self._night_observation(72))
    self.assertEqual(first, self.phase2_decision)
    self.assertEqual(second, self.phase3_decision)
    self.assertEqual(calls, ["phase2", "phase3"])

def test_day_records_continuity_and_first_night_can_use_phase3(self) -> None:
    engine, calls = self._recording_engine()
    engine.plan(self._day_observation(70))
    result = engine.plan(self._night_observation(71))
    self.assertEqual(result, self.phase3_decision)
    self.assertEqual(calls, ["phase2", "phase3"])

def test_duplicate_observation_returns_cached_decision_without_recalibration(self) -> None:
    engine, calls = self._recording_engine()
    observed = self._day_observation(70)
    first = engine.plan(observed)
    second = engine.plan(observed)
    self.assertIs(first, second)
    self.assertEqual(calls, ["phase2"])

def test_same_round_revision_replans_without_transition_update(self) -> None:
    engine, calls = self._recording_engine()
    engine.plan(self._day_observation(70, station_health=100))
    engine.plan(self._day_observation(70, station_health=90))
    self.assertEqual(calls, ["phase2", "phase2"])
    self.assertEqual(engine.session_for_tests("team").scenario_weights,
                     uniform_scenario_weights())
```

Add tests for round rollback, a gap larger than one, map/station signature change, two different team IDs, and missing/blank team ID. Rollback/gap/signature change must reset to uniform weights and use Phase 2 for that request.

- [ ] **Step 2: Write failing atomic-fallback and concurrency tests**

Add:

```python
def test_unsupported_state_falls_back_to_phase2(self) -> None:
    engine, calls = self._recording_engine(
        search_error=UnsupportedSimulation("missing cooldown")
    )
    engine.plan(self._day_observation(70))
    result = engine.plan(self._night_observation(71))
    self.assertEqual(result, self.phase2_decision)
    self.assertEqual(calls, ["phase2", "phase3", "phase2"])

def test_deadline_discards_partial_phase3_result(self) -> None:
    engine, calls = self._recording_engine(search_error=DeadlineExceeded())
    engine.plan(self._day_observation(70))
    result = engine.plan(self._night_observation(71))
    self.assertEqual(result, self.phase2_decision)
    self.assertEqual(calls[-2:], ["phase3", "phase2"])

def test_unexpected_phase3_exception_also_falls_back(self) -> None:
    engine, calls = self._recording_engine(search_error=RuntimeError("boom"))
    engine.plan(self._day_observation(70))
    self.assertEqual(engine.plan(self._night_observation(71)),
                     self.phase2_decision)

def test_concurrent_duplicates_execute_search_once(self) -> None:
    engine, calls = self._recording_engine()
    engine.plan(self._day_observation(70))
    observed = self._night_observation(71)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(lambda _: engine.plan(observed), range(4)))
    self.assertTrue(all(item == self.phase3_decision for item in results))
    self.assertEqual(calls.count("phase3"), 1)
```

Wrap these tests in `StrategyEngineTests(unittest.TestCase)` and define the fixtures and recording engine exactly:

```python
COMBAT_FIELDS = frozenset({"attackPower", "attackRange", "level", "cooldown"})

def setUp(self) -> None:
    self.phase2_decision = Decision(
        commands={1: Action.move(Position(3, 3))}
    )
    self.phase3_decision = Decision(
        commands={20: Action.attack(1, (Position(8, 5),))}
    )
    self.phase3_certificate = build_certificate((
        ScenarioOutcome(Fraction(1, 4), 100, 1, 0, 20, 1),
        ScenarioOutcome(Fraction(1, 4), 100, 1, 0, 20, 1),
        ScenarioOutcome(Fraction(1, 4), 100, 1, 0, 20, 1),
        ScenarioOutcome(Fraction(1, 4), 100, 1, 0, 20, 1),
    ))

def _day_observation(
    self, round_no: int, *, station_health: int = 100
) -> Observation:
    return observation(
        round_no=round_no,
        our_units=(
            unit(1, 4, 4, "worker"),
            unit(100, 5, 5, "station", health=station_health, level=1),
            unit(
                20, 5, 4, "gatling", attack_power=10,
                attack_range=6, level=1, cooldown=0,
                provided_fields=COMBAT_FIELDS,
            ),
        ),
    )

def _night_observation(
    self, round_no: int, *, station_health: int = 100
) -> Observation:
    return replace(
        self._day_observation(round_no, station_health=station_health),
        robots=(robot(30, 8, 5, target_team="challenger"),),
    )

def _recording_engine(
    self, *, search_error: Exception | None = None
) -> tuple[StrategyEngine, list[str]]:
    calls: list[str] = []

    def phase2(_: Observation) -> Decision:
        calls.append("phase2")
        return self.phase2_decision

    def phase3(
        observation: Observation,
        weights: ScenarioWeights,
        *,
        rules: RulesConfig,
        clock: Callable[[], float],
        deadline: float,
    ) -> SearchResult:
        del observation, weights, rules, clock, deadline
        calls.append("phase3")
        if search_error is not None:
            raise search_error
        return SearchResult(
            decision=self.phase3_decision,
            certificate=self.phase3_certificate,
            stats=SearchStats(1, 4, 1, 4),
        )

    return (
        StrategyEngine(
            phase2_planner=phase2,
            night_searcher=phase3,
            clock=lambda: 0.0,
        ),
        calls,
    )
```

Also verify that if the injected Phase 2 planner raises, the exception propagates to the existing controller boundary rather than being replaced inside StrategyEngine.

- [ ] **Step 3: Implement the locked engine**

Use this constructor and public method:

```python
Phase2Planner = Callable[[Observation], Decision]
NightSearcher = Callable[..., SearchResult]


class StrategyEngine:
    def __init__(
        self,
        *,
        rules: RulesConfig = DEFAULT_RULES,
        phase2_planner: Phase2Planner = plan_turn,
        night_searcher: NightSearcher = search_night,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._rules = rules
        self._phase2_planner = phase2_planner
        self._night_searcher = night_searcher
        self._clock = clock
        self._sessions = SessionStore()
        self._lock = RLock()

    def plan(self, observation: Observation) -> Decision:
        with self._lock:
            return self._plan_locked(observation)
```

Implement _plan_locked in this order:

1. A blank team_id immediately calls Phase 2 without creating a session.
2. Compute fingerprint and static signature. If the stored session has the same round and fingerprint, return its cached decision object.
3. Treat round rollback, static-signature change, or round gap greater than one as discontinuity and start with uniform weights.
4. A same-round changed fingerprint keeps current weights, does not reconcile, replans, and replaces the cached observation.
5. A consecutive observation calls reconcile_scenario_weights before planning.
6. Day or discontinuous night calls Phase 2.
7. Continuous night computes deadline = clock() + rules.phase3_watchdog_seconds and calls the injected searcher.
8. Catch UnsupportedSimulation, DeadlineExceeded, and ordinary Phase 3 Exception, log the internal reason, discard the search result, and call Phase 2 once from the original observation.
9. Store exactly the decision actually returned, the current observation/fingerprint/signature, and the updated weights.

Expose session_for_tests(team_id) as a read-only method returning the frozen record; do not expose the mutable store.

- [ ] **Step 4: Add consecutive HTTP fixtures**

Create two complete raw fixtures with the same 15x15 map, teamId, teamOur.type, station, controller role, walls, and weapons:

- `phase3_day_request.json`: round 70, no robots, and every weapon explicitly contains attackPower, attackRange, level, and cooldown.
- `phase3_night_request.json`: round 71, the controller already adjacent to a ready weapon, and at least two visible robots with targetTeam equal to teamOur.type.

Keep all parser-required fields and empty arrays/maps explicit. The night fixture must be valid for a multi-target level-two weapon and must produce a non-empty legal decision under both Phase 2 fallback and Phase 3.

- [ ] **Step 5: Connect one process-level engine**

Modify `controller.py`:

```python
from future_war_agent.strategy.engine import StrategyEngine


DEFAULT_STRATEGY_ENGINE = StrategyEngine()


def default_planner(observation: Observation) -> Decision:
    return DEFAULT_STRATEGY_ENGINE.plan(observation)
```

Keep planner injection, validate_decision, serialization, exception logging, and safe_payload() unchanged.

- [ ] **Step 6: Add controller integration tests**

Load the day and night fixtures in `tests/test_controller.py`. Use a fresh StrategyEngine through the existing injected-planner argument so tests do not depend on the module-level engine's prior history:

```python
def test_consecutive_requests_enable_phase3_without_schema_change(self) -> None:
    engine = StrategyEngine()
    handle_payload(self.phase3_day_payload, planner=engine.plan)
    response = handle_payload(self.phase3_night_payload, planner=engine.plan)
    self.assertEqual(set(response), {"roleCommandMap", "prompt", "executeCmd"})
    self.assertTrue(response["roleCommandMap"])

def test_duplicate_night_payload_is_byte_equivalent(self) -> None:
    engine = StrategyEngine()
    handle_payload(self.phase3_day_payload, planner=engine.plan)
    first = handle_payload(self.phase3_night_payload, planner=engine.plan)
    second = handle_payload(self.phase3_night_payload, planner=engine.plan)
    self.assertEqual(first, second)
```

Add a fixture mutation that removes weapon cooldown; after the day request, the night response must equal an explicitly invoked Phase 2 response after both pass through validate_decision and decision_to_payload.

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m unittest tests.test_strategy_engine tests.test_controller tests.test_server -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: lifecycle, atomic fallback, concurrent duplicate, controller schema, server, and every accumulated test pass.

- [ ] **Step 8: Commit the production engine**

```powershell
git add future_war_agent/strategy/engine.py future_war_agent/controller.py tests/test_strategy_engine.py tests/test_controller.py tests/fixtures/phase3_day_request.json tests/fixtures/phase3_night_request.json
git commit -m "feat: integrate stateful phase 3 strategy engine"
```

---

### Task 9: Phase 3 Acceptance Verification

**Files:**
- Verify: all files under `future_war_agent/` and `tests/`
- Modify only if a focused failing test demonstrates an acceptance defect.

**Interfaces:**
- Consumes: the complete service and both consecutive Phase 3 HTTP fixtures.
- Produces: fresh evidence for every Phase 3 acceptance criterion while preserving all Phase 1 and Phase 2 behavior.

- [ ] **Step 1: Run the complete standard-library suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: more than the original 72 tests run; the final unittest summary is OK with zero failures and zero errors. Logged exceptions from intentional controller/server failure tests are acceptable only when their tests pass.

- [ ] **Step 2: Compile all Python sources**

```powershell
python -m compileall -q main.py future_war_agent tests
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Audit dependency and placeholder constraints**

```powershell
rg -n "^(import|from) " future_war_agent
rg -n "TODO|TBD|PLACEHOLDER|NotImplementedError|pass$" future_war_agent tests
```

Expected: imports are standard-library or internal project modules. The placeholder scan has no Phase 3 implementation hit; a deliberate test double may use pass only when the surrounding test proves its purpose.

- [ ] **Step 4: Run consecutive real HTTP requests**

Start the service in one terminal:

```powershell
python main.py 18080
```

In a second terminal:

```powershell
$day = Get-Content -LiteralPath 'tests/fixtures/phase3_day_request.json' -Raw
$night = Get-Content -LiteralPath 'tests/fixtures/phase3_night_request.json' -Raw
$dayResponse = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:18080/' -ContentType 'application/json' -Body $day
$nightResponse1 = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:18080/' -ContentType 'application/json' -Body $night
$nightResponse2 = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:18080/' -ContentType 'application/json' -Body $night
$nightResponse1 | ConvertTo-Json -Depth 20
if (($nightResponse1 | ConvertTo-Json -Depth 20 -Compress) -ne ($nightResponse2 | ConvertTo-Json -Depth 20 -Compress)) { throw 'duplicate response changed' }
```

Expected: all requests return HTTP 200; both night responses contain the same non-empty roleCommandMap, empty prompt, and empty executeCmd. Stop the server with Ctrl+C.

- [ ] **Step 5: Re-run the fallback and watchdog proofs**

```powershell
python -m unittest tests.test_strategy_engine.StrategyEngineTests.test_unsupported_state_falls_back_to_phase2 -v
python -m unittest tests.test_strategy_engine.StrategyEngineTests.test_deadline_discards_partial_phase3_result -v
python -m unittest tests.test_strategy_engine.StrategyEngineTests.test_unexpected_phase3_exception_also_falls_back -v
```

Expected: all three pass and their injected call logs show Phase 2 is called after the failed Phase 3 attempt; no partial search decision is returned.

- [ ] **Step 6: Verify deterministic budget and concurrency proofs**

```powershell
python -m unittest tests.test_simulation_search.SimulationSearchTests.test_search_never_exceeds_fixed_budget -v
python -m unittest tests.test_simulation_search.SimulationSearchTests.test_deadline_discards_search_instead_of_returning_partial_best -v
python -m unittest tests.test_strategy_engine.StrategyEngineTests.test_concurrent_duplicates_execute_search_once -v
```

Expected: all three pass with at most 64 roots, exactly four scenarios per evaluated root, at most six steps per scenario, atomic deadline failure, and one search for concurrent duplicate requests.

- [ ] **Step 7: Check repository integrity**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -12
```

Expected: no whitespace errors, no uncommitted files, and HEAD on codex/phase3-night-simulation with the Phase 3 implementation commits above the design and plan commits.

- [ ] **Step 8: Compare implementation with the spec**

Confirm each item with a named test or HTTP probe:

- explicit combat-field presence gates Phase 3 without changing Phase 2 defaults;
- only visible robots targeting our.team_type enter simulation;
- Gatling, railgun, and rocket obey level, range, cone, ray, splash, overlap, penetration, and cooldown rules;
- weapon fire, role movement, robot intent, simultaneous damage, deaths, and cooldowns occur in the specified order;
- all four deterministic robot scenarios are evaluated with exact calibrated weights;
- search never exceeds 64 roots, four scenarios, or six turns;
- certificates and lexicographic ranking select a deterministic action, including all-unsafe positions;
- duplicate, revised, gap, rollback, map-change, and multi-team session cases behave as specified;
- timeout, unsupported state, and internal simulation failure discard Phase 3 and invoke Phase 2;
- the controller still validates, serializes, and safely catches an outer planner failure.

If an item lacks proof, add one focused failing test for that exact item, witness the failure, make the minimum correction, rerun Steps 1 through 7, and commit with:

```powershell
git commit -m "fix: satisfy phase 3 acceptance"
```

## Requirement Traceability

- Spec sections 5-6 (session lifecycle and eligibility): Tasks 1, 2, 7, and 8.
- Spec sections 7-9 (kernel, weapons, and legal joints): Tasks 2 through 5.
- Spec section 10 (robot scenarios and calibration): Tasks 4 and 7.
- Spec sections 11-13 (bounded search, certificate, and watchdog): Tasks 6 and 8.
- Spec sections 14-15 (fallback and module boundaries): Tasks 7 and 8.
- Spec sections 16-17 (verification and acceptance): Task 9.
