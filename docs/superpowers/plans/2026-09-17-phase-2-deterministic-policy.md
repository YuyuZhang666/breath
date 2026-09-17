# Phase 2 Deterministic Playable Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the empty default planner with a deterministic playable policy that can navigate, mine, sell, build a basic defense, recall before night, position controllers, and make conservative level-one attacks without creating friendly movement collisions.

**Architecture:** Convert each immutable `Observation` into an immutable `WorldGrid`, generate base-relative defensive layout and prioritized jobs, use deterministic A* to produce a small candidate set for each controllable role, and exactly enumerate at most 64 joint choices. The resulting `Decision` continues through the existing Phase 1 validator, serializer, and HTTP failure boundary.

**Tech Stack:** Python 3.11+ standard library, frozen dataclasses, `heapq`, `itertools`, `unittest`, existing `ThreadingHTTPServer` integration.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-2-deterministic-policy-design.md`

## Global Constraints

- Runtime and tests use only the Python standard library.
- Strategy consumes immutable `Observation` values and produces `Decision` values; raw request dictionaries never enter strategy code.
- Phase 1 validation, serialization, HTTP handling, and safe fallback remain the final authority and failure boundary.
- Planning is request-local and contains no mutable cross-request state.
- Identical observations must produce identical decisions.
- Formal interface documentation takes precedence over demo constants; rule differences are isolated in `RulesConfig`.
- Phase 2 does not implement beam search, simulation, persistent beliefs, tasks, upgrades, items, summons, exact multi-level combat, opponent modeling, or learning.
- Use TDD for every implementation task: add a focused failing test, run it and observe the expected failure, implement the minimum behavior, then run focused and full regression tests.
- Work directly in the current checkout on `codex/phase2-deterministic-policy`; do not create a worktree.

---

### Task 1: Central Rules and Immutable World Grid

**Files:**
- Create: `future_war_agent/strategy/__init__.py`
- Create: `future_war_agent/strategy/rules.py`
- Create: `future_war_agent/strategy/world.py`
- Create: `tests/strategy_helpers.py`
- Create: `tests/test_world.py`

**Interfaces:**
- Consumes: `Observation`, `Position`, `RobotState`, and `UnitState` from `future_war_agent.protocol.models`.
- Produces: `RulesConfig`, `DEFAULT_RULES`, `station_footprint(position, rules)`, `WorldGrid.from_observation(observation, rules)`, `WorldGrid.in_bounds`, `WorldGrid.unit_by_id`, `WorldGrid.positions_for_zone`, and `WorldGrid.interaction_cells`.

- [ ] **Step 1: Add reusable immutable observation builders**

Create `tests/strategy_helpers.py` with builders used by every strategy test:

```python
from collections.abc import Iterable

from future_war_agent.protocol.models import (
    EnemyTeamState,
    Observation,
    OurTeamState,
    Position,
    RobotState,
    ShopItem,
    UnitState,
    Zone,
)
from future_war_agent.protocol.time import TurnTime


def unit(
    unit_id: int,
    x: int,
    y: int,
    role_type: str,
    *,
    health: int = 100,
    attack_range: int = 0,
    backpack_capacity: int = 100,
    backpack: Iterable[str] = (),
    level: int | None = None,
    cooldown: int = 0,
) -> UnitState:
    return UnitState(
        unit_id=unit_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        attack_range=attack_range,
        backpack_capacity=backpack_capacity,
        backpack=tuple(backpack),
        level=level,
        cooldown=cooldown,
    )


def robot(
    robot_id: int,
    x: int,
    y: int,
    *,
    role_type: str = "smallRobot",
    health: int = 40,
    target_team: str | None = "challenger",
) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        position=Position(x, y),
        role_type=role_type,
        health=health,
        target_team=target_team,
    )


def observation(
    *,
    round_no: int = 1,
    width: int = 15,
    height: int = 15,
    our_units: Iterable[UnitState] = (),
    enemy_units: Iterable[UnitState] = (),
    robots: Iterable[RobotState] = (),
    zones: Iterable[Zone] = (),
    gold: int = 75,
    vendor_shop: Iterable[ShopItem] = (),
) -> Observation:
    return Observation(
        time=TurnTime.from_round(round_no),
        width=width,
        height=height,
        zones=tuple(zones),
        our=OurTeamState(
            team_type="challenger",
            team_id="team",
            gold=gold,
            units=tuple(our_units),
        ),
        enemy=EnemyTeamState(units=tuple(enemy_units)),
        robots=tuple(robots),
        phase_task="",
        vendor_shop=tuple(vendor_shop),
    )
```

- [ ] **Step 2: Write failing rules and world-grid tests**

Create `tests/test_world.py`:

```python
import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.rules import RulesConfig, station_footprint
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class WorldGridTests(unittest.TestCase):
    def test_station_footprint_uses_formal_top_left_convention(self) -> None:
        self.assertEqual(
            station_footprint(Position(4, 5)),
            frozenset(
                {
                    Position(4, 5),
                    Position(5, 5),
                    Position(4, 6),
                    Position(5, 6),
                }
            ),
        )

    def test_station_direction_is_configurable(self) -> None:
        rules = RulesConfig(station_y_direction=-1)
        self.assertEqual(
            station_footprint(Position(4, 5), rules),
            frozenset(
                {
                    Position(4, 5),
                    Position(5, 5),
                    Position(4, 4),
                    Position(5, 4),
                }
            ),
        )

    def test_builds_separate_hard_and_friendly_occupancy(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 5, 5, "station", level=1),
                unit(10020, 7, 5, "gatling", level=1),
            ),
            enemy_units=(unit(20010, 10, 10, "worker"),),
            robots=(robot(30001, 9, 9),),
            zones=(Zone(Position(3, 3), "stone"),),
        )

        world = WorldGrid.from_observation(observed)

        self.assertEqual(world.soft_friendly, frozenset({Position(2, 2)}))
        for blocked in (
            Position(3, 3),
            Position(5, 5),
            Position(6, 6),
            Position(7, 5),
            Position(9, 9),
            Position(10, 10),
        ):
            self.assertIn(blocked, world.hard_blocked)

    def test_interaction_cells_are_adjacent_and_traversable(self) -> None:
        target = Position(3, 3)
        world = WorldGrid.from_observation(
            observation(zones=(Zone(target, "stone"),))
        )

        cells = world.interaction_cells(target)

        self.assertEqual(len(cells), 8)
        self.assertTrue(all(cell.chebyshev_distance(target) == 1 for cell in cells))
        self.assertNotIn(target, cells)

    def test_diagonal_is_not_blocked_by_orthogonal_corners(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                zones=(
                    Zone(Position(2, 1), "stone"),
                    Zone(Position(1, 2), "iron"),
                )
            )
        )

        self.assertTrue(world.can_traverse(Position(2, 2)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the world tests and witness the import failure**

Run:

```powershell
python -m unittest tests.test_world -v
```

Expected: `ModuleNotFoundError` for `future_war_agent.strategy`.

- [ ] **Step 4: Implement central rules**

Create `future_war_agent/strategy/rules.py` with this public shape:

```python
from dataclasses import dataclass

from future_war_agent.protocol.models import Position


@dataclass(frozen=True, slots=True)
class RulesConfig:
    weapon_build_cost: int = 25
    wall_material: str = "stone"
    wall_material_cost: int = 1
    weapon_loadout: tuple[str, str, str] = ("gatling", "railgun", "rocket")
    twilight_safety_margin: int = 2
    station_y_direction: int = 1


DEFAULT_RULES = RulesConfig()


def station_footprint(
    position: Position,
    rules: RulesConfig = DEFAULT_RULES,
) -> frozenset[Position]:
    vertical = rules.station_y_direction
    return frozenset(
        {
            position,
            Position(position.x + 1, position.y),
            Position(position.x, position.y + vertical),
            Position(position.x + 1, position.y + vertical),
        }
    )
```

Reject a `station_y_direction` outside `{-1, 1}` in `RulesConfig.__post_init__`.

- [ ] **Step 5: Implement the immutable world grid**

Create `future_war_agent/strategy/world.py` around this frozen data shape:

```python
from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, Position, UnitState

from .rules import DEFAULT_RULES, RulesConfig, station_footprint


_ROLE_TYPES = frozenset({"worker", "pioneer"})
_WEAPON_TYPES = frozenset({"gatling", "railgun", "rocket"})


@dataclass(frozen=True, slots=True)
class WorldGrid:
    observation: Observation
    rules: RulesConfig
    neutral_cells: frozenset[Position]
    structure_cells: frozenset[Position]
    robot_cells: frozenset[Position]
    visible_enemy_role_cells: frozenset[Position]
    soft_friendly: frozenset[Position]
    friendly_roles: tuple[UnitState, ...]
    stations: tuple[UnitState, ...]
    weapons: tuple[UnitState, ...]
    walls: tuple[UnitState, ...]

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        rules: RulesConfig = DEFAULT_RULES,
    ) -> "WorldGrid":
        all_units = observation.our.units + observation.enemy.units
        structures = tuple(
            value
            for value in all_units
            if value.health > 0 and value.role_type not in _ROLE_TYPES
        )
        cells: set[Position] = set()
        for structure in structures:
            if structure.role_type == "station":
                cells.update(station_footprint(structure.position, rules))
            else:
                cells.add(structure.position)
        friendly_roles = tuple(
            sorted(
                (
                    value
                    for value in observation.our.units
                    if value.health > 0 and value.role_type in _ROLE_TYPES
                ),
                key=lambda value: value.unit_id,
            )
        )
        return cls(
            observation=observation,
            rules=rules,
            neutral_cells=frozenset(zone.position for zone in observation.zones),
            structure_cells=frozenset(cells),
            robot_cells=frozenset(
                value.position for value in observation.robots if value.health > 0
            ),
            visible_enemy_role_cells=frozenset(
                value.position
                for value in observation.enemy.units
                if value.health > 0 and value.role_type in _ROLE_TYPES
            ),
            soft_friendly=frozenset(value.position for value in friendly_roles),
            friendly_roles=friendly_roles,
            stations=tuple(
                value
                for value in observation.our.units
                if value.health > 0 and value.role_type == "station"
            ),
            weapons=tuple(
                sorted(
                    (
                        value
                        for value in observation.our.units
                        if value.health > 0 and value.role_type in _WEAPON_TYPES
                    ),
                    key=lambda value: value.unit_id,
                )
            ),
            walls=tuple(
                value
                for value in observation.our.units
                if value.health > 0 and value.role_type == "wall"
            ),
        )

    @property
    def hard_blocked(self) -> frozenset[Position]:
        return frozenset().union(
            self.neutral_cells,
            self.structure_cells,
            self.robot_cells,
            self.visible_enemy_role_cells,
        )
```

Add deterministic methods:

- `in_bounds(position)` checks observation dimensions.
- `can_traverse(position)` requires in-bounds and not hard-blocked.
- `unit_by_id(unit_id)` searches our units and returns `UnitState | None`.
- `positions_for_zone(kind)` returns positions sorted by `(x, y)`.
- `interaction_cells(target)` generates the eight neighbors, filters with
  `can_traverse`, and returns a tuple sorted by `(x, y)`.
- `our_station()` returns the lowest-ID living station or `None`.

- [ ] **Step 6: Run focused and full tests**

Run:

```powershell
python -m unittest tests.test_world -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all existing 29 tests plus 5 world tests pass.

- [ ] **Step 7: Commit the world model**

```powershell
git add future_war_agent/strategy tests/strategy_helpers.py tests/test_world.py
git commit -m "feat: model strategy world occupancy"
```

---

### Task 2: Deterministic Eight-Direction A* Pathfinding

**Files:**
- Create: `future_war_agent/strategy/pathfinding.py`
- Create: `tests/test_pathfinding.py`

**Interfaces:**
- Consumes: `WorldGrid`, `Position`, and an optional set of additional blocked cells.
- Produces: `PathResult`, `shortest_path`, `path_to_interaction`, and `first_step_options`.

- [ ] **Step 1: Write failing pathfinding tests**

Create `tests/test_pathfinding.py`:

```python
import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.pathfinding import (
    first_step_options,
    path_to_interaction,
    shortest_path,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation


class PathfindingTests(unittest.TestCase):
    def test_finds_deterministic_diagonal_shortest_path(self) -> None:
        world = WorldGrid.from_observation(observation())

        first = shortest_path(world, Position(1, 1), (Position(4, 4),))
        second = shortest_path(world, Position(1, 1), (Position(4, 4),))

        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        self.assertEqual(first.cost, 3)
        self.assertEqual(first.path[0], Position(1, 1))
        self.assertEqual(first.path[-1], Position(4, 4))

    def test_routes_around_hard_obstacles(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                zones=tuple(
                    Zone(Position(2, y), "stone") for y in range(1, 5)
                )
            )
        )

        result = shortest_path(world, Position(1, 2), (Position(3, 2),))

        self.assertIsNotNone(result)
        self.assertGreater(result.cost, 2)
        self.assertTrue(all(step not in world.hard_blocked for step in result.path))

    def test_returns_none_when_goal_is_unreachable(self) -> None:
        center = Position(2, 2)
        zones = tuple(
            Zone(Position(center.x + dx, center.y + dy), "stone")
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        )
        world = WorldGrid.from_observation(observation(zones=zones))

        self.assertIsNone(shortest_path(world, center, (Position(5, 5),)))

    def test_path_to_interaction_stops_adjacent_to_target(self) -> None:
        mine = Position(6, 6)
        world = WorldGrid.from_observation(
            observation(zones=(Zone(mine, "stone"),))
        )

        result = path_to_interaction(world, Position(1, 1), mine)

        self.assertIsNotNone(result)
        self.assertEqual(result.path[-1].chebyshev_distance(mine), 1)

    def test_first_step_options_are_distinct_and_stable(self) -> None:
        world = WorldGrid.from_observation(observation())

        options = first_step_options(
            world,
            Position(2, 2),
            (Position(6, 4),),
            limit=2,
        )

        self.assertEqual(options, tuple(dict.fromkeys(options)))
        self.assertLessEqual(len(options), 2)
        self.assertEqual(
            options,
            first_step_options(
                world,
                Position(2, 2),
                (Position(6, 4),),
                limit=2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_pathfinding -v
```

Expected: import failure for `strategy.pathfinding`.

- [ ] **Step 3: Implement deterministic A***

Create `future_war_agent/strategy/pathfinding.py` with:

```python
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from future_war_agent.protocol.models import Position

from .world import WorldGrid


_STEPS = tuple(
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)


@dataclass(frozen=True, slots=True)
class PathResult:
    path: tuple[Position, ...]
    cost: int


def shortest_path(
    world: WorldGrid,
    start: Position,
    goals: tuple[Position, ...],
    *,
    additional_blocked: frozenset[Position] = frozenset(),
) -> PathResult | None:
    valid_goals = frozenset(
        goal
        for goal in goals
        if world.in_bounds(goal)
        and (world.can_traverse(goal) or goal == start)
        and goal not in additional_blocked
    )
    if not valid_goals:
        return None
    order = count()
    frontier: list[tuple[int, int, int, int, int, Position]] = []
    heappush(frontier, (_heuristic(start, valid_goals), 0, start.x, start.y, next(order), start))
    came_from: dict[Position, Position] = {}
    best_cost = {start: 0}
    while frontier:
        _, cost, _, _, _, current = heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current in valid_goals:
            path = _reconstruct(came_from, start, current)
            return PathResult(path=path, cost=len(path) - 1)
        for dx, dy in _STEPS:
            neighbor = Position(current.x + dx, current.y + dy)
            if neighbor != start and (
                not world.can_traverse(neighbor)
                or neighbor in additional_blocked
            ):
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(neighbor, next_cost + 1):
                continue
            best_cost[neighbor] = next_cost
            came_from[neighbor] = current
            heappush(
                frontier,
                (
                    next_cost + _heuristic(neighbor, valid_goals),
                    next_cost,
                    neighbor.x,
                    neighbor.y,
                    next(order),
                    neighbor,
                ),
            )
    return None
```

Implement `_heuristic` as minimum Chebyshev distance, and `_reconstruct` as an
iterative reverse walk that includes start and goal. `path_to_interaction` calls
`world.interaction_cells(target)` and delegates to `shortest_path`.

Implement `first_step_options` by evaluating each traversable neighbor of
`start`, calculating its shortest remaining path to the goals, sorting by
`(1 + remaining_cost, x, y)`, removing duplicates, and returning at most
`limit`. This intentionally allows soft-friendly cells; Task 5 performs the
same-turn occupancy check.

- [ ] **Step 4: Run focused and full tests**

```powershell
python -m unittest tests.test_pathfinding -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 39 accumulated tests pass.

- [ ] **Step 5: Commit pathfinding**

```powershell
git add future_war_agent/strategy/pathfinding.py tests/test_pathfinding.py
git commit -m "feat: add deterministic strategy pathfinding"
```

---

### Task 3: Base-Relative Defensive Layout

**Files:**
- Create: `future_war_agent/strategy/layout.py`
- Create: `tests/test_layout.py`

**Interfaces:**
- Consumes: `WorldGrid`, `RulesConfig`, and the current own station.
- Produces: immutable `WeaponSite`, `DefensiveLayout`, and `build_defensive_layout(world)`.

- [ ] **Step 1: Write failing layout tests**

Create `tests/test_layout.py`:

```python
import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class DefensiveLayoutTests(unittest.TestCase):
    def test_selects_three_distinct_weapon_sites_in_loadout_order(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, "station", level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertEqual(
            tuple(site.weapon_type for site in layout.weapon_sites),
            ("gatling", "railgun", "rocket"),
        )
        self.assertEqual(len({site.position for site in layout.weapon_sites}), 3)

    def test_wall_ring_has_one_entrance_and_no_weapon_overlap(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, "station", level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertIsNotNone(layout.entrance)
        self.assertNotIn(layout.entrance, layout.wall_sites)
        self.assertTrue(
            set(layout.wall_sites).isdisjoint(
                site.position for site in layout.weapon_sites
            )
        )

    def test_layout_is_clipped_at_map_edge(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                width=8,
                height=8,
                our_units=(unit(10013, 0, 0, "station", level=1),),
            )
        )

        layout = build_defensive_layout(world)

        positions = tuple(site.position for site in layout.weapon_sites) + layout.wall_sites
        self.assertTrue(all(world.in_bounds(position) for position in positions))

    def test_missing_station_returns_empty_layout(self) -> None:
        layout = build_defensive_layout(
            WorldGrid.from_observation(observation())
        )

        self.assertEqual(layout.weapon_sites, ())
        self.assertEqual(layout.wall_sites, ())
        self.assertIsNone(layout.entrance)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_layout -v
```

- [ ] **Step 3: Implement the defensive layout**

Create these frozen types in `strategy/layout.py`:

```python
from dataclasses import dataclass

from future_war_agent.protocol.models import Position

from .rules import station_footprint
from .world import WorldGrid


@dataclass(frozen=True, slots=True)
class WeaponSite:
    position: Position
    weapon_type: str


@dataclass(frozen=True, slots=True)
class DefensiveLayout:
    weapon_sites: tuple[WeaponSite, ...] = ()
    wall_sites: tuple[Position, ...] = ()
    entrance: Position | None = None
```

Implement `build_defensive_layout` exactly as follows:

1. Return an empty layout when `world.our_station()` is `None`.
2. Compute the configured station footprint.
3. Generate ring-one and ring-two positions whose minimum Chebyshev distance to
   the footprint is one and two respectively.
4. Keep only in-bounds geographic land: no neutral zone and no station-footprint
   cell. Existing own weapons and walls do not change the geometric site set.
5. Order ring one by `(distance_to_map_center, x, y)`. Select the first weapon
   site, then repeatedly select the candidate maximizing minimum distance from
   selected sites, breaking ties by map-center distance, `x`, and `y`.
6. Zip at most three sites with `world.rules.weapon_loadout`.
7. Select the ring-two position closest to map center as the entrance.
8. Sort remaining ring-two positions clockwise using `atan2` around the station
   footprint center, then `(x, y)`, and exclude weapon positions.

Add `WorldGrid.is_geographic_land(position)` in `world.py`; it checks bounds,
neutral zones, and station footprints but deliberately ignores current weapons
and walls so layouts remain stable after construction.

- [ ] **Step 4: Run focused and full tests**

```powershell
python -m unittest tests.test_layout -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 43 accumulated tests pass.

- [ ] **Step 5: Commit layout generation**

```powershell
git add future_war_agent/strategy/layout.py future_war_agent/strategy/world.py tests/test_layout.py
git commit -m "feat: generate base defensive layout"
```

---

### Task 4: Prioritized Day Jobs

**Files:**
- Create: `future_war_agent/strategy/jobs.py`
- Create: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `Observation`, `WorldGrid`, `DefensiveLayout`, and path distances.
- Produces: `JobKind`, immutable `Job`, and `generate_day_jobs(observation, world, layout)` returning a read-only role-to-jobs mapping.

- [ ] **Step 1: Write failing day-job tests**

Create `tests/test_jobs.py` with seven focused cases:

```python
import unittest

from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.jobs import JobKind, generate_day_jobs
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class DayJobTests(unittest.TestCase):
    def jobs(self, observed):
        world = WorldGrid.from_observation(observed)
        return generate_day_jobs(observed, world, build_defensive_layout(world))

    def test_twilight_recall_outranks_ordinary_jobs(self) -> None:
        observed = observation(
            round_no=70,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
            ),
            zones=(Zone(Position(2, 2), "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.RECALL)

    def test_affordable_missing_weapon_is_a_build_job(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, "worker"),
                unit(10013, 5, 5, "station", level=1),
            ),
            gold=25,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.BUILD_WEAPON, {job.kind for job in jobs[10010]})

    def test_full_backpack_generates_unload_job(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    "worker",
                    backpack_capacity=2,
                    backpack=("iron", "iron"),
                ),
            ),
            zones=(Zone(Position(5, 5), "vendor"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.SELL)
        self.assertEqual(jobs[10010][0].quantity, 2)

    def test_low_gold_with_minerals_generates_proactive_sale(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker", backpack=("copper",)),
                unit(10013, 5, 5, "station", level=1),
            ),
            zones=(Zone(Position(5, 2), "vendor"),),
            vendor_shop=(ShopItem("copper", 8),),
            gold=10,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.SELL, {job.kind for job in jobs[10010]})

    def test_stone_is_preferred_while_wall_is_missing(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(
                Zone(Position(3, 2), "stone"),
                Zone(Position(2, 3), "copper"),
            ),
            vendor_shop=(ShopItem("stone", 1), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        mining = next(job for job in jobs[10010] if job.kind is JobKind.COLLECT)
        self.assertEqual(mining.name, "stone")

    def test_price_per_distance_selects_mineral_after_defense(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, "worker"),),
            zones=(
                Zone(Position(3, 1), "iron"),
                Zone(Position(6, 1), "copper"),
            ),
            vendor_shop=(ShopItem("iron", 2), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].name, "copper")

    def test_two_workers_may_receive_same_adjacent_mine(self) -> None:
        mine = Position(4, 4)
        observed = observation(
            our_units=(
                unit(10010, 3, 4, "worker"),
                unit(10012, 4, 3, "worker"),
            ),
            zones=(Zone(mine, "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].target, mine)
        self.assertEqual(jobs[10012][0].target, mine)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_jobs -v
```

- [ ] **Step 3: Implement job types and priority constants**

Create:

```python
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from future_war_agent.protocol.models import Position


class JobKind(StrEnum):
    RECALL = "recall"
    BUILD_WEAPON = "build_weapon"
    BUILD_WALL = "build_wall"
    SELL = "sell"
    COLLECT = "collect"
    PREPOSITION = "preposition"
    ATTACK = "attack"


@dataclass(frozen=True, slots=True)
class Job:
    role_id: int
    kind: JobKind
    target: Position
    priority: int
    value: float
    name: str | None = None
    quantity: int | None = None
    weapon_id: int | None = None

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            -self.priority,
            -self.value,
            self.kind.value,
            self.target.x,
            self.target.y,
            self.name or "",
        )
```

Use priorities `RECALL=500`, `BUILD_WEAPON=400`, `BUILD_WALL=350`,
`SELL=300`, `COLLECT=200`, and `PREPOSITION=100`.

- [ ] **Step 4: Implement deterministic day-job generation**

Implement `generate_day_jobs` with these exact rules:

1. Index living workers and pioneer by ID.
2. Match existing weapon types to their configured sites by position. Generate a
   build job for each missing site when current gold is at least one weapon cost.
3. Generate wall jobs only after all weapon sites are occupied and only for a
   worker whose own backpack contains `wall_material`.
4. Determine a worker is full with
   `backpack_capacity > 0 and len(backpack) >= backpack_capacity`.
5. Generate sell jobs when full, or when a weapon is missing, gold is below 25,
   and the backpack contains an item listed by the vendor. Select the mineral
   maximizing `price * count`, then name.
6. Generate mining jobs from observed `stone`, `iron`, and `copper` zones when
   the worker is not full. If walls are missing, rank stone ahead of all other
   minerals. Otherwise use `price / (interaction_distance + 1)`.
7. For each role and existing weapon, compute interaction distance. When
   `71 - round_in_phase <= distance + twilight_safety_margin`, insert a recall
   job with priority 500.
8. Give the pioneer preposition jobs to existing weapons, or to the layout
   entrance when no weapon exists.
9. Sort every role's jobs by `Job.sort_key`, copy sequences to tuples, and return
   `MappingProxyType(result)`.

Use `path_to_interaction` for every travel distance. Do not create a job whose
target is unreachable. Do not reserve build or mine targets here; Task 5 handles
the joint choice, including two-worker collection.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_jobs -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 50 accumulated tests pass.

- [ ] **Step 6: Commit day jobs**

```powershell
git add future_war_agent/strategy/jobs.py tests/test_jobs.py
git commit -m "feat: generate deterministic day jobs"
```

---

### Task 5: Tactical Candidates and Collision-Aware Joint Solver

**Files:**
- Create: `future_war_agent/strategy/joint.py`
- Create: `tests/test_joint.py`

**Interfaces:**
- Consumes: role jobs, `WorldGrid`, A* first steps, current resources, and typed Phase 1 actions.
- Produces: immutable `TacticalCandidate`, `candidates_for_jobs`, `is_valid_joint`, and `solve_joint`.

- [ ] **Step 1: Write failing joint-solver tests**

Create `tests/test_joint.py` using a small helper that constructs move and wait
candidates. Cover all seven required constraints:

```python
import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.joint import (
    TacticalCandidate,
    is_valid_joint,
    solve_joint,
)
from future_war_agent.strategy.jobs import JobKind
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


def candidate(
    role_id: int,
    start: Position,
    target: Position | None,
    *,
    priority: int = 100,
) -> TacticalCandidate:
    action = None if target is None else Action.move(target)
    return TacticalCandidate(
        role_id=role_id,
        command_actor_id=role_id,
        action=action,
        job_kind=JobKind.PREPOSITION,
        start=start,
        move_target=target,
        priority=priority,
        completes_job=False,
        progress=1 if target is not None else 0,
    )


class JointSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = observation(
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 2, 1, "pioneer"),
                unit(10012, 3, 1, "worker", backpack=("stone",)),
            )
        )
        self.world = WorldGrid.from_observation(self.observed)

    def test_rejects_duplicate_move_destinations(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 2)),
            candidate(10011, Position(2, 1), Position(2, 2)),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_position_swap(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), Position(1, 1)),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_move_into_stationary_role(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), None),
        )
        self.assertFalse(is_valid_joint(self.observed, self.world, joint))

    def test_accepts_move_into_vacated_role_cell(self) -> None:
        joint = (
            candidate(10010, Position(1, 1), Position(2, 1)),
            candidate(10011, Position(2, 1), Position(2, 2)),
        )
        self.assertTrue(is_valid_joint(self.observed, self.world, joint))

    def test_rejects_duplicate_build_and_gold_overspend(self) -> None:
        build_one = TacticalCandidate.build(
            10010,
            Position(1, 1),
            Position(2, 2),
            "gatling",
            priority=400,
            gold_cost=25,
        )
        build_two = TacticalCandidate.build(
            10012,
            Position(3, 1),
            Position(2, 2),
            "railgun",
            priority=400,
            gold_cost=25,
        )
        self.assertFalse(
            is_valid_joint(self.observed, self.world, (build_one, build_two))
        )
        poor = observation(our_units=self.observed.our.units, gold=25)
        world = WorldGrid.from_observation(poor)
        other = TacticalCandidate.build(
            10012,
            Position(3, 1),
            Position(3, 2),
            "railgun",
            priority=400,
            gold_cost=25,
        )
        self.assertFalse(is_valid_joint(poor, world, (build_one, other)))

    def test_controller_cannot_also_take_personal_action(self) -> None:
        attack = TacticalCandidate.attack(
            role_id=10011,
            start=Position(2, 1),
            weapon_id=10020,
            target=Position(6, 6),
            priority=500,
        )
        move = candidate(10011, Position(2, 1), Position(2, 2))
        self.assertFalse(is_valid_joint(self.observed, self.world, (attack, move)))

    def test_solver_is_deterministic(self) -> None:
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
        self.assertEqual(
            solve_joint(self.observed, self.world, choices),
            solve_joint(self.observed, self.world, choices),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_joint -v
```

- [ ] **Step 3: Implement tactical candidates**

Create a frozen `TacticalCandidate` with fields:

```python
@dataclass(frozen=True, slots=True)
class TacticalCandidate:
    role_id: int
    command_actor_id: int
    action: Action | None
    job_kind: JobKind
    start: Position
    move_target: Position | None = None
    build_target: Position | None = None
    priority: int = 0
    completes_job: bool = False
    progress: int = 0
    gold_cost: int = 0
    stone_cost: int = 0
```

Provide exact class methods:

- `wait(role_id, start)` produces no action.
- `build(role_id, start, target, name, priority, gold_cost=0, stone_cost=0)` emits
  `Action.build` and sets `build_target`.
- `attack(role_id, start, weapon_id, target, priority)` emits `Action.attack`
  under the weapon ID, uses `JobKind.ATTACK`, and records the controller role ID.

Implement `candidates_for_jobs(observation, world, role, jobs, limit=4)`:

- `COLLECT`: emit `Action.collect` when adjacent; otherwise emit up to two A*
  move steps toward interaction cells.
- `SELL`: emit `Action.sell(name, quantity)` when adjacent to the vendor;
  otherwise emit move steps.
- `BUILD_WEAPON` and `BUILD_WALL`: emit `Action.build` when adjacent; otherwise
  emit move steps. Record 25 gold for weapons and one stone for walls.
- `RECALL` and `PREPOSITION`: emit move steps directly toward the job target.
- De-duplicate candidates by `(command_actor_id, action)` and sort by
  `(-priority, -completes_job, -progress, command_actor_id, repr(action))`.
- Append exactly one wait candidate before applying the limit, ensuring wait is
  retained by replacing the last candidate when necessary.

- [ ] **Step 4: Implement exact joint validation and solving**

`is_valid_joint` must enforce:

```python
move_by_role = {
    item.role_id: item.move_target
    for item in joint
    if item.move_target is not None
}
```

Then reject duplicate roles, duplicate command actor IDs, duplicate move targets,
swaps, hard-blocked move cells, and moves into a friendly role cell unless that
occupant has a different move target. Reject duplicate non-`None` build targets,
sum `gold_cost` against `observation.our.gold`, and sum stone cost per role
against that role's backpack count.

`solve_joint` sorts role IDs, enumerates `itertools.product` over their candidate
tuples, keeps valid combinations, and maximizes:

```python
(
    sum(item.priority for item in joint),
    sum(item.completes_job for item in joint),
    sum(item.progress for item in joint),
    sum(item.action is not None for item in joint),
    tuple(
        (item.command_actor_id, repr(item.action))
        for item in sorted(joint, key=lambda value: value.command_actor_id)
    ),
)
```

Convert the winning non-wait candidates to a `Decision`. If there is no valid
combination, return `Decision()`.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_joint -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 57 accumulated tests pass.

- [ ] **Step 6: Commit the joint solver**

```powershell
git add future_war_agent/strategy/joint.py tests/test_joint.py
git commit -m "feat: solve collision-free joint actions"
```

---

### Task 6: Night Assignment, Safe Positioning, and Basic Attacks

**Files:**
- Create: `future_war_agent/strategy/night.py`
- Create: `tests/test_night.py`

**Interfaces:**
- Consumes: `Observation`, `WorldGrid`, A* distances, living roles, weapons, robots, and station footprint.
- Produces: `ControllerAssignment`, `assign_controllers`, and `generate_night_candidates` compatible with Task 5's joint solver.

- [ ] **Step 1: Write failing night tests**

Create six tests in `tests/test_night.py`:

```python
import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.protocol.models import Position
from future_war_agent.strategy.night import (
    assign_controllers,
    generate_night_candidates,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class NightPolicyTests(unittest.TestCase):
    def test_assignment_minimizes_total_distance(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 10, 10, "pioneer"),
                unit(10020, 2, 2, "gatling", level=1, attack_range=4),
                unit(10030, 9, 9, "railgun", level=1, attack_range=6),
            ),
        )
        world = WorldGrid.from_observation(observed)

        assignments = assign_controllers(observed, world)

        paired = {(item.role_id, item.weapon_id) for item in assignments}
        self.assertEqual(paired, {(10010, 10020), (10011, 10030)})

    def test_assignment_stand_cells_are_unique(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10011, 1, 3, "pioneer"),
                unit(10020, 4, 2, "gatling", level=1, attack_range=4),
                unit(10030, 4, 3, "railgun", level=1, attack_range=6),
            ),
        )
        assignments = assign_controllers(observed, WorldGrid.from_observation(observed))
        self.assertEqual(
            len({item.stand for item in assignments}),
            len(assignments),
        )

    def test_role_moves_toward_assigned_weapon(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10020, 5, 5, "gatling", level=1, attack_range=4),
            ),
        )
        world = WorldGrid.from_observation(observed)

        choices = generate_night_candidates(observed, world)

        self.assertTrue(
            any(item.action and item.action.kind is ActionKind.MOVE for item in choices[10010])
        )

    def test_ready_level_one_weapon_attacks_nearest_station_threat(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10013, 1, 1, "station", level=1),
                unit(10020, 5, 5, "gatling", level=1, attack_range=5),
            ),
            robots=(robot(30001, 3, 3), robot(30002, 7, 5)),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        attack = next(
            item for item in choices[10010]
            if item.action and item.action.kind is ActionKind.ATTACK
        )
        self.assertEqual(attack.action.target_positions, (Position(3, 3),))

    def test_cooldown_and_high_level_weapons_do_not_attack(self) -> None:
        for level, cooldown in ((1, 2), (2, 0)):
            with self.subTest(level=level, cooldown=cooldown):
                observed = observation(
                    round_no=71,
                    our_units=(
                        unit(10010, 4, 5, "worker"),
                        unit(
                            10020,
                            5,
                            5,
                            "gatling",
                            level=level,
                            cooldown=cooldown,
                            attack_range=5,
                        ),
                    ),
                    robots=(robot(30001, 3, 3),),
                )
                choices = generate_night_candidates(
                    observed,
                    WorldGrid.from_observation(observed),
                )
                self.assertFalse(
                    any(
                        item.action and item.action.kind is ActionKind.ATTACK
                        for item in choices[10010]
                    )
                )

    def test_unassigned_role_gets_safe_defensive_candidate(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10011, 2, 2, "pioneer"),
                unit(10013, 5, 5, "station", level=1),
                unit(10020, 5, 4, "gatling", level=1, attack_range=4),
            ),
            robots=(robot(30001, 1, 1),),
        )
        choices = generate_night_candidates(
            observed,
            WorldGrid.from_observation(observed),
        )
        self.assertIn(10011, choices)
        self.assertGreaterEqual(len(choices[10011]), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_night -v
```

- [ ] **Step 3: Implement controller assignment**

Create:

```python
@dataclass(frozen=True, slots=True)
class ControllerAssignment:
    role_id: int
    weapon_id: int
    stand: Position
    distance: int
```

`assign_controllers` must:

1. Sort living roles and weapons by ID.
2. For assignment sizes from `min(len(roles), len(weapons))` down to one,
   enumerate every weapon combination of that size and every role permutation
   of that size.
3. For each role/weapon pair, enumerate all reachable weapon interaction cells
   and their path costs.
4. Use `itertools.product` to enumerate stand-cell combinations and reject
   duplicate stands.
5. Minimize `(total_distance, tuple(role_id, weapon_id, stand.x, stand.y))`.
6. Return the best tuple of assignments, or an empty tuple when no pair is
   reachable.

- [ ] **Step 4: Implement night candidates and basic targeting**

For an assigned role not at its stand, use Task 5's movement candidate helper.
For an adjacent role, generate an attack only when the weapon level is one,
cooldown is zero, attack range is positive, and an in-range living robot exists.

Sort target robots by:

```python
(
    robot.target_team != observation.our.team_type,
    min(
        robot.position.chebyshev_distance(cell)
        for cell in station_cells
    ),
    robot.health,
    robot.robot_id,
)
```

The chosen attack candidate uses priority 500. Append a wait candidate for an
adjacent controller. When no own station exists, rank threats by distance to the
weapon instead of taking a minimum over an empty station footprint.

For unassigned roles, enumerate traversable cells inside the distance-two
station ring. Rank them by negative minimum robot distance, station distance,
`x`, and `y`; generate movement toward the safest reachable cell plus wait.

Return a read-only mapping of role IDs to candidate tuples. When there is no
station, use each unassigned role's current position as its safe target.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m unittest tests.test_night -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 63 accumulated tests pass.

- [ ] **Step 6: Commit the night policy**

```powershell
git add future_war_agent/strategy/night.py tests/test_night.py
git commit -m "feat: add safe night positioning"
```

---

### Task 7: Strategy Planner and Production Controller Integration

**Files:**
- Create: `future_war_agent/strategy/planner.py`
- Create: `tests/fixtures/strategy_request.json`
- Create: `tests/test_strategy_planner.py`
- Modify: `future_war_agent/controller.py`
- Modify: `tests/test_controller.py`

**Interfaces:**
- Consumes: every Phase 2 strategy component and the existing controller planner protocol.
- Produces: `plan_turn(observation, rules=DEFAULT_RULES) -> Decision` and a production `default_planner` that delegates to it.

- [ ] **Step 1: Write failing planner integration tests**

Create `tests/test_strategy_planner.py`:

```python
import unittest

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.serializer import decision_to_payload
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.planner import plan_turn
from tests.strategy_helpers import observation, robot, unit


class StrategyPlannerTests(unittest.TestCase):
    def test_day_worker_produces_a_legal_action(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
            ),
            zones=(Zone(Position(3, 2), "stone"),),
        )

        decision = plan_turn(observed)
        validated = validate_decision(observed, decision)

        self.assertTrue(validated.commands)

    def test_twilight_moves_remote_role_toward_defense(self) -> None:
        observed = observation(
            round_no=70,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10013, 9, 9, "station", level=1),
                unit(10020, 8, 8, "gatling", level=1, attack_range=4),
            ),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10010].kind, ActionKind.MOVE)

    def test_night_ready_weapon_attacks(self) -> None:
        observed = observation(
            round_no=71,
            our_units=(
                unit(10010, 4, 5, "worker"),
                unit(10013, 1, 1, "station", level=1),
                unit(10020, 5, 5, "gatling", level=1, attack_range=5),
            ),
            robots=(robot(30001, 3, 3),),
        )

        decision = plan_turn(observed)

        self.assertEqual(decision.commands[10020].kind, ActionKind.ATTACK)

    def test_identical_observations_serialize_identically(self) -> None:
        observed = observation(
            our_units=(unit(10010, 2, 2, "worker"),),
            zones=(Zone(Position(3, 2), "iron"),),
            vendor_shop=(ShopItem("iron", 4),),
        )
        self.assertEqual(
            decision_to_payload(plan_turn(observed)),
            decision_to_payload(plan_turn(observed)),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_strategy_planner -v
```

- [ ] **Step 3: Implement planner composition**

Create `strategy/planner.py`:

```python
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase

from .jobs import generate_day_jobs
from .joint import candidates_for_jobs, solve_joint
from .layout import build_defensive_layout
from .night import generate_night_candidates
from .rules import DEFAULT_RULES, RulesConfig
from .world import WorldGrid


def plan_turn(
    observation: Observation,
    rules: RulesConfig = DEFAULT_RULES,
) -> Decision:
    world = WorldGrid.from_observation(observation, rules)
    if not world.friendly_roles:
        return Decision()
    if observation.time.phase is Phase.NIGHT:
        candidates = generate_night_candidates(observation, world)
        return solve_joint(observation, world, candidates)

    layout = build_defensive_layout(world)
    jobs = generate_day_jobs(observation, world, layout)
    candidates = {
        role.unit_id: candidates_for_jobs(
            observation,
            world,
            role,
            jobs.get(role.unit_id, ()),
        )
        for role in world.friendly_roles
    }
    return solve_joint(observation, world, candidates)
```

- [ ] **Step 4: Connect the production default planner**

Modify `controller.py`:

```python
from future_war_agent.strategy.planner import plan_turn


def default_planner(observation: Observation) -> Decision:
    return plan_turn(observation)
```

Remove the now-unused `safe_decision` import but retain `safe_payload`.

- [ ] **Step 5: Add a representative raw strategy request**

Create `tests/fixtures/strategy_request.json` with:

- round 1;
- a 15x15 map;
- station at `(7, 7)`;
- workers at `(2, 2)` and `(2, 3)`;
- pioneer at `(6, 7)`;
- stone at `(3, 2)`, iron at `(4, 4)`, vendor at `(1, 1)`;
- 75 gold and vendor prices for stone and iron;
- empty enemy roles, robots, tasks, news, and errors.

Every object uses the exact parser field names already exercised by
`tests/fixtures/request.json`.

- [ ] **Step 6: Update controller tests for the production strategy**

Keep the existing fixture for injected-planner and failure tests. Add a second
payload loaded from `strategy_request.json`, replace the old empty-default test,
and assert:

```python
def test_default_controller_runs_phase_2_strategy(self) -> None:
    strategy_payload = json.loads(STRATEGY_FIXTURE.read_text(encoding="utf-8"))

    response = handle_payload(strategy_payload)

    self.assertTrue(response["roleCommandMap"])
    self.assertEqual(response["prompt"], "")
    self.assertEqual(response["executeCmd"], "")
```

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m unittest tests.test_strategy_planner -v
python -m unittest tests.test_controller -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all 67 accumulated tests pass.

- [ ] **Step 8: Commit production integration**

```powershell
git add future_war_agent/controller.py future_war_agent/strategy/planner.py tests/fixtures/strategy_request.json tests/test_controller.py tests/test_strategy_planner.py
git commit -m "feat: enable deterministic phase 2 planner"
```

---

### Task 8: Phase 2 Acceptance Verification

**Files:**
- Verify all Phase 1 and Phase 2 files; modify only after a focused failing test demonstrates an acceptance defect.

**Interfaces:**
- Consumes: the complete runnable service.
- Produces: fresh evidence that the Phase 2 spec and all Phase 1 guarantees hold on the final branch.

- [ ] **Step 1: Run the complete standard-library test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: 67 tests pass with zero failures and zero errors. Logged exceptions in
the two intentional server/controller failure tests are expected; the unittest
summary must still be `OK`.

- [ ] **Step 2: Compile every Python file**

```powershell
python -m compileall -q main.py future_war_agent tests
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Run a real HTTP strategy probe**

Start:

```powershell
python main.py 18080
```

In a second terminal, POST `tests/fixtures/strategy_request.json` to
`http://127.0.0.1:18080/`. Verify status 200, non-empty `roleCommandMap`, empty
`prompt`, and empty `executeCmd`. Stop the server with Ctrl+C.

- [ ] **Step 4: Verify deterministic replay**

POST the same fixture twice to a fresh server and compare parsed JSON objects.
They must be equal. This verifies that no cross-request mutation entered the
planner.

- [ ] **Step 5: Check repository integrity**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -10
```

Expected: no whitespace errors, no uncommitted files, and HEAD on
`codex/phase2-deterministic-policy`.

- [ ] **Step 6: Compare implementation to every acceptance criterion**

Confirm with a named test or probe that:

- the default response can contain a strategy command;
- target-cell conflicts and swaps are rejected;
- workers approach, collect, unload, and sell;
- two workers may collect one mine;
- construction prevents duplicate targets and resource overspending;
- twilight recall uses actual A* distance plus a two-round margin;
- night controllers approach unique weapon stands;
- ready level-one weapons make a conservative in-range attack;
- identical observations produce identical decisions;
- Phase 1 validation and fallback remain active.

If one item lacks evidence, add one focused failing test, witness the failure,
make the minimum correction, rerun all verification, and commit the correction
with `fix: satisfy phase 2 acceptance`.
