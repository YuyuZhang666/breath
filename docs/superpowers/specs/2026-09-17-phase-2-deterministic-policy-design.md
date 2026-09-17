# Phase 2 Deterministic Playable Policy Design

## Purpose

Phase 2 replaces the deliberately empty Phase 1 planner with the first
deterministic playable policy. It adds a reliable spatial model, eight-direction
pathfinding, collision-aware joint movement, a basic mining and selling loop,
base-relative construction, twilight recall, safe night positioning, and a
conservative level-one weapon attack.

This phase is the tactical foundation for later search. It does not add robust
beam search, a simulator, mine depletion beliefs, task solving, upgrades,
opponent modeling, exact weapon optimization, or cross-half learning.

## Constraints

- Runtime and tests use only the Python standard library.
- The supported runtime remains Python 3.11 or newer.
- Strategy consumes the immutable Phase 1 `Observation` and produces a Phase 1
  `Decision`; it never handles raw request dictionaries or serialized payloads.
- Phase 1 validation, serialization, HTTP handling, and safe fallback remain the
  final authority and failure boundary.
- Planning is request-local and contains no mutable cross-request state.
- The same observation always produces the same decision.
- Formal interface documentation takes precedence over constants in the demo.
- Unknown or insufficiently specified mechanics produce no command rather than
  a speculative command.

## Scope

Phase 2 includes:

- a typed world grid and occupancy model;
- deterministic A* pathfinding to cells and interaction ranges;
- base-relative level-one weapon and wall layouts;
- mining, vendor unloading, construction, twilight recall, and preposition jobs;
- at most four tactical candidates per living controllable role;
- exact enumeration of the resulting three-role joint action space;
- collision prevention for destination conflicts, swaps, and stationary roles;
- controller-to-weapon assignment and safe night positions;
- conservative level-one attacks against the nearest station threat;
- focused unit, integration, regression, compilation, and HTTP verification.

Phase 2 explicitly excludes:

- macro beam search and counterfactual rollouts;
- persistent mine remaining estimates and enemy hidden-state beliefs;
- tasks, LLM prompts, sandbox commands, treasure, and world-news reasoning;
- building upgrades, repair items, bombs, stun items, and summon orders;
- optimized gatling sectors, railgun rays, rocket splash, overkill control, and
  attacks by weapons above level one;
- opponent modeling, match memory, self-play, and PSRO.

## Architecture

```text
Observation
    |
    v
WorldGrid                 immutable spatial indexes and occupancy layers
    |
    +--> DefensiveLayout  base-relative weapon sites, wall ring, entrance
    |
    v
JobGenerator              mining, selling, building, recall, preposition
    |
    v
Pathfinder                deterministic eight-direction A*
    |
    v
CandidateGenerator        no more than four choices per controllable role
    |
    v
JointActionSolver         exact Cartesian-product enumeration and scoring
    |
    v
Decision
    |
    v
Phase 1 Validator -> Serializer -> HTTP response
```

The strategy package is organized by responsibility:

```text
future_war_agent/strategy/
|-- __init__.py
|-- rules.py          Central Phase 2 constants and geometry convention
|-- world.py          Occupancy layers, footprints, and interaction cells
|-- pathfinding.py    Deterministic A* paths and distances
|-- layout.py         Weapon sites, wall ring, and entrance selection
|-- jobs.py           Typed jobs, job generation, and priority ranking
|-- joint.py          Role candidates, collision checks, and exact solver
|-- night.py          Controller assignment, safe positions, basic attacks
`-- planner.py        Public `plan_turn(observation) -> Decision` composition
```

`future_war_agent.controller.default_planner` delegates to `plan_turn`. No
transport, parser, serializer, or server code understands strategy rules.

## Central Rules and Geometry

All Phase 2 constants live in an immutable `RulesConfig`. The initial values
that are required by the policy are:

- weapon build cost: 25 gold, sourced from the supplied official demo;
- wall material name: `stone`;
- wall material cost: one stone per wall;
- weapon loadout order: `gatling`, `railgun`, `rocket`;
- twilight safety margin: two rounds;
- eight legal movement offsets, excluding `(0, 0)`;
- build and collect interaction distance: Chebyshev distance one.

The formal interface says that the station coordinate is its top-left corner,
so the default 2x2 footprint is:

```text
(x, y)       (x + 1, y)
(x, y + 1)   (x + 1, y + 1)
```

The supplied demo instead subtracts one from `y`. The convention is isolated in
`RulesConfig.station_y_direction`, defaulting to `+1` to follow the formal
interface. A judge integration finding can change this one value without
rewriting occupancy, layout, or pathfinding.

## World Grid

`WorldGrid.from_observation(observation)` builds immutable spatial indexes. It
keeps occupancy categories separate because pathfinding and joint movement need
different interpretations:

- `neutral_cells`: every mine, vendor, weapon shop, and task point;
- `structure_cells`: all visible stations, weapons, and walls from both teams;
- `robot_cells`: every robot position;
- `friendly_roles`: living worker and pioneer positions keyed by unit ID;
- `visible_enemy_roles`: visible living enemy worker and pioneer positions;
- `hard_blocked`: neutral cells, structure cells, robots, and enemy roles;
- `soft_friendly`: current friendly-role cells, resolved by the joint solver.

Stations occupy their configured 2x2 footprint. Every other unit and zone
occupies one cell. Coordinates outside `0 <= x < width` and
`0 <= y < height` are never traversable.

Diagonal movement is legal even when the two orthogonal neighbor cells are
blocked, matching the official rule that adjacent obstacles do not prevent a
diagonal step.

Interaction targets such as mines, vendors, weapons, and build sites are not
path goals themselves. The grid returns the in-bounds, unblocked cells at
Chebyshev distance one from the target. A role already in one of these cells has
zero travel distance.

## Deterministic A* Pathfinding

A* uses unit movement cost for all eight directions and Chebyshev distance as
its admissible heuristic. Neighbor order is constant, and heap ties are broken
by coordinate and insertion order, so repeated calls return the same path.

The pathfinder exposes:

- a complete path from one traversable cell to another;
- distance to a cell;
- the best path to any valid interaction cell around a target;
- up to two distinct first steps that preserve the best or next-best route.

For candidate generation, other friendly roles are soft obstacles. A first step
into a friendly role's current cell may be proposed, but the joint solver keeps
it only when that role moves away in the same turn and the movement is not a
swap. Hard obstacles are never traversed. An unreachable goal returns no path
instead of raising an exception.

## Defensive Layout

The request does not expose the blue weapon-build and yellow wall-build regions.
Phase 2 therefore follows the supplied demo's base-relative construction model
while keeping it isolated behind `DefensiveLayout`.

Weapon candidates are in-bounds land cells at Chebyshev distance one from the
station footprint. The first site is closest to map center; later sites maximize
their minimum distance from already selected sites, with map-center distance,
`x`, and `y` as deterministic tie breakers. At most three sites are selected and
assigned the loadout `gatling`, `railgun`, `rocket`.

Wall candidates form the ring at Chebyshev distance two from the station
footprint. The in-bounds ring cell closest to map center is the entrance and is
never built. Other ring cells are ordered deterministically from the entrance
around the perimeter. A candidate is usable only when it is land, currently
unoccupied, and distinct from a weapon site.

This module is the only place that assumes a base-relative build region. If
judge integration exposes a different legal region, the layout can be replaced
without changing jobs, pathfinding, or joint solving.

## Jobs and Day Policy

Jobs are immutable values with a kind, target, allowed roles, priority, and
deterministic tie-break key. Day jobs are ranked in the following order.

### 1. Twilight Recall

For each living role, the planner computes the shortest distance to an
uncontested cell adjacent to its assigned weapon. The number of actions left
before night, including the current day action, is:

```text
71 - round_in_phase
```

When this is no greater than `travel_distance + 2`, an emergency recall job
replaces that role's ordinary job. If fewer weapons exist than roles, unmatched
roles recall to safe cells inside the defensive ring.

### 2. Basic Defense Construction

Missing weapon sites are filled in loadout order when combined weapon builds in
the selected joint action do not exceed current gold. A worker adjacent to the
site builds immediately; otherwise it receives a travel job.

After all three weapon sites are filled, workers carrying stone fill distinct
wall-ring cells. A worker must own the required stone. The joint solver rejects
duplicate build targets and any combination that overspends shared gold.

### 3. Vendor Unloading

A worker unloads when its backpack is full. It may also unload before a required
weapon build when current gold is below the configured weapon cost and its pack
contains saleable minerals.

At a vendor-adjacent cell, it sells all copies of one mineral in one command.
The chosen mineral maximizes `current vendor price * quantity`, then uses the
name as a stable tie breaker. Otherwise the worker travels to the nearest
reachable vendor.

### 4. Mining

Workers with free capacity mine observed `stone`, `iron`, or `copper` zones.
While walls remain missing, stone receives a construction bonus. Otherwise the
score is based on current vendor price divided by `travel_distance + 1`.
Distance, mineral name, coordinate, and worker ID provide stable tie breaks.

If already adjacent, the worker issues `collect`; otherwise it moves toward an
interaction cell. Both workers may collect the same mine in the same turn. This
preserves the official tail-depletion rule. They may not move to the same cell.

Phase 2 does not estimate hidden remaining mineral quantities. A mine that is
present in the current observation is eligible; a missing mine is not.

### 5. Pioneer Preposition

Phase 2 does not accept tasks. The pioneer stays inside the defensive area or
moves toward its assigned weapon when that movement does not conflict with a
higher-priority worker action.

## Night Policy

At night, the planner enumerates all assignments between living controllable
roles and living weapons, at most `3! = 6` full assignments. Assignment cost is
the sum of A* distances to valid adjacent stand cells. Conflicting stand cells
make an assignment invalid. The minimum cost assignment wins, with role and
weapon IDs as tie breakers.

For each assigned role:

- if it is not adjacent to its weapon, generate movement candidates toward the
  assigned stand cell;
- if it is adjacent and the weapon can safely make a Phase 2 attack, generate a
  controller action under the weapon ID and no personal action under the role;
- if it is adjacent but cannot attack, wait in place.

A Phase 2 attack is generated only when:

- the weapon is `gatling`, `railgun`, or `rocket`;
- its level is exactly one;
- its cooldown is zero;
- a living robot is within the observation's weapon attack range.

Robots targeting our team are preferred. The target minimizing distance to any
station footprint cell is selected, followed by health and robot ID as stable
tie breakers. The attack contains one target, which is valid for a level-one
weapon. Weapons above level one stay positioned but do not attack; their
multi-target geometry belongs to the exact-combat phase.

Unassigned roles may occupy an empty cell inside the defensive ring. Candidate
cells maximize minimum distance from living robots, then minimize distance to
the station, with coordinates as tie breakers. The policy does not predict
future robot movement in Phase 2.

## Candidate Generation and Joint Solving

Each living worker or pioneer receives at most four candidates:

1. perform the job immediately when already in interaction range;
2. take the deterministic best A* first step;
3. take one distinct next-best first step;
4. wait, represented by the absence of a command.

Weapon attacks appear as the assigned controller's tactical candidate but are
emitted under the weapon ID. Selecting one consumes the controller, so that
role cannot also move, collect, sell, build, or perform another action.

With three roles the solver evaluates at most `4^3 = 64` combinations. A
combination is rejected when:

- two moving roles choose the same target cell;
- two roles exchange their current positions;
- a role moves into a friendly role's cell and that role does not move away;
- a move enters a hard-blocked or out-of-bounds cell;
- two builds target the same cell;
- one controller is assigned to multiple weapons;
- selected builds exceed shared gold or an actor's inventory;
- an immediate action is not legal for the actor, phase, distance, or target.

Two workers collecting the same mine is explicitly legal. Moving toward the
same mine remains subject to movement collision rules.

Valid combinations are compared lexicographically by:

1. total job priority satisfied;
2. number of immediate job completions;
3. total reduction in remaining path distance;
4. number of non-wait actions;
5. serialized actor IDs and action fields.

The final key makes selection deterministic even when utility is otherwise
equal. The solver returns a Phase 1 `Decision` containing only the chosen
commands.

## Integration and Failure Handling

`default_planner(observation)` becomes the production Phase 2 entry point and
delegates to the stateless strategy planner. The existing injected-planner
controller interface remains unchanged.

Failure behavior is conservative:

- a missing station disables layout, construction, recall, and night defense,
  but does not disable independently legal mining or selling;
- missing mines, vendors, weapons, or roles produce no corresponding jobs;
- an unreachable goal produces a wait candidate;
- a role without a legal candidate does not suppress other roles;
- if every joint combination is invalid, the planner returns an empty decision;
- Phase 1 validation filters any remaining invalid individual action;
- unexpected strategy exceptions are caught by the Phase 1 controller and
  become the complete safe response.

No Phase 2 error text is returned over HTTP. No raw mutable request object enters
the strategy package.

## Testing

Tests use `unittest` and production code only.

### World-grid tests

- default formal-document station footprint;
- alternate configured station direction;
- neutral, structure, robot, friendly-role, and enemy-role occupancy;
- in-bounds interaction cells;
- legal diagonal passage between two orthogonal blockers.

### Pathfinding tests

- deterministic eight-direction shortest paths;
- obstacle detours;
- unreachable goals;
- paths ending adjacent to mines, vendors, weapons, and build sites;
- best and next-best first steps.

### Layout tests

- three distinct in-bounds weapon sites;
- weapon loadout order;
- a distance-two wall ring with exactly one entrance;
- map-edge clipping and occupied-site filtering.

### Job tests

- twilight recall outranks mining and construction;
- missing weapons outrank ordinary mining when affordable;
- full backpacks produce vendor jobs;
- insufficient weapon gold permits a proactive vendor job;
- stone preference while walls are incomplete;
- vendor-price-per-distance mining after the defense is complete;
- two adjacent workers may collect the same mine.

### Joint-solver tests

- duplicate move destinations are rejected;
- swaps are rejected;
- moving into a stationary role is rejected;
- moving into a role's vacated cell is accepted;
- duplicate builds and shared-gold overspending are rejected;
- a weapon controller cannot also perform a personal action;
- repeated inputs select the same joint action.

### Night tests

- minimum-total-distance role-to-weapon assignment;
- conflicting stand cells invalidate an assignment;
- a role moves toward its assigned weapon;
- a ready level-one weapon attacks the nearest station threat;
- cooldown and higher-level weapons do not attack;
- unassigned roles choose safe defensive cells.

### Integration and regression tests

- complete day, twilight, and night observations produce valid decisions;
- the default controller uses the Phase 2 planner;
- invalid input and strategy exceptions still return the safe payload;
- repeated real HTTP requests remain independent;
- every Phase 1 test continues to pass;
- `python -m compileall -q main.py future_war_agent tests` succeeds.

## Acceptance Criteria

- The default service no longer always returns an empty command map.
- The policy never deliberately selects a same-cell move conflict or role swap.
- Workers can complete the observable mining loop: approach, collect, approach a
  vendor when full, and sell.
- Both workers may collect the same adjacent mine in one turn.
- Workers can build the three configured level-one weapon types and basic wall
  ring without duplicate targets or shared-resource overspending.
- Roles recall early enough to reach assigned weapon stand cells with a two-turn
  margin according to current A* distance.
- Night roles approach weapons without friendly collisions, and ready
  level-one weapons issue a conservative attack when a target is in range.
- Identical observations produce identical decisions.
- Every decision still passes through Phase 1 validation and safe serialization.
- The complete standard-library test suite and compilation checks pass.

## Deferred Work

The next strategy phases may add persistent action reconciliation and mine
beliefs, robust diverse beam search, full simulation and survival proof,
optimized multi-level weapon attacks, item use, tasks, news and treasure
reasoning, enemy beliefs, summons, cross-half memory, and PSRO. These systems
must reuse the Phase 2 world, pathfinding, job, and joint-action boundaries
rather than bypass them.

Phase 4 parameterizes the weapon loadout, day priorities, resource reserve, and
confirmed medicine policy through an immutable `StrategicIntent`. Calling
`plan_turn` with its default intent preserves this Phase 2 design exactly, so
the policy remains the deterministic fallback for every later phase.
