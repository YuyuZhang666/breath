# Phase 1 Reliable Foundation Design

## Purpose

Phase 1 builds a reliable but intentionally non-strategic Future War agent. It must start with the official command-line contract, parse judge observations into typed internal models, validate and serialize decisions, and return a safe response for malformed input or internal failures.

The phase is complete when later strategy modules can consume a stable `Observation` and return a `Decision` without handling raw JSON or HTTP details.

## Constraints

- Runtime and tests use only the Python standard library.
- The supported runtime is Python 3.11 or newer.
- The service starts with `python main.py <port>` and listens on `0.0.0.0:<port>`.
- Formal interface documentation and live request fields take precedence over constants in the supplied demo.
- Phase 1 implements no mining, movement, economy, combat, search, task-solving, opponent-modeling, or self-play strategy.
- Unknown request fields are ignored so compatible protocol extensions do not crash the agent.

## Architecture

```text
main.py
`-- future_war_agent
    |-- server.py           HTTP transport and exception boundary
    |-- controller.py       Per-turn control entry point
    |-- protocol/
    |   |-- models.py       Immutable request-domain models
    |   |-- parser.py       Raw JSON to Observation
    |   `-- time.py         Day, phase, and phase-round calculations
    |-- decision/
    |   |-- actions.py      Typed action model
    |   |-- decision.py     Commands, prompt, and execute command
    |   |-- validator.py    Output validation against an observation
    |   `-- serializer.py   Decision to response payload
    `-- fallback.py         Deterministic safe decision

tests/
|-- fixtures/
|   `-- request.json
|-- test_parser.py
|-- test_time.py
|-- test_actions.py
|-- test_validator.py
|-- test_serializer.py
`-- test_server.py
```

Each module has one responsibility. HTTP code does not understand game rules, protocol parsing does not make decisions, and serialization does not silently invent strategy.

## Request Model

The parsing pipeline is:

```text
raw JSON object
-> structural validation and type conversion
-> immutable Observation
```

The domain model contains:

- `Position`
- `Zone`
- `UnitState`
- `RobotState`
- `TaskPointState`
- `OurTeamState`
- `EnemyTeamState`
- `WorldNews`
- `ShopItem`
- `GameError`
- `Observation`

`roundNo`, `mapInfo`, and `teamOur` are required. Missing or invalid values in these objects make the request invalid. Enemy information, robots, news, shops, tasks, previous-action results, task text, LLM output, command output, and errors may be absent and receive empty defaults.

Fields documented by the interface but omitted by a sample, including `Robot.targetTeam` and `PlayerTask.timeoutRounds`, are optional. Unknown fields are ignored. Numeric, string, boolean, coordinate, and identifier values are converted deliberately; booleans are not accepted as integers.

Models are frozen dataclasses containing tuples and read-only mappings where appropriate. Raw dictionaries do not cross into strategy-facing code.

## Time Model

Rounds are one-based. A day contains 130 rounds:

- phase rounds 1 through 70 are day;
- phase rounds 71 through 130 are night.

The time model exposes the day number, phase, zero-based round offset within the day, and one-based round within the current phase. Boundary behavior is covered by tests for rounds 1, 70, 71, 130, and 131.

## Decision Model

A `Decision` contains:

- at most one action for each command-map key;
- an optional LLM prompt represented internally and externally as a string;
- an optional sandbox command represented internally and externally as a string.

The supported actions are `move`, `attack`, `sell`, `buy`, `build`, `remove`, `acceptTask`, `submitAnswer`, `summonTreasure`, `use`, `drop`, and `collect`.

Actions use typed fields rather than unrestricted dictionaries. Serialization omits fields that do not apply to an action and emits command-map keys as strings.

## Validation

Validation occurs before serialization and has two levels.

Structural validation checks:

- the action kind is supported;
- fields required by the action are present;
- target positions have the required basic cardinality;
- names and task answers are non-empty where required;
- quantities are positive where required;
- a controller is not assigned to more than one weapon in a turn.

Observation-aware validation checks what Phase 1 can establish without simulating hidden game rules:

- the command-map actor exists in our current units;
- target positions are within map bounds;
- `attack` is rejected during day;
- `build` is rejected during night;
- attack commands are issued under a weapon ID;
- an attack controller exists, is alive, is a worker or pioneer, and is not also given a personal action;
- a controller is within Chebyshev distance one of the weapon;
- ordinary personal actions are not issued under building IDs.

The validator returns the valid subset of a decision. One invalid action does not suppress unrelated valid actions. If validation itself fails, the controller returns the deterministic fallback.

More detailed legality, including inventory ownership, build zones, weapon target count by level, line geometry, and action distance, belongs to later rule-engine phases once authoritative values and rules are represented centrally.

## Serialization and Safe Response

Every successful response contains all three top-level fields:

```json
{
  "roleCommandMap": {},
  "prompt": "",
  "executeCmd": ""
}
```

The official response sample is treated as a catalogue of action shapes rather than valid JSON because it repeats object keys. Production serialization always emits valid JSON with at most one command per key.

## HTTP and Failure Handling

The service uses `ThreadingHTTPServer` and `BaseHTTPRequestHandler` from the standard library. Each POST request reads its declared body, decodes UTF-8 JSON, invokes the controller, and returns UTF-8 JSON with HTTP status 200.

Failure policy:

- an invalid individual action is removed while other valid actions remain;
- a malformed decision becomes the safe response;
- malformed JSON or a request missing required protocol data becomes the safe response;
- unexpected exceptions are logged server-side and become the safe response;
- response bodies never expose stack traces, local paths, or exception text;
- a failed request does not stop the HTTP server.

Phase 1 state is request-local. No mutable observation data is shared between concurrent requests.

In the complete agent, this statement applies to the Phase 1 protocol and HTTP
boundary. Later strategy phases may keep process-local state above that
boundary. Such state is owned by the locked `StrategyEngine`; Phase 1 remains
strategy-agnostic and continues to validate and serialize every final decision.

## Controller Behavior

The Phase 1 controller deliberately returns an empty `Decision`. Its responsibility is to prove the integration path:

```text
parse observation
-> obtain decision
-> validate decision
-> serialize response
```

Strategy behavior begins in Phase 2. This separation prevents untested game logic from compromising the protocol boundary.

## Testing

Tests use `unittest` and real production code. No third-party test runner or mocking library is required.

Coverage includes:

1. Parsing the representative official request fixture.
2. Defaults for optional data missing from documented samples.
3. Rejection of missing required top-level fields and invalid primitive types.
4. Day/night boundary calculations.
5. Serialization of every supported action shape.
6. Structural rejection of actions missing required fields.
7. Observation-aware rejection of invalid actors, phases, coordinates, and controller conflicts.
8. Preservation of valid actions when a sibling action is invalid.
9. Complete safe-response serialization.
10. An HTTP round trip for a valid request.
11. Safe HTTP responses for malformed JSON and controller exceptions.
12. Repeated requests proving request-local state.

The complete test suite runs with:

```powershell
python -m unittest discover -s tests -v
```

## Acceptance Criteria

- `python main.py <port>` starts the service on `0.0.0.0`.
- A representative judge request produces a complete, valid JSON response.
- Invalid input and internal exceptions produce the deterministic safe response without terminating the server.
- Parsed observations contain no mutable raw request dictionaries.
- Invalid actions cannot reach the serializer output.
- All standard-library unit and HTTP integration tests pass.
- No strategy subsystem or placeholder module is introduced in Phase 1.

## Deferred Work

Phase 2 will add the first deterministic playable policy on top of these interfaces: occupancy, pathfinding, collision-aware joint movement, basic mining, construction, and safe night positioning. Exact combat, survival scenarios, robust beam search, task skills, enemy beliefs, summon optimization, cross-half learning, and PSRO remain separate later phases.
