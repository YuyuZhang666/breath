# Phase 1 Reliable Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standard-library-only Future War HTTP agent that parses judge requests into immutable models, validates typed decisions, serializes safe responses, and survives malformed requests and internal failures.

**Architecture:** Keep transport, protocol, decision, validation, and control boundaries separate. The Phase 1 controller intentionally produces no game strategy; it proves the complete request-to-safe-response path that later strategy phases will use.

**Tech Stack:** Python 3.11+ standard library, `dataclasses`, `enum`, `http.server`, `json`, `logging`, `types.MappingProxyType`, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-1-foundation-design.md`

## Global Constraints

- Runtime and tests use only the Python standard library.
- The service starts with `python main.py <port>` and listens on `0.0.0.0:<port>`.
- Formal interface documentation and live request fields take precedence over constants in the supplied demo.
- Unknown request fields are ignored.
- No mining, movement policy, economy, combat search, task solving, opponent modeling, or self-play strategy is implemented in Phase 1.
- Every production behavior follows a witnessed failing test.
- Do not push to `origin`; create local commits only.

## File Map

- `main.py`: command-line entry point.
- `.gitignore`: Python cache and local environment exclusions.
- `future_war_agent/__init__.py`: package marker and public version.
- `future_war_agent/protocol/time.py`: one-based round, day, and phase calculations.
- `future_war_agent/protocol/models.py`: immutable observation-domain dataclasses.
- `future_war_agent/protocol/parser.py`: strict required-field parsing with safe optional defaults.
- `future_war_agent/decision/actions.py`: typed action kinds and action values.
- `future_war_agent/decision/decision.py`: immutable per-turn decision.
- `future_war_agent/decision/serializer.py`: action and response payload generation.
- `future_war_agent/decision/validator.py`: structural and observation-aware decision filtering.
- `future_war_agent/fallback.py`: deterministic empty decision and safe payload.
- `future_war_agent/controller.py`: parse, plan, validate, serialize orchestration.
- `future_war_agent/server.py`: standard-library threaded HTTP service.
- `tests/fixtures/request.json`: compact representative judge observation.
- `tests/test_time.py`: round-boundary tests.
- `tests/test_parser.py`: protocol parsing tests.
- `tests/test_actions.py`: action construction tests.
- `tests/test_serializer.py`: exact response-shape tests.
- `tests/test_validator.py`: decision legality filtering tests.
- `tests/test_controller.py`: fallback and orchestration tests.
- `tests/test_main.py`: command-line parsing and runner handoff tests.
- `tests/test_server.py`: real HTTP integration tests.

---

### Task 1: Project Package and Time Model

**Files:**
- Create: `.gitignore`
- Create: `future_war_agent/__init__.py`
- Create: `future_war_agent/protocol/__init__.py`
- Create: `future_war_agent/protocol/time.py`
- Create: `tests/__init__.py`
- Create: `tests/test_time.py`

**Interfaces:**
- Produces: `Phase`, `TurnTime`, `TurnTime.from_round(round_no: int) -> TurnTime`.
- Consumes: nothing outside the standard library.

- [ ] **Step 1: Write the failing time-model tests**

```python
# tests/test_time.py
import unittest

from future_war_agent.protocol.time import Phase, TurnTime


class TurnTimeTests(unittest.TestCase):
    def test_round_boundaries(self) -> None:
        cases = {
            1: (1, Phase.DAY, 1, 0),
            70: (1, Phase.DAY, 70, 69),
            71: (1, Phase.NIGHT, 1, 70),
            130: (1, Phase.NIGHT, 60, 129),
            131: (2, Phase.DAY, 1, 0),
        }

        for round_no, expected in cases.items():
            with self.subTest(round_no=round_no):
                turn = TurnTime.from_round(round_no)
                self.assertEqual(
                    (turn.day_no, turn.phase, turn.round_in_phase, turn.offset_in_day),
                    expected,
                )

    def test_round_number_must_be_positive_integer(self) -> None:
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    TurnTime.from_round(invalid)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and witness the expected import failure**

Run:

```powershell
python -m unittest tests.test_time -v
```

Expected: `ModuleNotFoundError` for `future_war_agent.protocol.time`.

- [ ] **Step 3: Implement the minimal package and time model**

```python
# future_war_agent/protocol/time.py
from dataclasses import dataclass
from enum import StrEnum

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS


class Phase(StrEnum):
    DAY = "day"
    NIGHT = "night"


@dataclass(frozen=True, slots=True)
class TurnTime:
    round_no: int
    day_no: int
    phase: Phase
    round_in_phase: int
    offset_in_day: int

    @classmethod
    def from_round(cls, round_no: int) -> "TurnTime":
        if isinstance(round_no, bool) or not isinstance(round_no, int):
            raise TypeError("round number must be an integer")
        if round_no < 1:
            raise ValueError("round number must be positive")
        zero_based = round_no - 1
        day_no = zero_based // ROUNDS_PER_DAY + 1
        offset = zero_based % ROUNDS_PER_DAY
        if offset < DAY_ROUNDS:
            phase = Phase.DAY
            round_in_phase = offset + 1
        else:
            phase = Phase.NIGHT
            round_in_phase = offset - DAY_ROUNDS + 1
        return cls(round_no, day_no, phase, round_in_phase, offset)
```

Create empty package markers, set `__version__ = "0.1.0"` in `future_war_agent/__init__.py`, and add this `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.coverage
htmlcov/
```

- [ ] **Step 4: Run the task tests and full discovery**

```powershell
python -m unittest tests.test_time -v
python -m unittest discover -s tests -v
```

Expected: 2 tests pass with no warnings or tracebacks.

- [ ] **Step 5: Commit the time foundation**

```powershell
git add .gitignore future_war_agent tests
git commit -m "feat: add round time model"
```

---

### Task 2: Immutable Observation Models and Parser

**Files:**
- Create: `future_war_agent/protocol/models.py`
- Create: `future_war_agent/protocol/parser.py`
- Create: `tests/fixtures/request.json`
- Create: `tests/test_parser.py`

**Interfaces:**
- Consumes: `TurnTime.from_round` from Task 1.
- Produces: `ProtocolError`, `Position.chebyshev_distance`, all observation dataclasses, and `parse_observation(raw: object) -> Observation`.

- [ ] **Step 1: Add the representative fixture**

Use a valid compact fixture containing every documented top-level area and both documented/sample inconsistencies:

```json
{
  "roundNo": 85,
  "mapInfo": {
    "width": 41,
    "height": 32,
    "zones": [
      {"neutralType": "stone", "pos": {"x": 4, "y": 24}},
      {"neutralType": "weaponShop", "pos": {"x": 25, "y": 20}}
    ]
  },
  "teamOur": {
    "type": "challenger",
    "teamId": "6324",
    "teamName": "Challenger",
    "goldNum": 20,
    "totalScore": 280,
    "playerTasks": [
      {
        "taskType": "自进化类1",
        "taskPosition": {"x": 14, "y": 14},
        "coldDownRounds": 0,
        "scoreReward": 50,
        "goldReward": 30,
        "isValid": true
      }
    ],
    "roles": [
      {
        "id": 10010,
        "pos": {"x": 5, "y": 23},
        "roleType": "worker",
        "health": 220,
        "attackPower": 0,
        "attackRange": 0,
        "backPackCapability": 100,
        "backpack": ["stone"]
      },
      {
        "id": 10020,
        "pos": {"x": 6, "y": 23},
        "roleType": "gatling",
        "health": 1000,
        "attackPower": 10,
        "attackRange": 4,
        "level": 1,
        "cooldown": 0
      }
    ]
  },
  "teamEnemy": {"roles": []},
  "robot": {
    "roles": [
      {
        "id": 30001,
        "pos": {"x": 4, "y": 4},
        "roleType": "smallRobot",
        "health": 40,
        "abnormalState": ""
      }
    ]
  },
  "phaseTask": "",
  "lastRoundRoleActionResults": {"10010": true},
  "lastSummonTreasureResult": 0,
  "llmResp": "",
  "worldNews": {"officialNews": "今日无重大新闻", "folkLegends": ""},
  "lastCmdResult": "",
  "vendorShopList": [{"name": "stone", "price": 1}],
  "weaponShopList": [{"name": "Medicine", "price": 10}],
  "errors": []
}
```

The task intentionally omits `timeoutRounds`; the robot intentionally omits `targetTeam`.

- [ ] **Step 2: Write failing parser tests**

```python
# tests/test_parser.py
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from future_war_agent.protocol.parser import ProtocolError, parse_observation

FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ObservationParserTests(unittest.TestCase):
    def test_parses_representative_request(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        observed = parse_observation(raw)

        self.assertEqual(observed.time.round_no, 85)
        self.assertEqual(observed.width, 41)
        self.assertEqual(observed.our.gold, 20)
        self.assertEqual(observed.our.units[0].unit_id, 10010)
        self.assertEqual(observed.our.tasks[0].timeout_rounds, None)
        self.assertEqual(observed.robots[0].target_team, None)
        self.assertEqual(observed.last_action_results[10010], True)

    def test_optional_sections_default_to_empty_values(self) -> None:
        observed = parse_observation({
            "roundNo": 1,
            "mapInfo": {"width": 41, "height": 32},
            "teamOur": {},
        })

        self.assertEqual(observed.zones, ())
        self.assertEqual(observed.enemy.units, ())
        self.assertEqual(observed.robots, ())
        self.assertEqual(dict(observed.last_action_results), {})
        self.assertEqual(observed.world_news.official_news, "")

    def test_required_top_level_data_is_enforced(self) -> None:
        for raw in ({}, {"roundNo": 1}, {"roundNo": 1, "mapInfo": {}}):
            with self.subTest(raw=raw):
                with self.assertRaises(ProtocolError):
                    parse_observation(raw)

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_observation({
                "roundNo": True,
                "mapInfo": {"width": 41, "height": 32},
                "teamOur": {},
            })

    def test_models_and_collections_are_immutable(self) -> None:
        observed = parse_observation({
            "roundNo": 1,
            "mapInfo": {"width": 41, "height": 32},
            "teamOur": {},
        })

        with self.assertRaises(FrozenInstanceError):
            observed.width = 99
        with self.assertRaises(TypeError):
            observed.last_action_results[10010] = True


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the parser tests and witness the import failure**

```powershell
python -m unittest tests.test_parser -v
```

Expected: import failure for `future_war_agent.protocol.parser`.

- [ ] **Step 4: Implement the immutable model surface**

Create frozen, slotted dataclasses with these exact fields:

```python
@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def chebyshev_distance(self, other: "Position") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))


@dataclass(frozen=True, slots=True)
class Zone:
    position: Position
    neutral_type: str


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


@dataclass(frozen=True, slots=True)
class RobotState:
    robot_id: int
    position: Position
    role_type: str
    health: int
    abnormal_state: str = ""
    target_team: str | None = None


@dataclass(frozen=True, slots=True)
class TaskPointState:
    task_type: str
    position: Position
    cooldown_rounds: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int | None = None


@dataclass(frozen=True, slots=True)
class OurTeamState:
    team_type: str = ""
    team_id: str = ""
    team_name: str = ""
    gold: int = 0
    total_score: int = 0
    tasks: tuple[TaskPointState, ...] = ()
    units: tuple[UnitState, ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyTeamState:
    units: tuple[UnitState, ...] = ()


@dataclass(frozen=True, slots=True)
class WorldNews:
    official_news: str = ""
    folk_legends: str = ""


@dataclass(frozen=True, slots=True)
class ShopItem:
    name: str
    price: int


@dataclass(frozen=True, slots=True)
class GameError:
    error_code: int
    description: str


@dataclass(frozen=True, slots=True)
class Observation:
    time: TurnTime
    width: int
    height: int
    zones: tuple[Zone, ...]
    our: OurTeamState
    enemy: EnemyTeamState
    robots: tuple[RobotState, ...]
    phase_task: str
    last_action_results: Mapping[int, bool]
    last_summon_treasure_result: int
    llm_response: str
    world_news: WorldNews
    last_command_result: str
    vendor_shop: tuple[ShopItem, ...]
    weapon_shop: tuple[ShopItem, ...]
    errors: tuple[GameError, ...]
```

Store `last_action_results` with `MappingProxyType(dict(values))` in `Observation.__post_init__` using `object.__setattr__`.

- [ ] **Step 5: Implement strict helpers and `parse_observation`**

Define `ProtocolError(ValueError)` and helpers that:

- require mappings where objects are expected;
- reject booleans when parsing integers;
- require `Position.x`, `Position.y`, `Unit.id`, `Unit.pos`, `Unit.roleType`, `Unit.health`, `Robot.id`, `Robot.pos`, `Robot.roleType`, and `Robot.health`;
- default absent optional sequences to empty tuples;
- require actual booleans for `TaskPoint.isValid` and previous-action results;
- ignore unknown keys.

`parse_observation` must start with:

```python
def parse_observation(raw: object) -> Observation:
    root = _mapping(raw, "request")
    round_no = _required_int(root, "roundNo", "request")
    map_info = _required_mapping(root, "mapInfo", "request")
    team_our = _required_mapping(root, "teamOur", "request")
```

It must finish by constructing `Observation` exclusively from parsed dataclasses, tuples, strings, integers, booleans, and an immutable action-result mapping.

- [ ] **Step 6: Run parser tests and the complete suite**

```powershell
python -m unittest tests.test_parser -v
python -m unittest discover -s tests -v
```

Expected: all 7 accumulated tests pass.

- [ ] **Step 7: Commit protocol parsing**

```powershell
git add future_war_agent/protocol tests/fixtures tests/test_parser.py
git commit -m "feat: parse immutable observations"
```

---

### Task 3: Typed Actions

**Files:**
- Create: `future_war_agent/decision/__init__.py`
- Create: `future_war_agent/decision/actions.py`
- Create: `tests/test_actions.py`

**Interfaces:**
- Consumes: `Position` from Task 2.
- Produces: `ActionKind`, immutable `Action`, and one named constructor for every supported action.

- [ ] **Step 1: Write failing action-construction tests**

```python
# tests/test_actions.py
import unittest

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.protocol.models import Position


class ActionTests(unittest.TestCase):
    def test_named_constructors_cover_protocol_actions(self) -> None:
        pos = Position(3, 4)
        cases = {
            ActionKind.MOVE: Action.move(pos),
            ActionKind.ATTACK: Action.attack(10010, (pos,)),
            ActionKind.SELL: Action.sell("stone", 2),
            ActionKind.BUY: Action.buy("Medicine", 1),
            ActionKind.BUILD: Action.build("wall", pos),
            ActionKind.REMOVE: Action.remove(pos),
            ActionKind.ACCEPT_TASK: Action.accept_task(),
            ActionKind.SUBMIT_ANSWER: Action.submit_answer("answer"),
            ActionKind.SUMMON_TREASURE: Action.summon_treasure(pos, ("StarSand",)),
            ActionKind.USE: Action.use("Medicine"),
            ActionKind.DROP: Action.drop("stone"),
            ActionKind.COLLECT: Action.collect(pos),
        }

        self.assertEqual(set(cases), set(ActionKind))
        for kind, action in cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(action.kind, kind)

    def test_action_copies_iterables_to_tuples(self) -> None:
        targets = [Position(1, 2)]
        items = ["StarSand"]
        action = Action.summon_treasure(targets[0], items)
        targets.append(Position(2, 3))
        items.append("FlameBreath")

        self.assertEqual(action.target_positions, (Position(1, 2),))
        self.assertEqual(action.items, ("StarSand",))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_actions -v
```

- [ ] **Step 3: Implement action kinds and the immutable value object**

Use the exact wire values in this enum:

```python
class ActionKind(StrEnum):
    MOVE = "move"
    ATTACK = "attack"
    SELL = "sell"
    BUY = "buy"
    BUILD = "build"
    REMOVE = "remove"
    ACCEPT_TASK = "acceptTask"
    SUBMIT_ANSWER = "submitAnswer"
    SUMMON_TREASURE = "summonTreasure"
    USE = "use"
    DROP = "drop"
    COLLECT = "collect"
```

Use one frozen, slotted value object:

```python
@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    target_positions: tuple[Position, ...] = ()
    controller_id: int | None = None
    name: str | None = None
    quantity: int | None = None
    task_answer: str | None = None
    items: tuple[str, ...] = ()
```

Each named constructor populates only fields belonging to that protocol action and converts incoming iterables with `tuple(...)`. Constructors do not enforce game legality; Task 5 owns legality.

- [ ] **Step 4: Run action tests and full discovery**

```powershell
python -m unittest tests.test_actions -v
python -m unittest discover -s tests -v
```

Expected: all 9 accumulated tests pass.

- [ ] **Step 5: Commit typed actions**

```powershell
git add future_war_agent/decision tests/test_actions.py
git commit -m "feat: add typed decision actions"
```

---

### Task 4: Decision and Serialization

**Files:**
- Create: `future_war_agent/decision/decision.py`
- Create: `future_war_agent/decision/serializer.py`
- Create: `future_war_agent/fallback.py`
- Create: `tests/test_serializer.py`

**Interfaces:**
- Consumes: `Action` and `Position`.
- Produces: immutable `Decision`, `action_to_payload`, `decision_to_payload`, `safe_decision`, and `safe_payload`.

- [ ] **Step 1: Write failing serialization tests**

```python
# tests/test_serializer.py
import unittest

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.serializer import action_to_payload, decision_to_payload
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Position


class SerializerTests(unittest.TestCase):
    def test_serializes_protocol_field_names(self) -> None:
        pos = Position(3, 4)
        decision = Decision(
            commands={
                10020: Action.attack(10010, (pos,)),
                10011: Action.submit_answer("answer"),
            },
            prompt="prompt text",
            execute_command="python solve.py",
        )

        self.assertEqual(decision_to_payload(decision), {
            "roleCommandMap": {
                "10011": {"action": "submitAnswer", "taskAnswer": "answer"},
                "10020": {
                    "action": "attack",
                    "controllerId": "10010",
                    "targetPos": [{"x": 3, "y": 4}],
                },
            },
            "prompt": "prompt text",
            "executeCmd": "python solve.py",
        })

    def test_safe_payload_is_complete(self) -> None:
        self.assertEqual(safe_payload(), {
            "roleCommandMap": {},
            "prompt": "",
            "executeCmd": "",
        })

    def test_serializes_every_action_shape(self) -> None:
        pos = Position(3, 4)
        cases = (
            (Action.move(pos), {"action": "move", "targetPos": [{"x": 3, "y": 4}]}),
            (Action.attack(10010, (pos,)), {
                "action": "attack", "controllerId": "10010",
                "targetPos": [{"x": 3, "y": 4}],
            }),
            (Action.sell("stone", 2), {"action": "sell", "name": "stone", "num": 2}),
            (Action.buy("Medicine", 1), {"action": "buy", "name": "Medicine", "num": 1}),
            (Action.build("wall", pos), {
                "action": "build", "name": "wall",
                "targetPos": [{"x": 3, "y": 4}],
            }),
            (Action.remove(pos), {"action": "remove", "targetPos": [{"x": 3, "y": 4}]}),
            (Action.accept_task(), {"action": "acceptTask"}),
            (Action.submit_answer("answer"), {"action": "submitAnswer", "taskAnswer": "answer"}),
            (Action.summon_treasure(pos, ("StarSand",)), {
                "action": "summonTreasure", "targetPos": [{"x": 3, "y": 4}],
                "item": ["StarSand"],
            }),
            (Action.use("Medicine"), {"action": "use", "name": "Medicine"}),
            (Action.drop("stone"), {"action": "drop", "name": "stone"}),
            (Action.collect(pos), {"action": "collect", "targetPos": [{"x": 3, "y": 4}]}),
        )
        for action, expected in cases:
            with self.subTest(action=action.kind):
                self.assertEqual(action_to_payload(action), expected)

    def test_decision_copies_command_mapping(self) -> None:
        commands = {10010: Action.move(Position(1, 1))}
        decision = Decision(commands=commands)
        commands.clear()
        self.assertIn(10010, decision.commands)
        with self.assertRaises(TypeError):
            decision.commands[10011] = Action.accept_task()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_serializer -v
```

- [ ] **Step 3: Implement immutable `Decision` and safe fallback**

```python
@dataclass(frozen=True, slots=True)
class Decision:
    commands: Mapping[int, Action] = field(default_factory=dict)
    prompt: str = ""
    execute_command: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "commands", MappingProxyType(dict(self.commands)))
```

`safe_decision()` returns `Decision()`. `safe_payload()` returns a fresh three-field dictionary on every call.

- [ ] **Step 4: Implement deterministic serialization**

`action_to_payload` starts with `{"action": action.kind.value}` and adds only non-empty fields using exact wire names: `controllerId`, `targetPos`, `name`, `num`, `taskAnswer`, and `item`. `decision_to_payload` sorts integer actor IDs before creating string keys, then always includes `prompt` and `executeCmd`.

- [ ] **Step 5: Run serializer tests and full discovery**

```powershell
python -m unittest tests.test_serializer -v
python -m unittest discover -s tests -v
```

Expected: all 13 accumulated tests pass.

- [ ] **Step 6: Commit decision serialization**

```powershell
git add future_war_agent tests/test_serializer.py
git commit -m "feat: serialize safe decisions"
```

---

### Task 5: Structural and Observation-Aware Validation

**Files:**
- Create: `future_war_agent/decision/validator.py`
- Create: `tests/test_validator.py`

**Interfaces:**
- Consumes: `Observation`, `Decision`, `Action`, `ActionKind`, `Phase`.
- Produces: `validate_decision(observation: Observation, decision: Decision) -> Decision`.

- [ ] **Step 1: Write failing validation tests**

Load the fixture once in `setUp`, then exercise real decisions:

```python
# tests/test_validator.py
import json
import unittest
from dataclasses import replace
from pathlib import Path

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime

FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = parse_observation(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def test_keeps_valid_night_attack(self) -> None:
        decision = Decision(commands={
            10020: Action.attack(10010, (Position(4, 4),)),
        })
        validated = validate_decision(self.observed, decision)
        self.assertIn(10020, validated.commands)

    def test_drops_day_attack_but_keeps_valid_move(self) -> None:
        day = replace(self.observed, time=TurnTime.from_round(1))
        decision = Decision(commands={
            10020: Action.attack(10010, (Position(4, 4),)),
            10010: Action.move(Position(5, 24)),
        })
        validated = validate_decision(day, decision)
        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_drops_out_of_bounds_target(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(41, 0))})
        self.assertEqual(validate_decision(self.observed, decision).commands, {})

    def test_drops_attack_when_controller_has_personal_action(self) -> None:
        decision = Decision(commands={
            10020: Action.attack(10010, (Position(4, 4),)),
            10010: Action.move(Position(5, 24)),
        })
        validated = validate_decision(self.observed, decision)
        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_drops_structurally_incomplete_action(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(5, 24))})
        malformed = replace(decision.commands[10010], target_positions=())
        validated = validate_decision(
            self.observed,
            Decision(commands={10010: malformed}),
        )
        self.assertEqual(validated.commands, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_validator -v
```

- [ ] **Step 3: Implement structural validation**

Implement `_is_structurally_valid(action: Action) -> bool` with these exact rules:

- `move`, `build`, `remove`, `collect`, and `summonTreasure` require exactly one target;
- `attack` requires at least one target and a controller ID;
- `sell` and `buy` require a non-blank name and positive non-boolean quantity;
- `build`, `use`, and `drop` require a non-blank name;
- `submitAnswer` requires a non-blank answer;
- `summonTreasure` requires at least one non-blank item;
- `acceptTask` has no required payload fields;
- fields not used by an action cause that action to be rejected rather than silently serialized.

- [ ] **Step 4: Implement observation-aware filtering**

Build an index of our units. Process personal commands before attacks, then process attack commands in sorted weapon-ID order.

Apply these rules:

- every command key names a living own unit;
- every target is inside `0 <= x < width` and `0 <= y < height`;
- personal actions are issued only by `worker` or `pioneer`;
- `collect`, `build`, and `remove` require a worker;
- `acceptTask`, `submitAnswer`, and `summonTreasure` require a pioneer;
- `build` requires day;
- `attack` requires night and a `gatling`, `railgun`, or `rocket` command key;
- attack controller names a living worker or pioneer within Chebyshev distance one of the weapon;
- a controller with a retained personal command cannot control a weapon;
- a controller retained for one attack is unavailable for subsequent attacks.

Return a new `Decision` containing the valid subset and the original prompt and execute command strings.

- [ ] **Step 5: Run validator tests and full discovery**

```powershell
python -m unittest tests.test_validator -v
python -m unittest discover -s tests -v
```

Expected: all 18 accumulated tests pass.

- [ ] **Step 6: Commit validation**

```powershell
git add future_war_agent/decision/validator.py tests/test_validator.py
git commit -m "feat: validate decisions against observations"
```

---

### Task 6: Controller and Safe Failure Boundary

**Files:**
- Create: `future_war_agent/controller.py`
- Create: `tests/test_controller.py`

**Interfaces:**
- Consumes: parser, `Decision`, validator, serializer, and safe fallback.
- Produces: `default_planner(observation: Observation) -> Decision` and `handle_payload(payload: object, planner: Planner = default_planner) -> dict[str, object]`.

- [ ] **Step 1: Write failing controller tests**

```python
# tests/test_controller.py
import json
import unittest
from pathlib import Path

from future_war_agent.controller import handle_payload
from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.fallback import safe_payload
from future_war_agent.protocol.models import Observation, Position

FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_default_controller_returns_safe_payload(self) -> None:
        self.assertEqual(handle_payload(self.payload), safe_payload())

    def test_injected_planner_flows_through_validation(self) -> None:
        def planner(observed: Observation) -> Decision:
            self.assertEqual(observed.time.round_no, 85)
            return Decision(commands={10010: Action.move(Position(5, 24))})

        response = handle_payload(self.payload, planner=planner)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")

    def test_invalid_payload_returns_safe_payload(self) -> None:
        self.assertEqual(handle_payload({}), safe_payload())

    def test_planner_exception_returns_safe_payload(self) -> None:
        def broken(_: Observation) -> Decision:
            raise RuntimeError("private failure detail")

        self.assertEqual(handle_payload(self.payload, planner=broken), safe_payload())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_controller -v
```

- [ ] **Step 3: Implement controller orchestration**

Define:

```python
Planner = Callable[[Observation], Decision]


def default_planner(_: Observation) -> Decision:
    return safe_decision()


def handle_payload(payload: object, planner: Planner = default_planner) -> dict[str, object]:
    try:
        observation = parse_observation(payload)
        decision = planner(observation)
        validated = validate_decision(observation, decision)
        return decision_to_payload(validated)
    except Exception:
        LOGGER.exception("turn handling failed")
        return safe_payload()
```

Do not include exception text in the returned payload.

- [ ] **Step 4: Run controller tests and full discovery**

```powershell
python -m unittest tests.test_controller -v
python -m unittest discover -s tests -v
```

Expected: all 22 accumulated tests pass.

- [ ] **Step 5: Commit controller boundary**

```powershell
git add future_war_agent/controller.py tests/test_controller.py
git commit -m "feat: add safe turn controller"
```

---

### Task 7: HTTP Server and Command-Line Entry Point

**Files:**
- Create: `future_war_agent/server.py`
- Create: `main.py`
- Create: `tests/test_main.py`
- Create: `tests/test_server.py`

**Interfaces:**
- Consumes: `handle_payload` and `safe_payload`.
- Produces: `create_server(port: int, controller: Controller = handle_payload, host: str = "0.0.0.0") -> ThreadingHTTPServer`, `serve(port: int) -> None`, `parse_port(argv: Sequence[str]) -> int`, and `main(argv: Sequence[str] | None = None, runner: Callable[[int], None] = serve) -> int`.

- [ ] **Step 1: Write failing real-HTTP tests**

```python
# tests/test_server.py
import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from future_war_agent.fallback import safe_payload
from future_war_agent.server import create_server

FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(0, host="127.0.0.1")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, body: bytes) -> tuple[int, dict[str, object]]:
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_valid_request_returns_complete_safe_response(self) -> None:
        status, payload = self.post(FIXTURE.read_bytes())
        self.assertEqual(status, 200)
        self.assertEqual(payload, safe_payload())

    def test_malformed_json_returns_safe_response(self) -> None:
        status, payload = self.post(b"{not json")
        self.assertEqual(status, 200)
        self.assertEqual(payload, safe_payload())

    def test_repeated_requests_remain_independent(self) -> None:
        first = self.post(FIXTURE.read_bytes())
        second = self.post(FIXTURE.read_bytes())
        self.assertEqual(first, second)

    def test_controller_exception_returns_safe_response(self) -> None:
        def broken_controller(_: object) -> dict[str, object]:
            raise RuntimeError("private failure detail")

        server = create_server(0, controller=broken_controller, host="127.0.0.1")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            request = Request(url, data=FIXTURE.read_bytes(), method="POST")
            with urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload, safe_payload())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and witness the missing-module failure**

```powershell
python -m unittest tests.test_server -v
```

- [ ] **Step 3: Implement the HTTP handler factory**

`create_server` validates `0 <= port <= 65535`, closes over the injected controller, and returns a `ThreadingHTTPServer`. The handler:

1. reads exactly `Content-Length` bytes;
2. decodes UTF-8 and parses JSON;
3. calls the controller;
4. catches decode, JSON, controller, and serialization failures;
5. serializes a safe payload on failure;
6. sends status 200, `application/json; charset=utf-8`, and the correct byte `Content-Length`;
7. suppresses default access logging by overriding `log_message`.

- [ ] **Step 4: Implement the command-line entry point**

First write these tests in `tests/test_main.py` and witness their import failure:

```python
import unittest

from main import main, parse_port


class MainTests(unittest.TestCase):
    def test_parse_port_accepts_valid_port(self) -> None:
        self.assertEqual(parse_port(["18080"]), 18080)

    def test_parse_port_rejects_invalid_arguments(self) -> None:
        for argv in ([], ["1", "2"], ["abc"], ["0"], ["65536"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parse_port(argv)

    def test_main_passes_port_to_runner(self) -> None:
        seen: list[int] = []
        self.assertEqual(main(["18080"], runner=seen.append), 0)
        self.assertEqual(seen, [18080])


if __name__ == "__main__":
    unittest.main()
```

Then implement `main.py`. `parse_port` accepts exactly one decimal port in the inclusive range 1 through 65535. `main` configures stdout logging, calls the injected runner, and returns 0. Invalid argument count or port raises `SystemExit("Usage: python main.py <port>")` without starting the server.

- [ ] **Step 5: Run server tests and full discovery**

```powershell
python -m unittest tests.test_server -v
python -m unittest tests.test_main -v
python -m unittest discover -s tests -v
```

Expected: all 29 accumulated tests pass.

- [ ] **Step 6: Run syntax compilation and a manual startup probe**

```powershell
python -m compileall -q main.py future_war_agent tests
python main.py 18080
```

In a second terminal, send `tests/fixtures/request.json` to `http://127.0.0.1:18080/` and verify the three-field safe response. Stop the server with Ctrl+C after the probe.

- [ ] **Step 7: Commit the runnable Phase 1 service**

```powershell
git add main.py future_war_agent/server.py tests/test_main.py tests/test_server.py
git commit -m "feat: serve safe agent responses"
```

---

### Task 8: Phase 1 Acceptance Verification

**Files:**
- Verify all Phase 1 files; modify only if a failing acceptance check first demonstrates a defect.

**Interfaces:**
- Consumes: the entire Phase 1 package.
- Produces: fresh evidence that the design acceptance criteria are met.

- [ ] **Step 1: Run the complete standard-library test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: 29 tests pass, zero failures, zero errors.

- [ ] **Step 2: Compile every Python file**

```powershell
python -m compileall -q main.py future_war_agent tests
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Check repository whitespace and status**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and no uncommitted files.

- [ ] **Step 4: Compare implementation against the design acceptance criteria**

Confirm explicitly that:

- startup uses `python main.py <port>`;
- the server binds `0.0.0.0` by default;
- representative and malformed requests return complete JSON responses;
- raw mutable request dictionaries do not cross the parser boundary;
- validation filters invalid actions before serialization;
- no strategy or placeholder subsystem was added.

If any item fails, add a focused failing test, witness the failure, implement the minimum correction, rerun all verification, and create a local fix commit.
