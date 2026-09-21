from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
import re

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import (
    Observation,
    Position,
    TaskPointState,
    UnitState,
)
from future_war_agent.protocol.time import Phase

from .pathfinding import PathResult, path_to_interaction
from .policy import StrategicIntent
from .world import WorldGrid


MAX_ANSWER_LENGTH = 2048
MAX_PROMPT_LENGTH = 2048
MAX_NEWS_LENGTH = 512
MAX_COMMAND_LENGTH = 512
MAX_COMMAND_RESULT_LENGTH = 32_768
MAX_SOPS = 16
MAX_TASK_TYPE_LENGTH = 128
MAX_ANSWER_CANDIDATES = 16
MAX_SUBMITTED_ANSWERS = 32


class CommandResultKind(StrEnum):
    SUCCESS = 'success'
    ERROR = 'error'
    TIMEOUT = 'timeout'
    JUDGER_ERROR = 'judger_error'
    UNKNOWN = 'unknown'


@dataclass(frozen=True, slots=True)
class CommandResult:
    kind: CommandResultKind
    output: str
    exit_code: int | None = None
    truncated: bool = False


def parse_command_result(value: str) -> CommandResult:
    output = _bounded(value, MAX_COMMAND_RESULT_LENGTH)
    truncated = '[TRUNCATED]' in output
    if '[TIMEOUT]' in output:
        kind = CommandResultKind.TIMEOUT
        exit_code = None
    elif '[JUDGER_ERROR]' in output:
        kind = CommandResultKind.JUDGER_ERROR
        exit_code = None
    else:
        match = re.search(r'\[exitCode:(-?\d+)\]', output)
        exit_code = int(match.group(1)) if match is not None else None
        if exit_code is None:
            kind = CommandResultKind.UNKNOWN
        else:
            kind = (
                CommandResultKind.SUCCESS
                if exit_code == 0
                else CommandResultKind.ERROR
            )
    return CommandResult(kind, output, exit_code, truncated)


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    answer: str
    fingerprint: str
    source: str
    format_valid: bool
    field_coverage: int
    local_validation: CommandResultKind
    created_round: int
    submitted: bool = False

    @property
    def quality_key(self) -> tuple[int, int, int, int]:
        validation = {
            CommandResultKind.SUCCESS: 2,
            CommandResultKind.UNKNOWN: 1,
        }.get(self.local_validation, 0)
        return (
            int(self.format_valid),
            validation,
            self.field_coverage,
            self.created_round,
        )


@dataclass(frozen=True, slots=True)
class TaskAbandonPolicy:
    margin: int = 500
    cooldown_rounds: int = 30
    cooldown_cost_per_round: int = 20
    time_cost_per_round: int = 1

    def __post_init__(self) -> None:
        if min(
            self.margin,
            self.cooldown_rounds,
            self.cooldown_cost_per_round,
            self.time_cost_per_round,
        ) < 0:
            raise ValueError('task abandon policy values cannot be negative')

    def should_abandon(
        self,
        *,
        continue_value: int | None,
        alternative_value: int | None,
        remaining_rounds: int | None,
        survival_risk: int | None,
    ) -> bool:
        if (
            continue_value is None
            or alternative_value is None
            or remaining_rounds is None
            or survival_risk is None
        ):
            return False
        continue_ev = (
            continue_value
            - max(0, remaining_rounds) * self.time_cost_per_round
            - max(0, survival_risk)
        )
        abandon_ev = (
            alternative_value
            - self.cooldown_rounds * self.cooldown_cost_per_round
        )
        return abandon_ev > continue_ev + self.margin


@dataclass(frozen=True, slots=True)
class TaskSop:
    task_type: str
    answer: str
    task_fingerprint: str = ''
    task_template: str = ''
    sample_task: str = ''
    command_steps: int = 0

    def __post_init__(self) -> None:
        if not self.task_type.strip() or not self.answer.strip():
            raise ValueError('task SOP fields must be nonblank')
        if len(self.task_type) > MAX_TASK_TYPE_LENGTH:
            raise ValueError('task SOP type is too long')
        if len(self.answer) > MAX_ANSWER_LENGTH:
            raise ValueError('task SOP answer is too long')
        if len(self.sample_task) > MAX_PROMPT_LENGTH:
            raise ValueError('task SOP sample is too long')
        if self.command_steps < 0:
            raise ValueError('task SOP command steps cannot be negative')


@dataclass(frozen=True, slots=True)
class TaskAgentState:
    active_task_type: str = ''
    selected_task_position: Position | None = None
    selected_task_value: int | None = None
    accepted_round: int | None = None
    timeout_rounds: int | None = None
    deadline_round: int | None = None
    last_prompt_fingerprint: str = ''
    pending_prompt_fingerprint: str = ''
    pending_prompt_round: int | None = None
    prompt_attempts: int = 0
    best_answer: str = ''
    pending_task_type: str = ''
    pending_answer: str = ''
    pending_pioneer_id: int | None = None
    pending_round: int | None = None
    pending_task_text: str = ''
    pending_task_fingerprint: str = ''
    command_step: int = 0
    pending_command_round: int | None = None
    last_command_result: str = ''
    last_command_kind: CommandResultKind = CommandResultKind.UNKNOWN
    consumed_command_results: tuple[str, ...] = ()
    candidates: tuple[AnswerCandidate, ...] = ()
    submitted_answer_fingerprints: tuple[str, ...] = ()
    abandoned_task_fingerprint: str = ''
    abandoned_task_type: str = ''
    abandoned_task_position: Position | None = None
    abandon_until_round: int | None = None
    completed_task_fingerprint: str = ''
    sops: tuple[TaskSop, ...] = ()
    official_news: str = ''
    folk_legends: str = ''

    def __post_init__(self) -> None:
        if len(self.sops) > MAX_SOPS:
            raise ValueError('task SOP capacity exceeded')
        for value in (self.active_task_type, self.pending_task_type):
            if len(value) > MAX_TASK_TYPE_LENGTH:
                raise ValueError('task state type is too long')
        for value in (self.best_answer, self.pending_answer):
            if len(value) > MAX_ANSWER_LENGTH:
                raise ValueError('task state answer is too long')
        if len(self.pending_task_text) > MAX_PROMPT_LENGTH:
            raise ValueError('pending task text is too long')
        if len(self.official_news) > MAX_NEWS_LENGTH:
            raise ValueError('official news capacity exceeded')
        if len(self.folk_legends) > MAX_NEWS_LENGTH:
            raise ValueError('folk legends capacity exceeded')
        if self.command_step < 0:
            raise ValueError('command_step cannot be negative')
        if self.prompt_attempts < 0:
            raise ValueError('prompt_attempts cannot be negative')
        if len(self.last_command_result) > MAX_COMMAND_RESULT_LENGTH:
            raise ValueError('command result capacity exceeded')
        if len(self.candidates) > MAX_ANSWER_CANDIDATES:
            raise ValueError('answer candidate capacity exceeded')
        if len(self.submitted_answer_fingerprints) > MAX_SUBMITTED_ANSWERS:
            raise ValueError('submitted answer capacity exceeded')


EMPTY_TASK_STATE = TaskAgentState()


@dataclass(frozen=True, slots=True)
class TaskAgentResult:
    decision: Decision
    state: TaskAgentState


@dataclass(frozen=True, slots=True)
class _TaskCandidate:
    task: TaskPointState
    path: PathResult
    value: int

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            -self.value,
            self.path.cost,
            self.task.task_type,
            self.task.position.x,
            self.task.position.y,
        )


class TaskAgent:
    def __init__(
        self,
        *,
        execute_commands: tuple[str, ...] = (),
        abandon_policy: TaskAbandonPolicy | None = None,
        prompt_retry_rounds: int = 2,
    ) -> None:
        normalized = tuple(command.strip() for command in execute_commands)
        if any(
            not command
            or '\x00' in command
            or len(command) > MAX_COMMAND_LENGTH
            for command in normalized
        ):
            raise ValueError('execute commands must be bounded and nonblank')
        self._execute_commands = normalized
        self._abandon_policy = (
            abandon_policy if abandon_policy is not None else TaskAbandonPolicy()
        )
        if prompt_retry_rounds < 1:
            raise ValueError('prompt_retry_rounds must be positive')
        self._prompt_retry_rounds = prompt_retry_rounds

    def reconcile(
        self,
        observation: Observation,
        previous_state: TaskAgentState | None = None,
    ) -> TaskAgentState:
        state = previous_state if previous_state is not None else EMPTY_TASK_STATE
        state = _reconcile_feedback(observation, state)
        state = _reconcile_command_result(observation, state)
        return replace(
            state,
            official_news=_bounded(
                observation.world_news.official_news,
                MAX_NEWS_LENGTH,
            ),
            folk_legends=_bounded(
                observation.world_news.folk_legends,
                MAX_NEWS_LENGTH,
            ),
        )

    def apply(
        self,
        observation: Observation,
        base_decision: Decision,
        intent: StrategicIntent,
        *,
        previous_state: TaskAgentState | None = None,
    ) -> TaskAgentResult:
        state = self.reconcile(observation, previous_state)
        pioneer = _living_pioneer(observation)
        if pioneer is None:
            return TaskAgentResult(base_decision, state)
        if _has_emergency_item_action(base_decision, pioneer.unit_id):
            return TaskAgentResult(base_decision, state)
        if _has_twilight_recall(observation, base_decision, pioneer):
            return TaskAgentResult(base_decision, state)

        task_text = observation.phase_task.strip()
        if not intent.allow_tasks and not task_text:
            return TaskAgentResult(base_decision, state)
        if task_text:
            task_fingerprint = _fingerprint(task_text)
            if state.completed_task_fingerprint == task_fingerprint:
                return TaskAgentResult(base_decision, state)
            if (
                state.abandoned_task_fingerprint == task_fingerprint
                and state.abandon_until_round is not None
                and observation.time.round_no <= state.abandon_until_round
            ):
                return TaskAgentResult(base_decision, state)
            if intent.allow_tasks and _should_abandon_active_task(
                observation,
                pioneer,
                state,
                self._abandon_policy,
            ):
                abandoned_type = state.active_task_type
                abandoned_position = state.selected_task_position
                cleared = _clear_task_lifecycle(state)
                return TaskAgentResult(
                    base_decision,
                    replace(
                        cleared,
                        abandoned_task_fingerprint=task_fingerprint,
                        abandoned_task_type=abandoned_type,
                        abandoned_task_position=abandoned_position,
                        abandon_until_round=(
                            observation.time.round_no
                            + self._abandon_policy.cooldown_rounds
                        ),
                    ),
                )
            return self._handle_active_task(
                observation,
                base_decision,
                pioneer,
                state,
                task_text,
                intent,
            )

        if (
            state.accepted_round is not None
            and observation.time.round_no > state.accepted_round
        ):
            state = _clear_task_lifecycle(state)
        if state.completed_task_fingerprint:
            state = replace(
                state,
                completed_task_fingerprint='',
            )
        if (
            state.abandon_until_round is not None
            and observation.time.round_no > state.abandon_until_round
        ):
            state = replace(
                state,
                abandoned_task_fingerprint='',
                abandoned_task_type='',
                abandoned_task_position=None,
                abandon_until_round=None,
            )
        if state.accepted_round is not None:
            return TaskAgentResult(base_decision, state)
        selected = _continue_selected_task(observation, pioneer, state)
        if selected is None:
            selected = _select_task(
                observation,
                pioneer,
                excluded_task=(
                    state.abandoned_task_type,
                    state.abandoned_task_position,
                )
                if state.abandon_until_round is not None
                and observation.time.round_no <= state.abandon_until_round
                else None,
            )
        if selected is None:
            return TaskAgentResult(base_decision, state)
        task_type = _normalize_task_type(selected.task.task_type)
        state = replace(
            state,
            active_task_type=task_type,
            selected_task_position=selected.task.position,
            selected_task_value=_task_abandon_value(selected),
            timeout_rounds=selected.task.timeout_rounds,
        )
        if pioneer.position.chebyshev_distance(selected.task.position) <= 1:
            deadline_round = (
                observation.time.round_no + selected.task.timeout_rounds
                if selected.task.timeout_rounds is not None
                else None
            )
            state = replace(
                state,
                accepted_round=observation.time.round_no,
                deadline_round=deadline_round,
            )
            decision = _overlay(
                observation,
                base_decision,
                pioneer,
                Action.accept_task(),
            )
            return TaskAgentResult(decision, state)
        if len(selected.path.path) < 2:
            return TaskAgentResult(base_decision, state)
        decision = _overlay(
            observation,
            base_decision,
            pioneer,
            Action.move(selected.path.path[1]),
        )
        return TaskAgentResult(decision, state)

    def _handle_active_task(
        self,
        observation: Observation,
        base_decision: Decision,
        pioneer: UnitState,
        state: TaskAgentState,
        task_text: str,
        intent: StrategicIntent,
    ) -> TaskAgentResult:
        task_type = state.active_task_type or _task_key(task_text)
        fingerprint = _fingerprint(task_text)
        exact_sop = next(
            (
                sop
                for sop in state.sops
                if sop.task_type == task_type
                and sop.task_fingerprint == fingerprint
            ),
            None,
        )
        parameter_sop = next(
            (
                sop
                for sop in state.sops
                if sop.task_type == task_type
                and sop.task_template
                and sop.task_template == _task_template(task_text)
            ),
            None,
        )
        response_answer = (
            _bounded(observation.llm_response, MAX_ANSWER_LENGTH)
            if state.pending_prompt_fingerprint == fingerprint
            else ''
        )
        candidates = state.candidates
        if state.best_answer and not candidates:
            candidates = _upsert_candidate(
                candidates,
                _answer_candidate(
                    state.best_answer,
                    task_text,
                    source='legacy_best',
                    round_no=observation.time.round_no,
                    validation=CommandResultKind.UNKNOWN,
                ),
            )
        if response_answer:
            candidates = _upsert_candidate(
                candidates,
                _answer_candidate(
                    response_answer,
                    task_text,
                    source='llm',
                    round_no=observation.time.round_no,
                    validation=CommandResultKind.UNKNOWN,
                ),
            )
        if exact_sop is not None:
            candidates = _upsert_candidate(
                candidates,
                _answer_candidate(
                    exact_sop.answer,
                    task_text,
                    source='sop',
                    round_no=observation.time.round_no,
                    validation=CommandResultKind.SUCCESS,
                ),
            )

        best = _best_candidate(candidates)
        state = replace(
            state,
            candidates=candidates,
            best_answer=best.answer if best is not None else state.best_answer,
        )
        answer_candidate = _next_anytime_candidate(state)
        at_deadline = (
            state.deadline_round is not None
            and observation.time.round_no >= state.deadline_round
        )
        if answer_candidate is not None and (
            answer_candidate.source in {'sop', 'llm'} or at_deadline
        ):
            decision = _overlay(
                observation,
                base_decision,
                pioneer,
                Action.submit_answer(answer_candidate.answer),
            )
            candidates = tuple(
                replace(candidate, submitted=True)
                if candidate.fingerprint == answer_candidate.fingerprint
                else candidate
                for candidate in state.candidates
            )
            submitted = (
                answer_candidate.fingerprint,
                *(
                    value
                    for value in state.submitted_answer_fingerprints
                    if value != answer_candidate.fingerprint
                ),
            )[:MAX_SUBMITTED_ANSWERS]
            next_state = replace(
                state,
                active_task_type=task_type,
                last_prompt_fingerprint=fingerprint,
                best_answer=answer_candidate.answer,
                pending_task_type=task_type,
                pending_answer=answer_candidate.answer,
                pending_pioneer_id=pioneer.unit_id,
                pending_round=observation.time.round_no,
                pending_task_text=_bounded(task_text, MAX_PROMPT_LENGTH),
                pending_task_fingerprint=fingerprint,
                candidates=candidates,
                submitted_answer_fingerprints=submitted,
            )
            return TaskAgentResult(decision, next_state)

        waiting_for_response = (
            state.pending_prompt_fingerprint == fingerprint
            and state.pending_prompt_round is not None
            and observation.time.round_no
            <= state.pending_prompt_round + self._prompt_retry_rounds
        )
        if waiting_for_response:
            commands = {
                actor_id: action
                for actor_id, action in base_decision.commands.items()
                if actor_id != pioneer.unit_id
                and action.controller_id != pioneer.unit_id
            }
            return TaskAgentResult(Decision(commands=commands), state)

        prompt = _build_prompt(
            task_type,
            task_text,
            command_result=state.last_command_result,
            sop=parameter_sop,
        )
        execute_command = ''
        next_state = state
        if (
            intent.feature_flags.enable_task_execute_commands
            and (parameter_sop is None or state.prompt_attempts > 0)
            and state.pending_command_round is None
            and state.command_step < len(self._execute_commands)
        ):
            execute_command = self._execute_commands[state.command_step]
            next_state = replace(
                state,
                command_step=state.command_step + 1,
                pending_command_round=observation.time.round_no,
            )
        commands = {
            actor_id: action
            for actor_id, action in base_decision.commands.items()
            if actor_id != pioneer.unit_id
            and action.controller_id != pioneer.unit_id
        }
        decision = Decision(
            commands=commands,
            prompt=prompt,
            execute_command=execute_command,
        )
        return TaskAgentResult(
            decision,
            replace(
                next_state,
                active_task_type=task_type,
                last_prompt_fingerprint=fingerprint,
                pending_prompt_fingerprint=fingerprint,
                pending_prompt_round=observation.time.round_no,
                prompt_attempts=next_state.prompt_attempts + 1,
                command_step=(
                    max(next_state.command_step, parameter_sop.command_steps)
                    if parameter_sop is not None
                    else next_state.command_step
                ),
            ),
        )


def _living_pioneer(observation: Observation) -> UnitState | None:
    return next(
        (
            unit
            for unit in sorted(observation.our.units, key=lambda item: item.unit_id)
            if unit.health > 0 and unit.role_type == 'pioneer'
        ),
        None,
    )


def _has_emergency_item_action(
    decision: Decision,
    pioneer_id: int,
) -> bool:
    action = decision.commands.get(pioneer_id)
    return action is not None and action.kind is ActionKind.USE


def _has_twilight_recall(
    observation: Observation,
    decision: Decision,
    pioneer: UnitState,
) -> bool:
    action = decision.commands.get(pioneer.unit_id)
    if (
        observation.time.phase is not Phase.DAY
        or action is None
        or action.kind is not ActionKind.MOVE
    ):
        return False
    world = WorldGrid.from_observation(observation)
    rounds_left = 71 - observation.time.round_in_phase
    return any(
        path is not None
        and rounds_left <= path.cost + world.rules.twilight_safety_margin
        for weapon in world.weapons
        for path in (path_to_interaction(world, pioneer.position, weapon.position),)
    )


def _continue_selected_task(
    observation: Observation,
    pioneer: UnitState,
    state: TaskAgentState,
) -> _TaskCandidate | None:
    if not state.active_task_type or state.selected_task_position is None:
        return None
    world = WorldGrid.from_observation(observation)
    for task in observation.our.tasks:
        if (
            _normalize_task_type(task.task_type) == state.active_task_type
            and task.position == state.selected_task_position
        ):
            return _candidate_for_task(task, world, pioneer)
    return None


def _select_task(
    observation: Observation,
    pioneer: UnitState,
    *,
    excluded_task: tuple[str, Position | None] | None = None,
) -> _TaskCandidate | None:
    world = WorldGrid.from_observation(observation)
    candidates: list[_TaskCandidate] = []
    for task in observation.our.tasks:
        if excluded_task == (
            _normalize_task_type(task.task_type),
            task.position,
        ):
            continue
        candidate = _candidate_for_task(task, world, pioneer)
        if candidate is not None:
            candidates.append(candidate)
    return min(candidates, key=lambda item: item.sort_key) if candidates else None


def _should_abandon_active_task(
    observation: Observation,
    pioneer: UnitState,
    state: TaskAgentState,
    policy: TaskAbandonPolicy,
) -> bool:
    if (
        state.selected_task_value is None
        or state.best_answer
        or state.candidates
        or state.pending_pioneer_id is not None
    ):
        return False
    remaining = (
        max(0, state.deadline_round - observation.time.round_no)
        if state.deadline_round is not None
        else state.timeout_rounds
    )
    if remaining is None:
        return False
    world = WorldGrid.from_observation(observation)
    alternatives = tuple(
        _task_abandon_value(candidate)
        for task in observation.our.tasks
        for candidate in (_candidate_for_task(task, world, pioneer),)
        if candidate is not None
        and (
            task.position != state.selected_task_position
            or _normalize_task_type(task.task_type) != state.active_task_type
        )
    )
    return policy.should_abandon(
        continue_value=state.selected_task_value,
        alternative_value=max(alternatives) if alternatives else None,
        remaining_rounds=remaining,
        survival_risk=0,
    )


def _task_abandon_value(candidate: _TaskCandidate) -> int:
    task = candidate.task
    timeout = task.timeout_rounds if task.timeout_rounds is not None else 100
    urgency = max(0, 100 - min(timeout, 100))
    return (
        task.score_reward * 100
        + task.gold_reward
        + urgency
        - candidate.path.cost * 10
    )


def _candidate_for_task(
    task: TaskPointState,
    world: WorldGrid,
    pioneer: UnitState,
) -> _TaskCandidate | None:
    if (
        not task.is_valid
        or task.cooldown_rounds != 0
        or task.score_reward + task.gold_reward <= 0
    ):
        return None
    path = path_to_interaction(world, pioneer.position, task.position)
    if path is None:
        return None
    timeout = task.timeout_rounds if task.timeout_rounds is not None else 1000
    urgency = max(0, 100 - min(timeout, 100))
    value = (
        task.score_reward * 10_000
        + task.gold_reward * 100
        + urgency
        - path.cost * 10
    )
    return _TaskCandidate(task, path, value)


def _overlay(
    observation: Observation,
    base_decision: Decision,
    pioneer: UnitState,
    action: Action,
) -> Decision:
    commands = {
        actor_id: retained
        for actor_id, retained in base_decision.commands.items()
        if actor_id != pioneer.unit_id
        and retained.controller_id != pioneer.unit_id
    }
    if action.kind is ActionKind.MOVE:
        target = action.target_positions[0]
        occupied = {
            unit.position
            for unit in observation.our.units
            if unit.health > 0 and unit.unit_id != pioneer.unit_id
        }
        reserved = {
            retained.target_positions[0]
            for retained in commands.values()
            if retained.kind in {ActionKind.MOVE, ActionKind.BUILD}
            and retained.target_positions
        }
        if target in occupied or target in reserved:
            return base_decision
    commands[pioneer.unit_id] = action
    return Decision(
        commands=commands,
        prompt=base_decision.prompt,
        execute_command='',
    )


def _reconcile_feedback(
    observation: Observation,
    state: TaskAgentState,
) -> TaskAgentState:
    pioneer_id = state.pending_pioneer_id
    if pioneer_id is None:
        return state
    expected_round = (
        state.pending_round + 1
        if state.pending_round is not None
        else observation.time.round_no
    )
    if observation.time.round_no < expected_round:
        return state
    has_immediate_result = (
        observation.time.round_no == expected_round
        and pioneer_id in observation.last_action_results
    )
    sops = state.sops
    if has_immediate_result and observation.last_action_results[pioneer_id]:
        task_text = state.pending_task_text
        learned = TaskSop(
            state.pending_task_type,
            state.pending_answer,
            task_fingerprint=state.pending_task_fingerprint,
            task_template=_task_template(task_text) if task_text else '',
            sample_task=task_text,
            command_steps=state.command_step,
        )
        sops = (
            learned,
            *(
                sop
                for sop in sops
                if (
                    sop.task_type,
                    sop.task_template or sop.task_fingerprint,
                )
                != (
                    learned.task_type,
                    learned.task_template or learned.task_fingerprint,
                )
            ),
        )[:MAX_SOPS]
        return replace(
            _clear_task_lifecycle(state),
            sops=sops,
            completed_task_fingerprint=(
                _fingerprint(task_text) if task_text else ''
            ),
        )
    candidates = state.candidates
    if has_immediate_result:
        pending_fingerprint = _fingerprint(state.pending_answer)
        candidates = tuple(
            replace(candidate, local_validation=CommandResultKind.ERROR)
            if candidate.fingerprint == pending_fingerprint
            else candidate
            for candidate in candidates
        )
    return replace(
        state,
        pending_task_type='',
        pending_answer='',
        pending_pioneer_id=None,
        pending_round=None,
        pending_task_text='',
        pending_task_fingerprint='',
        candidates=candidates,
    )


def _reconcile_command_result(
    observation: Observation,
    state: TaskAgentState,
) -> TaskAgentState:
    if (
        state.pending_command_round is None
        or observation.time.round_no <= state.pending_command_round
    ):
        return state
    result = _bounded(observation.last_command_result, MAX_COMMAND_RESULT_LENGTH)
    parsed = parse_command_result(result if result else '[TIMEOUT] no result')
    fingerprint = _fingerprint(parsed.output)
    if fingerprint in state.consumed_command_results:
        return replace(
            state,
            pending_command_round=None,
            pending_prompt_fingerprint='',
            pending_prompt_round=None,
        )
    return replace(
        state,
        pending_command_round=None,
        last_command_result=parsed.output,
        last_command_kind=parsed.kind,
        consumed_command_results=(
            fingerprint,
            *(
                value
                for value in state.consumed_command_results
                if value != fingerprint
            ),
        )[:16],
        pending_prompt_fingerprint='',
        pending_prompt_round=None,
    )


def _clear_task_lifecycle(state: TaskAgentState) -> TaskAgentState:
    return replace(
        state,
        active_task_type='',
        selected_task_position=None,
        selected_task_value=None,
        accepted_round=None,
        timeout_rounds=None,
        deadline_round=None,
        last_prompt_fingerprint='',
        pending_prompt_fingerprint='',
        pending_prompt_round=None,
        prompt_attempts=0,
        best_answer='',
        pending_task_type='',
        pending_answer='',
        pending_pioneer_id=None,
        pending_round=None,
        pending_task_text='',
        pending_task_fingerprint='',
        command_step=0,
        pending_command_round=None,
        last_command_result='',
        last_command_kind=CommandResultKind.UNKNOWN,
        consumed_command_results=(),
        candidates=(),
        submitted_answer_fingerprints=(),
        abandoned_task_fingerprint='',
        abandoned_task_type='',
        abandoned_task_position=None,
        abandon_until_round=None,
        completed_task_fingerprint='',
    )


def _bounded(value: str, limit: int) -> str:
    return value.replace('\x00', '').strip()[:limit]


def _fingerprint(value: str) -> str:
    return sha256(value.encode('utf-8')).hexdigest()


def _task_template(task_text: str) -> str:
    normalized = ' '.join(task_text.replace('\x00', '').split())
    return re.sub(r'(?<![A-Za-z_])-?\d+(?:\.\d+)?', '<n>', normalized)


def _answer_candidate(
    answer: str,
    task_text: str,
    *,
    source: str,
    round_no: int,
    validation: CommandResultKind,
) -> AnswerCandidate:
    normalized = _bounded(answer, MAX_ANSWER_LENGTH)
    required_fields = tuple(
        dict.fromkeys(
            match
            for match in re.findall(r'[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]', task_text)
        )
    )
    if required_fields:
        coverage = sum(field in normalized for field in required_fields)
    else:
        coverage = int(bool(normalized))
    return AnswerCandidate(
        answer=normalized,
        fingerprint=_fingerprint(normalized),
        source=source,
        format_valid=bool(normalized) and '\x00' not in answer,
        field_coverage=coverage,
        local_validation=validation,
        created_round=round_no,
    )


def _upsert_candidate(
    candidates: tuple[AnswerCandidate, ...],
    candidate: AnswerCandidate,
) -> tuple[AnswerCandidate, ...]:
    existing = next(
        (
            value
            for value in candidates
            if value.fingerprint == candidate.fingerprint
        ),
        None,
    )
    if existing is not None:
        candidate = replace(
            candidate,
            submitted=existing.submitted,
            created_round=min(existing.created_round, candidate.created_round),
        )
    return (
        candidate,
        *(
            value
            for value in candidates
            if value.fingerprint != candidate.fingerprint
        ),
    )[:MAX_ANSWER_CANDIDATES]


def _best_candidate(
    candidates: tuple[AnswerCandidate, ...],
) -> AnswerCandidate | None:
    eligible = tuple(value for value in candidates if value.format_valid)
    return (
        max(
            eligible,
            key=lambda value: (value.quality_key, -value.created_round, value.fingerprint),
        )
        if eligible
        else None
    )


def _next_anytime_candidate(state: TaskAgentState) -> AnswerCandidate | None:
    submitted = tuple(
        value
        for value in state.candidates
        if value.submitted
        or value.fingerprint in state.submitted_answer_fingerprints
        if value.local_validation is not CommandResultKind.ERROR
    )
    submitted_quality = max(
        (value.quality_key for value in submitted),
        default=(-1, -1, -1, -1),
    )
    eligible = tuple(
        value
        for value in state.candidates
        if value.format_valid
        and not value.submitted
        and value.fingerprint not in state.submitted_answer_fingerprints
        and value.quality_key > submitted_quality
    )
    return _best_candidate(eligible)


def _task_key(task_text: str) -> str:
    return _normalize_task_type('text:' + task_text)


def _normalize_task_type(task_type: str) -> str:
    normalized = task_type.replace('\x00', '').strip()
    if len(normalized) <= MAX_TASK_TYPE_LENGTH:
        return normalized
    digest = sha256(normalized.encode('utf-8')).hexdigest()[:16]
    prefix_length = MAX_TASK_TYPE_LENGTH - len(digest) - 1
    return normalized[:prefix_length] + ':' + digest


def _build_prompt(
    task_type: str,
    task_text: str,
    *,
    command_result: str = '',
    sop: TaskSop | None = None,
) -> str:
    sandbox_context = (
        f'\nSandbox result:\n{command_result}' if command_result else ''
    )
    sop_context = (
        '\nA structurally similar task was solved successfully. Use it only as '
        'a method/example; recompute the answer for the current parameters.'
        f'\nExample task: {sop.sample_task}'
        f'\nExample answer: {sop.answer}'
        if sop is not None
        else ''
    )
    prompt = (
        'Solve the game task below. Return only the final answer, with no '
        'explanation, markdown, or shell commands.\n'
        f'Task type: {task_type}\n'
        f'Task: {task_text}'
        f'{sop_context}'
        f'{sandbox_context}'
    )
    return prompt[:MAX_PROMPT_LENGTH]
