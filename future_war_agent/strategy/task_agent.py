from dataclasses import dataclass, replace
from hashlib import sha256

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
MAX_SOPS = 16
MAX_TASK_TYPE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class TaskSop:
    task_type: str
    answer: str

    def __post_init__(self) -> None:
        if not self.task_type.strip() or not self.answer.strip():
            raise ValueError('task SOP fields must be nonblank')
        if len(self.task_type) > MAX_TASK_TYPE_LENGTH:
            raise ValueError('task SOP type is too long')
        if len(self.answer) > MAX_ANSWER_LENGTH:
            raise ValueError('task SOP answer is too long')


@dataclass(frozen=True, slots=True)
class TaskAgentState:
    active_task_type: str = ''
    selected_task_position: Position | None = None
    accepted_round: int | None = None
    timeout_rounds: int | None = None
    deadline_round: int | None = None
    last_prompt_fingerprint: str = ''
    best_answer: str = ''
    pending_task_type: str = ''
    pending_answer: str = ''
    pending_pioneer_id: int | None = None
    pending_round: int | None = None
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
        if len(self.official_news) > MAX_NEWS_LENGTH:
            raise ValueError('official news capacity exceeded')
        if len(self.folk_legends) > MAX_NEWS_LENGTH:
            raise ValueError('folk legends capacity exceeded')


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
    def apply(
        self,
        observation: Observation,
        base_decision: Decision,
        intent: StrategicIntent,
        *,
        previous_state: TaskAgentState | None = None,
    ) -> TaskAgentResult:
        state = previous_state if previous_state is not None else EMPTY_TASK_STATE
        state = _reconcile_feedback(observation, state)
        state = replace(
            state,
            official_news=_bounded(observation.world_news.official_news, MAX_NEWS_LENGTH),
            folk_legends=_bounded(observation.world_news.folk_legends, MAX_NEWS_LENGTH),
        )
        if not intent.allow_tasks:
            return TaskAgentResult(base_decision, state)

        pioneer = _living_pioneer(observation)
        if pioneer is None:
            return TaskAgentResult(base_decision, state)
        if _has_emergency_item_action(base_decision, pioneer.unit_id):
            return TaskAgentResult(base_decision, state)
        if _has_twilight_recall(observation, base_decision, pioneer):
            return TaskAgentResult(base_decision, state)

        task_text = observation.phase_task.strip()
        if task_text:
            return self._handle_active_task(
                observation,
                base_decision,
                pioneer,
                state,
                task_text,
            )

        if (
            state.accepted_round is not None
            and observation.time.round_no > state.accepted_round
        ):
            state = _clear_task_lifecycle(state)
        if state.accepted_round is not None:
            return TaskAgentResult(base_decision, state)
        selected = _continue_selected_task(observation, pioneer, state)
        if selected is None:
            selected = _select_task(observation, pioneer)
        if selected is None:
            return TaskAgentResult(base_decision, state)
        task_type = _normalize_task_type(selected.task.task_type)
        state = replace(
            state,
            active_task_type=task_type,
            selected_task_position=selected.task.position,
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
    ) -> TaskAgentResult:
        task_type = state.active_task_type or _task_key(task_text)
        fingerprint = sha256(task_text.encode('utf-8')).hexdigest()
        sop_answer = next(
            (sop.answer for sop in state.sops if sop.task_type == task_type),
            '',
        )
        response_answer = _bounded(observation.llm_response, MAX_ANSWER_LENGTH)
        deadline_answer = (
            state.best_answer
            if state.deadline_round is not None
            and observation.time.round_no >= state.deadline_round
            else ''
        )
        answer = sop_answer or response_answer or deadline_answer
        if answer:
            decision = _overlay(
                observation,
                base_decision,
                pioneer,
                Action.submit_answer(answer),
            )
            next_state = replace(
                state,
                active_task_type=task_type,
                last_prompt_fingerprint=fingerprint,
                best_answer=answer,
                pending_task_type=task_type,
                pending_answer=answer,
                pending_pioneer_id=pioneer.unit_id,
                pending_round=observation.time.round_no,
            )
            return TaskAgentResult(decision, next_state)

        prompt = _build_prompt(task_type, task_text)
        commands = {
            actor_id: action
            for actor_id, action in base_decision.commands.items()
            if actor_id != pioneer.unit_id
        }
        decision = Decision(commands=commands, prompt=prompt, execute_command='')
        return TaskAgentResult(
            decision,
            replace(
                state,
                active_task_type=task_type,
                last_prompt_fingerprint=fingerprint,
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
) -> _TaskCandidate | None:
    world = WorldGrid.from_observation(observation)
    candidates: list[_TaskCandidate] = []
    for task in observation.our.tasks:
        candidate = _candidate_for_task(task, world, pioneer)
        if candidate is not None:
            candidates.append(candidate)
    return min(candidates, key=lambda item: item.sort_key) if candidates else None


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
        learned = TaskSop(state.pending_task_type, state.pending_answer)
        sops = (
            learned,
            *(sop for sop in sops if sop.task_type != learned.task_type),
        )[:MAX_SOPS]
        return replace(
            _clear_task_lifecycle(state),
            sops=sops,
        )
    return replace(
        state,
        pending_task_type='',
        pending_answer='',
        pending_pioneer_id=None,
        pending_round=None,
    )


def _clear_task_lifecycle(state: TaskAgentState) -> TaskAgentState:
    return replace(
        state,
        active_task_type='',
        selected_task_position=None,
        accepted_round=None,
        timeout_rounds=None,
        deadline_round=None,
        last_prompt_fingerprint='',
        best_answer='',
        pending_task_type='',
        pending_answer='',
        pending_pioneer_id=None,
        pending_round=None,
    )


def _bounded(value: str, limit: int) -> str:
    return value.replace('\x00', '').strip()[:limit]


def _task_key(task_text: str) -> str:
    return _normalize_task_type('text:' + task_text)


def _normalize_task_type(task_type: str) -> str:
    normalized = task_type.replace('\x00', '').strip()
    if len(normalized) <= MAX_TASK_TYPE_LENGTH:
        return normalized
    digest = sha256(normalized.encode('utf-8')).hexdigest()[:16]
    prefix_length = MAX_TASK_TYPE_LENGTH - len(digest) - 1
    return normalized[:prefix_length] + ':' + digest


def _build_prompt(task_type: str, task_text: str) -> str:
    prompt = (
        'Solve the game task below. Return only the final answer, with no '
        'explanation, markdown, or shell commands.\n'
        f'Task type: {task_type}\n'
        f'Task: {task_text}'
    )
    return prompt[:MAX_PROMPT_LENGTH]
