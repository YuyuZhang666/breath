from dataclasses import dataclass, replace

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, TaskPointState, UnitState

from .pathfinding import PathResult, path_to_interaction
from .policy import StrategicIntent
from .world import WorldGrid


MAX_ANSWER_LENGTH = 2048
MAX_PROMPT_LENGTH = 2048
MAX_NEWS_LENGTH = 512
MAX_SOPS = 16


@dataclass(frozen=True, slots=True)
class TaskSop:
    task_type: str
    answer: str

    def __post_init__(self) -> None:
        if not self.task_type.strip() or not self.answer.strip():
            raise ValueError('task SOP fields must be nonblank')


@dataclass(frozen=True, slots=True)
class TaskAgentState:
    active_task_type: str = ''
    accepted_round: int | None = None
    timeout_rounds: int | None = None
    pending_task_type: str = ''
    pending_answer: str = ''
    pending_pioneer_id: int | None = None
    sops: tuple[TaskSop, ...] = ()
    official_news: str = ''
    folk_legends: str = ''

    def __post_init__(self) -> None:
        if len(self.sops) > MAX_SOPS:
            raise ValueError('task SOP capacity exceeded')
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

        task_text = observation.phase_task.strip()
        if task_text:
            return self._handle_active_task(
                observation,
                base_decision,
                pioneer,
                state,
                task_text,
            )

        selected = _select_task(observation, pioneer)
        if selected is None:
            return TaskAgentResult(base_decision, state)
        state = replace(
            state,
            active_task_type=selected.task.task_type,
            accepted_round=observation.time.round_no,
            timeout_rounds=selected.task.timeout_rounds,
        )
        if pioneer.position.chebyshev_distance(selected.task.position) <= 1:
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
        sop_answer = next(
            (sop.answer for sop in state.sops if sop.task_type == task_type),
            '',
        )
        answer = sop_answer or _bounded(
            observation.llm_response,
            MAX_ANSWER_LENGTH,
        )
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
                pending_task_type=task_type,
                pending_answer=answer,
                pending_pioneer_id=pioneer.unit_id,
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
            replace(state, active_task_type=task_type),
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


def _select_task(
    observation: Observation,
    pioneer: UnitState,
) -> _TaskCandidate | None:
    world = WorldGrid.from_observation(observation)
    candidates: list[_TaskCandidate] = []
    for task in observation.our.tasks:
        if (
            not task.is_valid
            or task.cooldown_rounds != 0
            or task.score_reward + task.gold_reward <= 0
        ):
            continue
        path = path_to_interaction(world, pioneer.position, task.position)
        if path is None:
            continue
        timeout = task.timeout_rounds if task.timeout_rounds is not None else 1000
        urgency = max(0, 100 - min(timeout, 100))
        value = (
            task.score_reward * 10_000
            + task.gold_reward * 100
            + urgency
            - path.cost * 10
        )
        candidates.append(_TaskCandidate(task, path, value))
    return min(candidates, key=lambda item: item.sort_key) if candidates else None


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
            if retained.kind is ActionKind.MOVE and retained.target_positions
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
    if pioneer_id is None or pioneer_id not in observation.last_action_results:
        return state
    sops = state.sops
    if observation.last_action_results[pioneer_id]:
        learned = TaskSop(state.pending_task_type, state.pending_answer)
        sops = (
            learned,
            *(sop for sop in sops if sop.task_type != learned.task_type),
        )[:MAX_SOPS]
    return replace(
        state,
        active_task_type='',
        accepted_round=None,
        timeout_rounds=None,
        pending_task_type='',
        pending_answer='',
        pending_pioneer_id=None,
        sops=sops,
    )


def _bounded(value: str, limit: int) -> str:
    return value.replace('\x00', '').strip()[:limit]


def _task_key(task_text: str) -> str:
    return 'text:' + task_text.strip()[:128]


def _build_prompt(task_type: str, task_text: str) -> str:
    prompt = (
        'Solve the game task below. Return only the final answer, with no '
        'explanation, markdown, or shell commands.\n'
        f'Task type: {task_type}\n'
        f'Task: {task_text}'
    )
    return prompt[:MAX_PROMPT_LENGTH]
