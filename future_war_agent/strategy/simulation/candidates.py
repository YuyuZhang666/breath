from dataclasses import dataclass

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.strategy.joint import solve_joint
from future_war_agent.strategy.joint_fire import choose_joint_fire_attacks
from future_war_agent.strategy.night import (
    ControllerAssignment,
    assign_controllers,
    generate_night_candidates,
)
from future_war_agent.strategy.policy import (
    DEFAULT_STRATEGIC_INTENT,
    StrategicIntent,
    StrategyProfile,
)
from future_war_agent.strategy.world import WorldGrid

from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .errors import UnsupportedSimulation
from .state import SimState
from .weapons import WeaponAttack


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
    kind: str = 'joint_fire'


def generate_root_actions(
    observation: Observation,
    world: WorldGrid,
    state: SimState,
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
    *,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    baseline_decision: Decision | None = None,
    max_roots: int | None = None,
) -> tuple[RootAction, ...]:
    limit = config.max_root_actions if max_roots is None else max_roots
    if limit <= 0:
        raise ValueError('max_roots must be positive')
    assignments = (
        assign_controllers(
            observation,
            world,
            mode_key=intent.profile.value,
        )
        if controller_assignments is None
        else controller_assignments
    )
    baseline = baseline_decision
    if baseline is None:
        choices = generate_night_candidates(
            observation,
            world,
            intent,
            controller_assignments=assignments,
        )
        baseline = solve_joint(
            observation,
            world,
            choices,
            gold_reserve=intent.gold_reserve,
        )
    unsupported = tuple(
        action.kind
        for action in baseline.commands.values()
        if action.kind not in {ActionKind.MOVE, ActionKind.ATTACK}
    )
    if unsupported:
        raise UnsupportedSimulation(
            f'Phase 3 cannot simulate root actions: {unsupported!r}'
        )

    roots: list[RootAction] = []
    seen: set[tuple[object, ...]] = set()

    def add(decision: Decision, kind: str) -> None:
        root = _root_for_decision(decision, kind)
        if root.stable_key in seen or len(roots) >= limit:
            return
        seen.add(root.stable_key)
        roots.append(root)

    add(baseline, 'phase2_5')
    base_commands = {
        actor_id: action
        for actor_id, action in baseline.commands.items()
        if action.kind is ActionKind.MOVE
    }
    profiles = [StrategyProfile.SURVIVE, StrategyProfile.SCORE]
    if limit > 4:
        profiles.append(StrategyProfile.DESPERATION)
    for profile in profiles:
        selection = choose_joint_fire_attacks(
            state,
            profile,
            excluded_controller_ids=frozenset(base_commands),
            simulation_config=config,
        )
        commands = dict(base_commands)
        for attack in selection.attacks:
            commands[attack.weapon_id] = Action.attack(
                attack.controller_id,
                attack.targets,
            )
        add(Decision(commands=commands), f'{profile.value}_fire')

    if any(
        weapon.health > 0
        and weapon.role_type == 'rocket'
        and weapon.cooldown == 0
        for weapon in state.weapons
    ):
        rocket_hold = choose_joint_fire_attacks(
            state,
            intent.profile,
            excluded_controller_ids=frozenset(base_commands),
            hold_rockets=True,
            simulation_config=config,
        )
        hold_commands = dict(base_commands)
        for attack in rocket_hold.attacks:
            hold_commands[attack.weapon_id] = Action.attack(
                attack.controller_id,
                attack.targets,
            )
        add(Decision(commands=hold_commands), 'hold_rocket')
    add(Decision(commands=base_commands), 'hold_fire')
    return tuple(roots[:limit])


def _root_for_decision(decision: Decision, kind: str) -> RootAction:
    role_moves = tuple(
        sorted(
            (
                RoleMove(actor_id, action.target_positions[0])
                for actor_id, action in decision.commands.items()
                if action.kind is ActionKind.MOVE
                and len(action.target_positions) == 1
            ),
            key=lambda move: move.role_id,
        )
    )
    weapon_attacks = tuple(
        sorted(
            (
                WeaponAttack(
                    weapon_id=actor_id,
                    controller_id=action.controller_id,
                    targets=action.target_positions,
                )
                for actor_id, action in decision.commands.items()
                if action.kind is ActionKind.ATTACK
            ),
            key=lambda attack: attack.stable_key,
        )
    )
    simulation_action = SimJointAction(role_moves, weapon_attacks)
    return RootAction(
        decision=decision,
        simulation_action=simulation_action,
        stable_key=_root_key(decision, simulation_action),
        kind=kind,
    )


def _root_key(
    decision: Decision,
    simulation_action: SimJointAction,
) -> tuple[object, ...]:
    return (
        tuple(
            (move.role_id, move.target.x, move.target.y)
            for move in simulation_action.role_moves
        ),
        tuple(attack.stable_key for attack in simulation_action.weapon_attacks),
        tuple(
            (actor_id, _action_key(action))
            for actor_id, action in sorted(decision.commands.items())
        ),
    )


def _action_key(action: Action) -> tuple[object, ...]:
    return (
        action.kind.value,
        action.controller_id,
        tuple((target.x, target.y) for target in action.target_positions),
        action.name,
        action.quantity,
    )
