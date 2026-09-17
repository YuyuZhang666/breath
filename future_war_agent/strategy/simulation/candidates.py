from dataclasses import dataclass

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.strategy.joint import (
    AttackValidator,
    Joint,
    TacticalCandidate,
    decision_for_joint,
    enumerate_legal_joints,
)
from future_war_agent.strategy.night import (
    assign_controllers,
    generate_night_candidates,
)
from future_war_agent.strategy.world import WorldGrid

from .config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .geometry import is_legal_cone
from .state import SimState
from .weapons import WeaponAttack, generate_weapon_attacks


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
    assignments = assign_controllers(observation, world)
    assignment_by_role = {
        assignment.role_id: assignment for assignment in assignments
    }
    sim_weapon_by_id = {weapon.unit_id: weapon for weapon in state.weapons}
    phase_2_choices = generate_night_candidates(observation, world)
    choices: dict[int, tuple[TacticalCandidate, ...]] = {}

    for role in sorted(world.friendly_roles, key=lambda item: item.unit_id):
        assignment = assignment_by_role.get(role.unit_id)
        if assignment is None or role.position != assignment.stand:
            choices[role.unit_id] = phase_2_choices[role.unit_id]
            continue

        sim_weapon = sim_weapon_by_id.get(assignment.weapon_id)
        if sim_weapon is None:
            choices[role.unit_id] = (
                TacticalCandidate.wait(role.unit_id, role.position),
            )
            continue

        attacks = generate_weapon_attacks(
            state,
            sim_weapon,
            role.unit_id,
            config,
        )
        choices[role.unit_id] = tuple(
            TacticalCandidate.attack(
                role_id=role.unit_id,
                start=role.position,
                weapon_id=attack.weapon_id,
                targets=attack.targets,
                priority=500,
            )
            for attack in attacks
        ) + (TacticalCandidate.wait(role.unit_id, role.position),)

    validator = _phase_3_attack_validator(state)
    joints = enumerate_legal_joints(
        observation,
        world,
        choices,
        limit=config.max_root_actions,
        attack_validator=validator,
    )
    return tuple(_root_for_joint(joint) for joint in joints)


def _phase_3_attack_validator(
    state: SimState,
) -> AttackValidator:
    sim_weapon_by_id = {weapon.unit_id: weapon for weapon in state.weapons}

    def validate(role: UnitState, weapon: UnitState, action: Action) -> bool:
        simulated = sim_weapon_by_id.get(weapon.unit_id)
        if simulated is None:
            return False
        targets = action.target_positions
        if (
            role.position.chebyshev_distance(weapon.position) != 1
            or action.controller_id != role.unit_id
            or simulated.cooldown != 0
            or len(set(targets)) != len(targets)
        ):
            return False
        if any(
            not (0 <= target.x < state.width and 0 <= target.y < state.height)
            or simulated.position.chebyshev_distance(target)
            > simulated.attack_range
            for target in targets
        ):
            return False
        if simulated.role_type == "railgun":
            return len(targets) == 1
        if simulated.role_type == "gatling":
            return len(targets) == simulated.level and is_legal_cone(
                simulated.position,
                targets,
            )
        if simulated.role_type == "rocket":
            return len(targets) == simulated.level
        return False

    return validate


def _root_for_joint(joint: Joint) -> RootAction:
    decision = decision_for_joint(joint)
    role_moves = tuple(
        sorted(
            (
                RoleMove(item.role_id, item.action.target_positions[0])
                for item in joint
                if item.action is not None
                and item.action.kind is ActionKind.MOVE
                and len(item.action.target_positions) == 1
            ),
            key=lambda move: move.role_id,
        )
    )
    weapon_attacks = tuple(
        sorted(
            (
                WeaponAttack(
                    weapon_id=item.command_actor_id,
                    controller_id=item.role_id,
                    targets=item.action.target_positions,
                )
                for item in joint
                if item.action is not None
                and item.action.kind is ActionKind.ATTACK
            ),
            key=lambda attack: attack.stable_key,
        )
    )
    simulation_action = SimJointAction(
        role_moves=role_moves,
        weapon_attacks=weapon_attacks,
    )
    return RootAction(
        decision=decision,
        simulation_action=simulation_action,
        stable_key=_root_key(decision, simulation_action),
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
