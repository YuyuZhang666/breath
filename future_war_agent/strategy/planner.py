import logging

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.time import Phase
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .joint_fire import plan_joint_fire
from .build_recovery import EMPTY_BUILD_RECOVERY_STATE, BuildRecoveryState
from .collapse_summons import generate_collapse_summon_jobs
from .jobs import JobKind, generate_day_jobs
from .joint import candidates_for_jobs, solve_joint
from .layout import build_defensive_layout
from .market import MarketView
from .night import ControllerAssignment, generate_night_candidates
from .opponent_memory import OpponentMemory
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .rules import DEFAULT_RULES, RulesConfig
from .simulation.errors import UnsupportedSimulation
from .strategic_items import (
    apply_emergency_combat_items,
    generate_strategic_item_jobs,
    merge_strategic_item_jobs,
)
from .world import WorldGrid


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def plan_turn(
    observation: Observation,
    rules: RulesConfig = DEFAULT_RULES,
    *,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    world: WorldGrid | None = None,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
    fortification_threats: tuple[Position, ...] = (),
    expected_wall_losses: int = 0,
    market_view: MarketView | None = None,
    previous_decision: Decision | None = None,
    previously_built_wall_sites: frozenset[Position] = frozenset(),
    build_recovery: BuildRecoveryState = EMPTY_BUILD_RECOVERY_STATE,
    opponent_memory: OpponentMemory | None = None,
) -> Decision:
    world = WorldGrid.from_observation(observation, rules) if world is None else world
    if not world.friendly_roles:
        return Decision()
    if observation.time.phase is Phase.NIGHT:
        try:
            with telemetry.measure('phase2_5_ms'):
                plan = plan_joint_fire(
                    observation,
                    world,
                    intent,
                    controller_assignments=controller_assignments,
                )
            telemetry.set(
                phase2_5_combination_count=plan.combinations_evaluated,
                phase2_5_active_weapon_count=plan.active_weapon_count,
                candidate_generation_ms=plan.candidate_generation_ms,
                phase2_5_weapon_log=plan.weapon_log,
            )
            return apply_emergency_combat_items(
                observation,
                plan.decision,
                intent,
            )
        except UnsupportedSimulation:
            telemetry.increment('phase2_5_fallback_count')
            telemetry.set(fallback_used=True)
            LOGGER.debug('Phase 2.5 unavailable; using Phase 2', exc_info=True)
        except Exception:
            telemetry.increment('phase2_5_fallback_count')
            telemetry.set(fallback_used=True)
            LOGGER.exception('Phase 2.5 failed; using Phase 2')
        candidates = generate_night_candidates(
            observation,
            world,
            intent,
            controller_assignments=controller_assignments,
        )
        baseline = solve_joint(
            observation,
            world,
            candidates,
            gold_reserve=intent.gold_reserve,
        )
        return apply_emergency_combat_items(observation, baseline, intent)

    layout = build_defensive_layout(
        world,
        intent.build_plan,
        recent_threat_positions=fortification_threats,
    )
    jobs = generate_day_jobs(
        observation,
        world,
        layout,
        intent,
        expected_wall_losses=expected_wall_losses,
        market_view=market_view,
        previous_decision=previous_decision,
        previously_built_wall_sites=previously_built_wall_sites,
        build_recovery=build_recovery,
        telemetry=telemetry,
    )
    jobs = merge_strategic_item_jobs(
        jobs,
        generate_strategic_item_jobs(
            observation,
            world,
            layout,
            intent,
            expected_wall_losses=expected_wall_losses,
        ),
    )
    jobs = merge_strategic_item_jobs(
        jobs,
        generate_collapse_summon_jobs(
            observation,
            world,
            layout,
            intent,
            opponent_memory,
        ),
    )
    candidates = {
        role.unit_id: candidates_for_jobs(
            observation,
            world,
            role,
            jobs.get(role.unit_id, ()),
        )
        for role in world.friendly_roles
    }
    decision = solve_joint(
        observation,
        world,
        candidates,
        gold_reserve=intent.gold_reserve,
    )
    _record_day_joint_overrides(
        observation,
        jobs,
        decision,
        telemetry,
    )
    return decision


def _record_day_joint_overrides(
    observation: Observation,
    jobs,
    decision: Decision,
    telemetry: TelemetryRecorder,
) -> None:
    roles = {
        role.unit_id: role
        for role in observation.our.units
        if role.health > 0
    }
    entries = list(telemetry.current('action_override_log', ()))
    for role_id, role_jobs in sorted(jobs.items()):
        if not role_jobs:
            continue
        intended = role_jobs[0]
        role = roles.get(role_id)
        if (
            role is None
            or intended.kind is not JobKind.BUILD_WALL
            or role.position.chebyshev_distance(intended.target) != 1
        ):
            continue
        original = Action.build('wall', intended.target)
        final = decision.commands.get(role_id)
        if final == original:
            continue
        entries.append(
            f'ACTION_OVERRIDE:id={role_id}:'
            f'original=build[name=wall;targets='
            f'{intended.target.x},{intended.target.y}]:'
            f'final={_action_summary(final)}:'
            f'module=joint_planner:'
            f'reason=joint_legality_or_global_optimization'
        )
        if len(entries) >= 32:
            break
    telemetry.set(action_override_log=tuple(entries))


def _action_summary(action: Action | None) -> str:
    if action is None:
        return 'none'
    details: list[str] = []
    if action.name is not None:
        details.append(f'name={action.name}')
    if action.target_positions:
        details.append(
            'targets='
            + '|'.join(
                f'{target.x},{target.y}'
                for target in action.target_positions
            )
        )
    joined = ';'.join(details)
    return (
        action.kind.value
        if not details
        else f'{action.kind.value}[{joined}]'
    )
