import logging

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase
from future_war_agent.telemetry import DEFAULT_TELEMETRY, TelemetryRecorder

from .joint_fire import plan_joint_fire
from .jobs import generate_day_jobs
from .joint import candidates_for_jobs, solve_joint
from .layout import build_defensive_layout
from .night import ControllerAssignment, generate_night_candidates
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .rules import DEFAULT_RULES, RulesConfig
from .simulation.errors import UnsupportedSimulation
from .world import WorldGrid


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def plan_turn(
    observation: Observation,
    rules: RulesConfig = DEFAULT_RULES,
    *,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    telemetry: TelemetryRecorder = DEFAULT_TELEMETRY,
) -> Decision:
    world = WorldGrid.from_observation(observation, rules)
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
            )
            return plan.decision
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
        return solve_joint(
            observation,
            world,
            candidates,
            gold_reserve=intent.gold_reserve,
        )

    layout = build_defensive_layout(world, intent.build_plan)
    jobs = generate_day_jobs(observation, world, layout, intent)
    candidates = {
        role.unit_id: candidates_for_jobs(
            observation,
            world,
            role,
            jobs.get(role.unit_id, ()),
        )
        for role in world.friendly_roles
    }
    return solve_joint(
        observation,
        world,
        candidates,
        gold_reserve=intent.gold_reserve,
    )
