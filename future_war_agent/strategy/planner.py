from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation
from future_war_agent.protocol.time import Phase

from .jobs import generate_day_jobs
from .joint import candidates_for_jobs, solve_joint
from .layout import build_defensive_layout
from .night import ControllerAssignment, generate_night_candidates
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent
from .rules import DEFAULT_RULES, RulesConfig
from .world import WorldGrid


def plan_turn(
    observation: Observation,
    rules: RulesConfig = DEFAULT_RULES,
    *,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
) -> Decision:
    world = WorldGrid.from_observation(observation, rules)
    if not world.friendly_roles:
        return Decision()
    if observation.time.phase is Phase.NIGHT:
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
