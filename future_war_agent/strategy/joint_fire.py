from dataclasses import dataclass
from itertools import combinations, product
from time import perf_counter_ns

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.time import Phase

from .joint import solve_joint
from .night import (
    ControllerAssignment,
    assign_controllers,
    generate_night_candidates,
)
from .policy import DEFAULT_STRATEGIC_INTENT, StrategicIntent, StrategyProfile
from .simulation.config import DEFAULT_PHASE3_CONFIG, Phase3Config
from .simulation.errors import UnsupportedSimulation
from .simulation.geometry import is_legal_cone
from .simulation.state import AssignedStand, SimRobot, SimState, SimWeapon, build_sim_state
from .simulation.weapons import WeaponAttack, weapon_damage
from .world import WorldGrid


@dataclass(frozen=True, slots=True)
class JointFireConfig:
    max_options_per_weapon: int = 4
    target_cell_slack: int = 5
    target_cell_cap: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.max_options_per_weapon <= 4:
            raise ValueError('max_options_per_weapon must be between one and four')
        if self.target_cell_slack < 0:
            raise ValueError('target_cell_slack cannot be negative')
        if self.target_cell_cap <= 0:
            raise ValueError('target_cell_cap must be positive')


DEFAULT_JOINT_FIRE_CONFIG = JointFireConfig()


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    robot_id: int
    station_distance: int
    turns_to_station_attack: int
    attack_power: int
    kill_score: int
    threatens_critical_asset: bool

    def value(self, profile: StrategyProfile) -> int:
        imminent = max(0, 8 - self.turns_to_station_attack)
        station_pressure = imminent * self.attack_power * 20
        critical_pressure = (
            self.attack_power * 160 if self.threatens_critical_asset else 0
        )
        attack_pressure = self.attack_power * 20
        score_value = self.kill_score * 25
        if profile in {StrategyProfile.SURVIVE, StrategyProfile.DESPERATION}:
            return (
                station_pressure * 3
                + critical_pressure * 2
                + attack_pressure
                + score_value
            )
        if profile is StrategyProfile.SCORE:
            return (
                station_pressure
                + critical_pressure
                + attack_pressure
                + score_value * 4
            )
        return (
            station_pressure * 2
            + critical_pressure
            + attack_pressure
            + score_value * 2
        )


@dataclass(frozen=True, slots=True)
class FireOption:
    attack: WeaponAttack
    weapon_type: str
    damage: tuple[tuple[int, int], ...]

    @property
    def stable_key(self) -> tuple[object, ...]:
        return self.attack.stable_key


@dataclass(frozen=True, slots=True)
class JointFirePlan:
    decision: Decision
    combinations_evaluated: int
    active_weapon_count: int
    attack_option_count: int
    candidate_generation_ms: float
    scoring_ms: float


def plan_joint_fire(
    observation: Observation,
    world: WorldGrid,
    intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    *,
    controller_assignments: tuple[ControllerAssignment, ...] | None = None,
    config: JointFireConfig = DEFAULT_JOINT_FIRE_CONFIG,
    simulation_config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> JointFirePlan:
    if observation.time.phase is not Phase.NIGHT:
        raise UnsupportedSimulation('Phase 2.5 requires a night observation')

    candidate_started_ns = perf_counter_ns()
    assignments = (
        assign_controllers(
            observation,
            world,
            mode_key=intent.profile.value,
        )
        if controller_assignments is None
        else controller_assignments
    )
    phase2_choices = generate_night_candidates(
        observation,
        world,
        intent,
        controller_assignments=assignments,
    )
    phase2_decision = solve_joint(
        observation,
        world,
        phase2_choices,
        gold_reserve=intent.gold_reserve,
    )
    base_commands = {
        actor_id: action
        for actor_id, action in phase2_decision.commands.items()
        if action.kind is not ActionKind.ATTACK
    }

    state = build_sim_state(
        observation,
        tuple(
            AssignedStand(item.role_id, item.weapon_id, item.stand)
            for item in assignments
        ),
        simulation_config,
    )
    threats = assess_threats(state)
    role_by_id = {role.unit_id: role for role in world.friendly_roles}
    weapon_by_id = {weapon.unit_id: weapon for weapon in state.weapons}
    option_sets: list[tuple[FireOption | None, ...]] = []
    option_count = 0

    for assignment in sorted(assignments, key=lambda item: item.weapon_id):
        role = role_by_id.get(assignment.role_id)
        weapon = weapon_by_id.get(assignment.weapon_id)
        if (
            role is None
            or weapon is None
            or role.position != assignment.stand
            or assignment.role_id in base_commands
        ):
            continue
        options = generate_fire_options(
            state,
            weapon,
            assignment.role_id,
            threats,
            intent.profile,
            config=config,
            simulation_config=simulation_config,
        )
        if not options:
            continue
        option_count += len(options)
        option_sets.append((None,) + options)

    candidate_generation_ms = (
        perf_counter_ns() - candidate_started_ns
    ) / 1_000_000
    if not option_sets:
        return JointFirePlan(
            decision=Decision(commands=base_commands),
            combinations_evaluated=0,
            active_weapon_count=0,
            attack_option_count=0,
            candidate_generation_ms=candidate_generation_ms,
            scoring_ms=0.0,
        )

    scoring_started_ns = perf_counter_ns()
    robot_by_id = {robot.robot_id: robot for robot in state.robots}
    winner: tuple[FireOption, ...] = ()
    winner_score: tuple[int, ...] | None = None
    winner_key: tuple[object, ...] | None = None
    combinations_evaluated = 0
    for possible in product(*option_sets):
        selected = tuple(item for item in possible if item is not None)
        combinations_evaluated += 1
        score = _joint_score(selected, robot_by_id, threats, intent.profile)
        stable_key = tuple(item.stable_key for item in selected)
        if (
            winner_score is None
            or score > winner_score
            or (score == winner_score and stable_key < winner_key)
        ):
            winner = selected
            winner_score = score
            winner_key = stable_key

    scoring_ms = (perf_counter_ns() - scoring_started_ns) / 1_000_000
    commands = dict(base_commands)
    for option in winner:
        commands[option.attack.weapon_id] = Action.attack(
            option.attack.controller_id,
            option.attack.targets,
        )
    return JointFirePlan(
        decision=Decision(commands=commands),
        combinations_evaluated=combinations_evaluated,
        active_weapon_count=len(option_sets),
        attack_option_count=option_count,
        candidate_generation_ms=candidate_generation_ms,
        scoring_ms=scoring_ms,
    )


def assess_threats(state: SimState) -> dict[int, ThreatAssessment]:
    critical_cells = set(state.station.occupied_cells)
    critical_cells.update(wall.position for wall in state.walls)
    critical_cells.update(weapon.position for weapon in state.weapons)
    critical_cells.update(role.position for role in state.roles)
    assessments: dict[int, ThreatAssessment] = {}
    for robot in state.robots:
        station_distance = min(
            robot.position.chebyshev_distance(cell)
            for cell in state.station.occupied_cells
        )
        turns_to_attack = max(0, station_distance - robot.attack_range)
        threatens_critical = any(
            robot.position.chebyshev_distance(cell) <= robot.attack_range + 1
            for cell in critical_cells
        )
        assessments[robot.robot_id] = ThreatAssessment(
            robot_id=robot.robot_id,
            station_distance=station_distance,
            turns_to_station_attack=turns_to_attack,
            attack_power=robot.attack_power,
            kill_score=robot.kill_score,
            threatens_critical_asset=threatens_critical,
        )
    return assessments


def generate_fire_options(
    state: SimState,
    weapon: SimWeapon,
    controller_id: int,
    threats: dict[int, ThreatAssessment],
    profile: StrategyProfile,
    *,
    config: JointFireConfig = DEFAULT_JOINT_FIRE_CONFIG,
    simulation_config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> tuple[FireOption, ...]:
    if weapon.cooldown != 0 or weapon.attack_range <= 0 or weapon.level <= 0:
        return ()
    living = tuple(robot for robot in state.robots if robot.health > 0)
    if not living:
        return ()

    groups = _target_groups(state, weapon, living, threats, profile, config)
    unique: dict[tuple[tuple[int, int], ...], FireOption] = {}
    for targets in groups:
        attack = WeaponAttack(weapon.unit_id, controller_id, targets)
        damage = tuple(sorted(weapon_damage(state, attack, simulation_config).items()))
        if not damage:
            continue
        option = FireOption(attack, weapon.role_type, damage)
        previous = unique.get(damage)
        if previous is None or option.stable_key < previous.stable_key:
            unique[damage] = option

    robot_by_id = {robot.robot_id: robot for robot in living}
    ranked = sorted(
        unique.values(),
        key=lambda option: (
            tuple(
                -value
                for value in _joint_score(
                    (option,),
                    robot_by_id,
                    threats,
                    profile,
                )
            ),
            option.stable_key,
        ),
    )
    return tuple(ranked[: config.max_options_per_weapon])


def _target_groups(
    state: SimState,
    weapon: SimWeapon,
    robots: tuple[SimRobot, ...],
    threats: dict[int, ThreatAssessment],
    profile: StrategyProfile,
    config: JointFireConfig,
) -> tuple[tuple[Position, ...], ...]:
    if weapon.role_type == 'railgun':
        cells = _rank_robot_cells(weapon, robots, threats, profile)
        return tuple((cell,) for cell in cells)

    if weapon.role_type == 'rocket':
        cells = _rank_rocket_cells(state, weapon, robots, threats, profile)
    elif weapon.role_type == 'gatling':
        cells = _rank_robot_cells(weapon, robots, threats, profile)
    else:
        return ()

    limit = max(
        weapon.level,
        min(config.target_cell_cap, weapon.level + config.target_cell_slack),
    )
    cells = cells[:limit]
    if len(cells) < weapon.level:
        return ()
    groups = tuple(combinations(cells, weapon.level))
    if weapon.role_type == 'gatling':
        groups = tuple(
            group
            for group in groups
            if is_legal_cone(weapon.position, group)
        )
    return groups


def _rank_robot_cells(
    weapon: SimWeapon,
    robots: tuple[SimRobot, ...],
    threats: dict[int, ThreatAssessment],
    profile: StrategyProfile,
) -> tuple[Position, ...]:
    robots_by_cell: dict[Position, list[SimRobot]] = {}
    for robot in robots:
        if weapon.position.chebyshev_distance(robot.position) <= weapon.attack_range:
            robots_by_cell.setdefault(robot.position, []).append(robot)
    return tuple(
        sorted(
            robots_by_cell,
            key=lambda cell: (
                -max(threats[item.robot_id].value(profile) for item in robots_by_cell[cell]),
                min(item.health for item in robots_by_cell[cell]),
                cell.x,
                cell.y,
            ),
        )
    )


def _rank_rocket_cells(
    state: SimState,
    weapon: SimWeapon,
    robots: tuple[SimRobot, ...],
    threats: dict[int, ThreatAssessment],
    profile: StrategyProfile,
) -> tuple[Position, ...]:
    cells = {
        Position(robot.position.x + delta_x, robot.position.y + delta_y)
        for robot in robots
        for delta_x in (-1, 0, 1)
        for delta_y in (-1, 0, 1)
    }
    legal = tuple(
        cell
        for cell in cells
        if 0 <= cell.x < state.width
        and 0 <= cell.y < state.height
        and weapon.position.chebyshev_distance(cell) <= weapon.attack_range
    )

    def key(cell: Position) -> tuple[object, ...]:
        affected = tuple(
            robot
            for robot in robots
            if cell.chebyshev_distance(robot.position) <= 1
        )
        center = tuple(robot for robot in affected if robot.position == cell)
        return (
            -sum(threats[robot.robot_id].value(profile) for robot in affected),
            -sum(threats[robot.robot_id].value(profile) for robot in center),
            -len(affected),
            cell.x,
            cell.y,
        )

    return tuple(sorted(legal, key=key))


def _joint_score(
    options: tuple[FireOption, ...],
    robot_by_id: dict[int, SimRobot],
    threats: dict[int, ThreatAssessment],
    profile: StrategyProfile,
) -> tuple[int, ...]:
    total_damage: dict[int, int] = {}
    isolated_effective_damage = 0
    for option in options:
        for robot_id, damage in option.damage:
            robot = robot_by_id.get(robot_id)
            if robot is None:
                continue
            isolated_effective_damage += min(damage, robot.health)
            total_damage[robot_id] = total_damage.get(robot_id, 0) + damage

    effective_damage = 0
    overkill = 0
    threat_hit = 0
    weighted_damage = 0
    removed_threat = 0
    imminent_damage_removed = 0
    kill_score = 0
    killed_count = 0
    for robot_id, damage in total_damage.items():
        robot = robot_by_id.get(robot_id)
        threat = threats.get(robot_id)
        if robot is None or threat is None:
            continue
        effective = min(damage, robot.health)
        effective_damage += effective
        overkill += max(0, damage - robot.health)
        value = threat.value(profile)
        threat_hit += value
        weighted_damage += value * effective // max(1, robot.health)
        if damage >= robot.health:
            killed_count += 1
            kill_score += robot.kill_score
            removed_threat += value
            if threat.turns_to_station_attack <= 1:
                imminent_damage_removed += robot.attack_power

    duplicate_damage = max(0, isolated_effective_damage - effective_damage)
    rocket_cost = sum(
        1 for option in options if option.weapon_type == 'rocket'
    )
    fired_count = len(options)
    if profile is StrategyProfile.SCORE:
        return (
            kill_score,
            killed_count,
            imminent_damage_removed,
            removed_threat,
            threat_hit,
            weighted_damage,
            effective_damage,
            -duplicate_damage,
            -overkill,
            -rocket_cost,
            -fired_count,
        )
    return (
        imminent_damage_removed,
        threat_hit,
        removed_threat,
        weighted_damage,
        killed_count,
        kill_score,
        effective_damage,
        -duplicate_damage,
        -overkill,
        -rocket_cost,
        -fired_count,
    )
