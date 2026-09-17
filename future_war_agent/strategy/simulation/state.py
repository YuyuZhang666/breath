from dataclasses import dataclass

from future_war_agent.protocol.models import Observation, Position, UnitState
from future_war_agent.protocol.time import NIGHT_ROUNDS, Phase
from future_war_agent.strategy.rules import station_footprint

from .config import DEFAULT_PHASE3_CONFIG, ROBOT_SPECS, Phase3Config
from .errors import UnsupportedSimulation


_ROLE_TYPES = frozenset({"worker", "pioneer"})
_WEAPON_TYPES = frozenset({"gatling", "railgun", "rocket"})
_STRUCTURE_TYPES = frozenset({"station", "wall"})
_REQUIRED_WEAPON_FIELDS = frozenset(
    {"attackPower", "attackRange", "level", "cooldown"}
)


@dataclass(frozen=True, slots=True)
class AssignedStand:
    role_id: int
    weapon_id: int
    stand: Position


@dataclass(frozen=True, slots=True)
class SimRole:
    unit_id: int
    role_type: str
    position: Position
    health: int
    assigned_weapon_id: int | None
    assigned_stand: Position | None


@dataclass(frozen=True, slots=True)
class SimStructure:
    unit_id: int
    role_type: str
    position: Position
    occupied_cells: frozenset[Position]
    health: int
    level: int | None


@dataclass(frozen=True, slots=True)
class SimWeapon:
    unit_id: int
    role_type: str
    position: Position
    health: int
    attack_power: int
    attack_range: int
    level: int
    cooldown: int


@dataclass(frozen=True, slots=True)
class SimRobot:
    robot_id: int
    role_type: str
    position: Position
    health: int
    attack_power: int
    attack_range: int
    kill_score: int
    waits_this_turn: bool


@dataclass(frozen=True, slots=True)
class SimState:
    round_no: int
    width: int
    height: int
    remaining_night_turns: int
    team_type: str
    static_blocked: frozenset[Position]
    roles: tuple[SimRole, ...]
    station: SimStructure
    walls: tuple[SimStructure, ...]
    weapons: tuple[SimWeapon, ...]
    robots: tuple[SimRobot, ...]
    owned_kill_score: int = 0


def build_sim_state(
    observation: Observation,
    assignments: tuple[AssignedStand, ...],
    config: Phase3Config = DEFAULT_PHASE3_CONFIG,
) -> SimState:
    del config
    if observation.time.phase is not Phase.NIGHT:
        raise UnsupportedSimulation("Phase 3 requires a night observation")

    team_type = observation.our.team_type.strip()
    if not team_type:
        raise UnsupportedSimulation("our team type is required")

    living_stations = tuple(
        unit
        for unit in observation.our.units
        if unit.health > 0 and unit.role_type == "station"
    )
    if len(living_stations) != 1:
        raise UnsupportedSimulation("exactly one living station is required")

    _validate_observed_coordinates(observation)

    alive_robots = tuple(robot for robot in observation.robots if robot.health > 0)
    for robot in alive_robots:
        if robot.target_team is None or not robot.target_team.strip():
            raise UnsupportedSimulation("every living robot requires targetTeam")

    retained_robots = tuple(
        robot for robot in alive_robots if robot.target_team == team_type
    )
    sim_robots: list[SimRobot] = []
    for robot in sorted(retained_robots, key=lambda value: value.robot_id):
        spec = ROBOT_SPECS.get(robot.role_type)
        if spec is None:
            raise UnsupportedSimulation(f"unknown robot type: {robot.role_type}")
        sim_robots.append(
            SimRobot(
                robot_id=robot.robot_id,
                role_type=robot.role_type,
                position=robot.position,
                health=robot.health,
                attack_power=spec.attack_power,
                attack_range=spec.attack_range,
                kill_score=spec.kill_score,
                waits_this_turn=robot.abnormal_state.casefold() == "dizzy",
            )
        )

    living_own_units = tuple(
        unit for unit in observation.our.units if unit.health > 0
    )
    weapon_units: list[UnitState] = []
    for unit in living_own_units:
        if unit.role_type in _ROLE_TYPES or unit.role_type in _STRUCTURE_TYPES:
            continue
        if unit.role_type not in _WEAPON_TYPES:
            raise UnsupportedSimulation(f"unknown weapon type: {unit.role_type}")
        _validate_weapon(unit)
        weapon_units.append(unit)

    assignment_by_role: dict[int, AssignedStand] = {}
    for assignment in assignments:
        if assignment.role_id in assignment_by_role:
            raise UnsupportedSimulation("duplicate controller assignment")
        if not _in_bounds(assignment.stand, observation.width, observation.height):
            raise UnsupportedSimulation("controller stand is outside the map")
        assignment_by_role[assignment.role_id] = AssignedStand(
            role_id=assignment.role_id,
            weapon_id=assignment.weapon_id,
            stand=assignment.stand,
        )

    roles = tuple(
        SimRole(
            unit_id=unit.unit_id,
            role_type=unit.role_type,
            position=unit.position,
            health=unit.health,
            assigned_weapon_id=(
                assignment_by_role[unit.unit_id].weapon_id
                if unit.unit_id in assignment_by_role
                else None
            ),
            assigned_stand=(
                assignment_by_role[unit.unit_id].stand
                if unit.unit_id in assignment_by_role
                else None
            ),
        )
        for unit in sorted(living_own_units, key=lambda value: value.unit_id)
        if unit.role_type in _ROLE_TYPES
    )

    station_unit = living_stations[0]
    station_cells = station_footprint(station_unit.position)
    _require_cells_in_bounds(
        station_cells,
        observation.width,
        observation.height,
        "station footprint",
    )
    station = _structure(station_unit, station_cells)
    walls = tuple(
        _structure(unit, frozenset({unit.position}))
        for unit in sorted(living_own_units, key=lambda value: value.unit_id)
        if unit.role_type == "wall"
    )
    weapons = tuple(
        SimWeapon(
            unit_id=unit.unit_id,
            role_type=unit.role_type,
            position=unit.position,
            health=unit.health,
            attack_power=unit.attack_power,
            attack_range=unit.attack_range,
            level=_weapon_level(unit),
            cooldown=unit.cooldown,
        )
        for unit in sorted(weapon_units, key=lambda value: value.unit_id)
    )

    static_blocked = set(zone.position for zone in observation.zones)
    for unit in observation.our.units + observation.enemy.units:
        if unit.health <= 0 or unit.role_type in _ROLE_TYPES:
            continue
        occupied = (
            station_footprint(unit.position)
            if unit.role_type == "station"
            else frozenset({unit.position})
        )
        _require_cells_in_bounds(
            occupied,
            observation.width,
            observation.height,
            "visible building footprint",
        )
        static_blocked.update(occupied)

    return SimState(
        round_no=observation.time.round_no,
        width=observation.width,
        height=observation.height,
        remaining_night_turns=(
            NIGHT_ROUNDS - observation.time.round_in_phase + 1
        ),
        team_type=team_type,
        static_blocked=frozenset(static_blocked),
        roles=roles,
        station=station,
        walls=walls,
        weapons=weapons,
        robots=tuple(sim_robots),
    )


def _validate_observed_coordinates(observation: Observation) -> None:
    positions = [zone.position for zone in observation.zones]
    positions.extend(unit.position for unit in observation.our.units)
    positions.extend(unit.position for unit in observation.enemy.units)
    positions.extend(robot.position for robot in observation.robots)
    positions.extend(task.position for task in observation.our.tasks)
    for position in positions:
        if not _in_bounds(position, observation.width, observation.height):
            raise UnsupportedSimulation("observed coordinate is outside the map")


def _validate_weapon(unit: UnitState) -> None:
    if not _REQUIRED_WEAPON_FIELDS <= unit.provided_fields:
        raise UnsupportedSimulation("weapon fields must be explicit")
    if unit.attack_range <= 0:
        raise UnsupportedSimulation("weapon range must be positive")
    if unit.level is None or unit.level <= 0:
        raise UnsupportedSimulation("weapon level must be positive")
    if unit.cooldown < 0:
        raise UnsupportedSimulation("weapon cooldown cannot be negative")
    if unit.role_type == "gatling" and unit.attack_power != 10:
        raise UnsupportedSimulation("gatling attack power must be 10")
    if unit.role_type == "rocket" and unit.attack_power != 20:
        raise UnsupportedSimulation("rocket attack power must be 20")
    if unit.role_type == "railgun" and unit.attack_power <= 0:
        raise UnsupportedSimulation("railgun energy must be positive")


def _weapon_level(unit: UnitState) -> int:
    if unit.level is None:
        raise UnsupportedSimulation("weapon level must be explicit")
    return unit.level


def _structure(unit: UnitState, cells: frozenset[Position]) -> SimStructure:
    return SimStructure(
        unit_id=unit.unit_id,
        role_type=unit.role_type,
        position=unit.position,
        occupied_cells=cells,
        health=unit.health,
        level=unit.level,
    )


def _in_bounds(position: Position, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _require_cells_in_bounds(
    cells: frozenset[Position],
    width: int,
    height: int,
    label: str,
) -> None:
    if any(not _in_bounds(cell, width, height) for cell in cells):
        raise UnsupportedSimulation(f"{label} extends outside the map")
