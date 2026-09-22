from dataclasses import dataclass
from math import atan2

from future_war_agent.protocol.models import Position

from .policy import DEFAULT_BUILD_PLAN, BuildPlan
from .rules import station_footprint
from .world import WorldGrid


@dataclass(frozen=True, slots=True)
class WeaponSite:
    position: Position
    weapon_type: str


@dataclass(frozen=True, slots=True)
class DefensiveLayout:
    weapon_sites: tuple[WeaponSite, ...] = ()
    wall_sites: tuple[Position, ...] = ()
    critical_wall_sites: tuple[Position, ...] = ()
    controller_sites: tuple[Position, ...] = ()
    entrance: Position | None = None


def build_defensive_layout(
    world: WorldGrid,
    build_plan: BuildPlan = DEFAULT_BUILD_PLAN,
    *,
    recent_threat_positions: tuple[Position, ...] = (),
) -> DefensiveLayout:
    station = world.our_station()
    if station is None:
        return DefensiveLayout()

    footprint = station_footprint(station.position, world.rules)
    ring_one = _ring(world, footprint, 1)
    ring_two = _ring(world, footprint, 2)
    map_center = (
        (world.observation.width - 1) / 2,
        (world.observation.height - 1) / 2,
    )

    weapon_loadout = (
        build_plan.weapon_loadout if build_plan.build_weapons else ()
    )
    selected: list[Position] = []
    remaining = list(ring_one)
    if remaining and weapon_loadout:
        first = min(
            remaining,
            key=lambda value: (
                _center_distance(value, map_center),
                value.x,
                value.y,
            ),
        )
        selected.append(first)
        remaining.remove(first)
    while remaining and len(selected) < len(weapon_loadout):
        choice = min(
            remaining,
            key=lambda value: (
                -min(value.chebyshev_distance(other) for other in selected),
                _center_distance(value, map_center),
                value.x,
                value.y,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)

    weapon_sites = tuple(
        WeaponSite(position=position, weapon_type=weapon_type)
        for position, weapon_type in zip(selected, weapon_loadout)
    )

    entrance = (
        max(
            ring_two,
            key=lambda value: (
                _center_distance(value, map_center),
                -value.x,
                -value.y,
            ),
        )
        if ring_two
        else None
    )
    footprint_center = (
        sum(value.x for value in footprint) / len(footprint),
        sum(value.y for value in footprint) / len(footprint),
    )
    weapon_positions = frozenset(site.position for site in weapon_sites)
    current_threats = tuple(
        robot.position
        for robot in world.observation.robots
        if robot.health > 0
        and robot.target_team == world.observation.our.team_type
    )
    threat_positions = recent_threat_positions or current_threats
    if not threat_positions:
        threat_positions = (
            Position(round(map_center[0]), round(map_center[1])),
        )
    wall_sites = tuple(
        sorted(
            (
                value
                for value in ring_two
                if value != entrance and value not in weapon_positions
            ),
            key=lambda value: (
                -_threat_pressure(value, threat_positions, world),
                min(
                    value.chebyshev_distance(threat)
                    for threat in threat_positions
                ),
                -atan2(value.y - footprint_center[1], value.x - footprint_center[0]),
                value.x,
                value.y,
            ),
        )
    ) if build_plan.build_walls else ()
    if build_plan.wall_site_limit is not None:
        wall_sites = wall_sites[: build_plan.wall_site_limit]
    critical_wall_sites = wall_sites[: build_plan.opening_critical_wall_count]
    controller_sites, wall_sites = _reserve_controller_sites(
        world,
        weapon_sites,
        wall_sites,
    )
    critical_wall_sites = tuple(
        site for site in critical_wall_sites if site in wall_sites
    )
    return DefensiveLayout(
        weapon_sites=weapon_sites,
        wall_sites=wall_sites,
        critical_wall_sites=critical_wall_sites,
        controller_sites=controller_sites,
        entrance=entrance,
    )


def _threat_pressure(
    position: Position,
    threats: tuple[Position, ...],
    world: WorldGrid,
) -> int:
    scale = world.observation.width + world.observation.height
    return sum(
        max(0, scale - position.chebyshev_distance(threat))
        for threat in threats
    )


def _reserve_controller_sites(
    world: WorldGrid,
    weapon_sites: tuple[WeaponSite, ...],
    wall_sites: tuple[Position, ...],
) -> tuple[tuple[Position, ...], tuple[Position, ...]]:
    planned_walls = set(wall_sites)
    weapon_positions = {site.position for site in weapon_sites}
    reserved: list[Position] = []
    for site in weapon_sites:
        candidates = tuple(
            sorted(
                (
                    cell
                    for cell in world.interaction_cells(site.position)
                    if cell not in weapon_positions
                    and cell not in reserved
                ),
                key=lambda cell: (
                    cell in planned_walls,
                    cell.x,
                    cell.y,
                ),
            )
        )
        if not candidates:
            continue
        selected = candidates[0]
        reserved.append(selected)
        planned_walls.discard(selected)
    retained_walls = tuple(site for site in wall_sites if site in planned_walls)
    return tuple(reserved), retained_walls


def _ring(
    world: WorldGrid,
    footprint: frozenset[Position],
    distance: int,
) -> tuple[Position, ...]:
    candidates = (
        Position(x, y)
        for x in range(world.observation.width)
        for y in range(world.observation.height)
    )
    return tuple(
        value
        for value in candidates
        if world.is_geographic_land(value)
        and min(value.chebyshev_distance(cell) for cell in footprint) == distance
    )


def _center_distance(position: Position, center: tuple[float, float]) -> float:
    return max(abs(position.x - center[0]), abs(position.y - center[1]))
