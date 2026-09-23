from dataclasses import dataclass
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
    ring_one = _ring(world, 1)
    ring_two = _ring(world, 2)
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

    footprint_center = (
        sum(value.x for value in footprint) / len(footprint),
        sum(value.y for value in footprint) / len(footprint),
    )
    rear_x = (
        min((value.x for value in ring_two), default=None)
        if footprint_center[0] <= map_center[0]
        else max((value.x for value in ring_two), default=None)
    )
    rear_candidates = tuple(
        value for value in ring_two if value.x == rear_x
    )
    entrance = (
        max(
            rear_candidates,
            key=lambda value: (
                _center_distance(value, map_center),
                -value.y,
            ),
        )
        if rear_candidates
        else None
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
    ordinary_wall_sites = (
        _directional_wall_order(
            tuple(
                value
                for value in ring_two
                if value != entrance and value not in weapon_positions
            ),
            threat_positions,
            footprint_center,
            world,
        )
        if build_plan.build_walls
        else ()
    )
    if build_plan.wall_site_limit is not None:
        ordinary_wall_sites = ordinary_wall_sites[:build_plan.wall_site_limit]
    # The map-edge face is a permanent access lane.  Keeping it out of the
    # plan prevents late-day infill from undoing the three-sided layout.
    wall_sites = ordinary_wall_sites
    critical_wall_sites = ordinary_wall_sites[
        : build_plan.opening_critical_wall_count
    ]
    controller_sites = _reserve_controller_sites(
        world,
        weapon_sites,
        wall_sites + ((entrance,) if entrance is not None else ()),
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


def _directional_wall_order(
    candidates: tuple[Position, ...],
    threats: tuple[Position, ...],
    footprint_center: tuple[float, float],
    world: WorldGrid,
) -> tuple[Position, ...]:
    # Build a stable U-shaped barrier facing the map centre.  Transient robot
    # positions may reorder cells within those three faces, but must never
    # rotate the structure and accidentally spend stone on the rear face.
    if not candidates:
        return ()
    threat_center = (
        sum(position.x for position in threats) / len(threats),
        sum(position.y for position in threats) / len(threats),
    )
    map_center_x = (world.observation.width - 1) / 2
    forward = (1 if map_center_x >= footprint_center[0] else -1, 0)
    lateral = (0, 1)

    def projections(position: Position) -> tuple[float, float]:
        relative_x = position.x - footprint_center[0]
        relative_y = position.y - footprint_center[1]
        return (
            relative_x * forward[0] + relative_y * forward[1],
            relative_x * lateral[0] + relative_y * lateral[1],
        )

    projected = {position: projections(position) for position in candidates}
    front_level = max(value[0] for value in projected.values())
    rear_level = min(value[0] for value in projected.values())
    negative_edge = min(value[1] for value in projected.values())
    positive_edge = max(value[1] for value in projected.values())
    threat_lateral = (
        threat_center[0] - footprint_center[0]
    ) * lateral[0] + (
        threat_center[1] - footprint_center[1]
    ) * lateral[1]
    front = tuple(
        sorted(
            (
                position
                for position in candidates
                if projected[position][0] == front_level
            ),
            key=lambda position: (
                abs(projected[position][1] - threat_lateral),
                -_threat_pressure(position, threats, world),
                projected[position][1],
                position.x,
                position.y,
            ),
        )[:6]
    )
    selected = set(front)
    negative_flank = _nearest_flank(
        candidates,
        projected,
        selected,
        negative_edge,
        front_level,
        rear_level,
    )
    selected.update(negative_flank)
    positive_flank = _nearest_flank(
        candidates,
        projected,
        selected,
        positive_edge,
        front_level,
        rear_level,
    )
    selected.update(positive_flank)

    directional = (
        tuple(
            sorted(
                negative_flank,
                key=lambda position: (
                    projected[position][0],
                    position.x,
                    position.y,
                ),
            )
        )
        + tuple(
            sorted(
                front,
                key=lambda position: (
                    projected[position][1],
                    position.x,
                    position.y,
                ),
            )
        )
        + tuple(
            sorted(
                positive_flank,
                key=lambda position: (
                    -projected[position][0],
                    position.x,
                    position.y,
                ),
            )
        )
    )
    remaining = tuple(
        sorted(
            (
                position
                for position in candidates
                if position not in selected
                and projected[position][0] > rear_level
            ),
            key=lambda position: (
                -_threat_pressure(position, threats, world),
                min(
                    position.chebyshev_distance(threat)
                    for threat in threats
                ),
                -projected[position][0],
                projected[position][1],
                position.x,
                position.y,
            ),
        )
    )
    return directional + remaining


def _nearest_flank(
    candidates: tuple[Position, ...],
    projected: dict[Position, tuple[float, float]],
    selected: set[Position],
    lateral_edge: float,
    front_level: float,
    rear_level: float,
) -> tuple[Position, ...]:
    return tuple(
        sorted(
            (
                position
                for position in candidates
                if projected[position][1] == lateral_edge
                and projected[position][0] < front_level
                and projected[position][0] > rear_level
                and position not in selected
            ),
            key=lambda position: (
                -projected[position][0],
                position.x,
                position.y,
            ),
        )[:3]
    )


def _reserve_controller_sites(
    world: WorldGrid,
    weapon_sites: tuple[WeaponSite, ...],
    wall_sites: tuple[Position, ...],
) -> tuple[Position, ...]:
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
                    and cell not in planned_walls
                ),
                key=lambda cell: (cell.x, cell.y),
            )
        )
        if not candidates:
            continue
        selected = candidates[0]
        reserved.append(selected)
    return tuple(reserved)


def _ring(
    world: WorldGrid,
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
        if (
            world.is_weapon_build_site(value)
            if distance == 1
            else world.is_wall_build_site(value)
        )
    )


def _center_distance(position: Position, center: tuple[float, float]) -> float:
    return max(abs(position.x - center[0]), abs(position.y - center[1]))
