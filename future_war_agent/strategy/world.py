from dataclasses import dataclass, field

from future_war_agent.protocol.models import Observation, Position, UnitState

from .rules import DEFAULT_RULES, RulesConfig, station_footprint


_ROLE_TYPES = frozenset({"worker", "pioneer"})
_WEAPON_TYPES = frozenset({"gatling", "railgun", "rocket"})


@dataclass(frozen=True, slots=True)
class WorldGrid:
    observation: Observation
    rules: RulesConfig
    neutral_cells: frozenset[Position]
    structure_cells: frozenset[Position]
    robot_cells: frozenset[Position]
    visible_enemy_role_cells: frozenset[Position]
    hard_blocked: frozenset[Position]
    soft_friendly: frozenset[Position]
    station_cells: frozenset[Position]
    friendly_roles: tuple[UnitState, ...]
    stations: tuple[UnitState, ...]
    weapons: tuple[UnitState, ...]
    walls: tuple[UnitState, ...]
    _path_cache: dict[tuple[object, ...], object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _heuristic_cache: dict[
        frozenset[Position], dict[Position, int]
    ] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _distance_cache: dict[
        tuple[frozenset[Position], frozenset[Position]],
        dict[Position, int],
    ] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _interaction_cache: dict[Position, tuple[Position, ...]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        rules: RulesConfig = DEFAULT_RULES,
    ) -> "WorldGrid":
        all_units = observation.our.units + observation.enemy.units
        structures = tuple(
            value
            for value in all_units
            if value.health > 0 and value.role_type not in _ROLE_TYPES
        )
        cells: set[Position] = set()
        for structure in structures:
            if structure.role_type == "station":
                cells.update(station_footprint(structure.position, rules))
            else:
                cells.add(structure.position)

        friendly_roles = tuple(
            sorted(
                (
                    value
                    for value in observation.our.units
                    if value.health > 0 and value.role_type in _ROLE_TYPES
                ),
                key=lambda value: value.unit_id,
            )
        )
        own_stations = tuple(
            sorted(
                (
                    value
                    for value in observation.our.units
                    if value.health > 0 and value.role_type == "station"
                ),
                key=lambda value: value.unit_id,
            )
        )
        station_cells = frozenset().union(
            *(
                station_footprint(station.position, rules)
                for station in own_stations
            )
        ) if own_stations else frozenset()
        neutral_cells = frozenset(
            zone.position for zone in observation.zones
        )
        structure_cells = frozenset(cells)
        robot_cells = frozenset(
            value.position for value in observation.robots if value.health > 0
        )
        visible_enemy_role_cells = frozenset(
            value.position
            for value in observation.enemy.units
            if value.health > 0 and value.role_type in _ROLE_TYPES
        )
        return cls(
            observation=observation,
            rules=rules,
            neutral_cells=neutral_cells,
            structure_cells=structure_cells,
            robot_cells=robot_cells,
            visible_enemy_role_cells=visible_enemy_role_cells,
            hard_blocked=frozenset().union(
                neutral_cells,
                structure_cells,
                robot_cells,
                visible_enemy_role_cells,
            ),
            soft_friendly=frozenset(value.position for value in friendly_roles),
            station_cells=station_cells,
            friendly_roles=friendly_roles,
            stations=own_stations,
            weapons=tuple(
                sorted(
                    (
                        value
                        for value in observation.our.units
                        if value.health > 0 and value.role_type in _WEAPON_TYPES
                    ),
                    key=lambda value: value.unit_id,
                )
            ),
            walls=tuple(
                sorted(
                    (
                        value
                        for value in observation.our.units
                        if value.health > 0 and value.role_type == "wall"
                    ),
                    key=lambda value: value.unit_id,
                )
            ),
        )

    def in_bounds(self, position: Position) -> bool:
        return (
            0 <= position.x < self.observation.width
            and 0 <= position.y < self.observation.height
        )

    def can_traverse(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.hard_blocked

    def unit_by_id(self, unit_id: int) -> UnitState | None:
        return next(
            (unit for unit in self.observation.our.units if unit.unit_id == unit_id),
            None,
        )

    def positions_for_zone(self, kind: str) -> tuple[Position, ...]:
        return tuple(
            sorted(
                (
                    zone.position
                    for zone in self.observation.zones
                    if zone.neutral_type == kind
                ),
                key=lambda position: (position.x, position.y),
            )
        )

    def interaction_cells(self, target: Position) -> tuple[Position, ...]:
        cached = self._interaction_cache.get(target)
        if cached is not None:
            return cached
        cells = (
            Position(target.x + dx, target.y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if dx != 0 or dy != 0
        )
        resolved = tuple(
            sorted(
                (cell for cell in cells if self.can_traverse(cell)),
                key=lambda position: (position.x, position.y),
            )
        )
        self._interaction_cache[target] = resolved
        return resolved

    def our_station(self) -> UnitState | None:
        return self.stations[0] if self.stations else None

    def is_geographic_land(self, position: Position) -> bool:
        if not self.in_bounds(position) or position in self.neutral_cells:
            return False
        return position not in self.station_cells

    def station_distance(self, position: Position) -> int | None:
        if not self.station_cells:
            return None
        return min(
            position.chebyshev_distance(cell) for cell in self.station_cells
        )

    def is_weapon_build_site(self, position: Position) -> bool:
        return self.is_geographic_land(position) and self.station_distance(position) == 1

    def is_wall_build_site(self, position: Position) -> bool:
        return self.is_geographic_land(position) and self.station_distance(position) == 2

    def is_legal_build_site(self, name: str, position: Position) -> bool:
        if name == 'wall':
            return self.is_wall_build_site(position)
        if name in _WEAPON_TYPES:
            return self.is_weapon_build_site(position)
        return False

    def is_inside_defense(self, position: Position) -> bool:
        distance = self.station_distance(position)
        return distance is not None and distance <= 1
