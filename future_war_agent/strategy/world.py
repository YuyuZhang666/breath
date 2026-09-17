from dataclasses import dataclass

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
    soft_friendly: frozenset[Position]
    friendly_roles: tuple[UnitState, ...]
    stations: tuple[UnitState, ...]
    weapons: tuple[UnitState, ...]
    walls: tuple[UnitState, ...]

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
        return cls(
            observation=observation,
            rules=rules,
            neutral_cells=frozenset(zone.position for zone in observation.zones),
            structure_cells=frozenset(cells),
            robot_cells=frozenset(
                value.position for value in observation.robots if value.health > 0
            ),
            visible_enemy_role_cells=frozenset(
                value.position
                for value in observation.enemy.units
                if value.health > 0 and value.role_type in _ROLE_TYPES
            ),
            soft_friendly=frozenset(value.position for value in friendly_roles),
            friendly_roles=friendly_roles,
            stations=tuple(
                sorted(
                    (
                        value
                        for value in observation.our.units
                        if value.health > 0 and value.role_type == "station"
                    ),
                    key=lambda value: value.unit_id,
                )
            ),
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

    @property
    def hard_blocked(self) -> frozenset[Position]:
        return frozenset().union(
            self.neutral_cells,
            self.structure_cells,
            self.robot_cells,
            self.visible_enemy_role_cells,
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
        cells = (
            Position(target.x + dx, target.y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if dx != 0 or dy != 0
        )
        return tuple(
            sorted(
                (cell for cell in cells if self.can_traverse(cell)),
                key=lambda position: (position.x, position.y),
            )
        )

    def our_station(self) -> UnitState | None:
        return self.stations[0] if self.stations else None
