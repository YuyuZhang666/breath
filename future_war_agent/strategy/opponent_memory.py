from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction

from future_war_agent.protocol.models import Observation, Position


WEAPON_ROLES = frozenset({'gatling', 'railgun', 'rocket'})
STRUCTURE_ROLES = frozenset({'station', 'wall', *WEAPON_ROLES})
MAX_STRUCTURE_SIGHTINGS = 128
MAX_ROLE_SIGHTINGS = 64
MAX_GROWTH_OBSERVATIONS = 32
MAX_DAMAGE_PATTERNS = 32
MAX_HALF_SUMMARIES = 8
MAX_HALF_TRANSITIONS = 8
MAX_KNOWN_STRUCTURE_IDS = 256


class OpponentTaskTendency(StrEnum):
    UNKNOWN = 'unknown'


class EvidenceAttribution(StrEnum):
    UNKNOWN = 'unknown'


@dataclass(frozen=True, slots=True)
class OpponentStructureSighting:
    unit_id: int
    role_type: str
    position: Position
    health: int
    level: int | None
    last_seen_round: int
    observation_tick: int


@dataclass(frozen=True, slots=True)
class OpponentRoleSighting:
    unit_id: int
    role_type: str
    position: Position
    health: int
    last_seen_round: int
    observation_tick: int


@dataclass(frozen=True, slots=True)
class DefenseGrowthObservation:
    half_index: int
    round_no: int
    elapsed_ticks: int
    added_structure_ids: tuple[int, ...]
    added_structure_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.elapsed_ticks <= 0:
            raise ValueError('defense growth observation needs positive elapsed time')
        if not self.added_structure_ids:
            raise ValueError('defense growth observation needs an addition')


@dataclass(frozen=True, slots=True)
class DamagePatternObservation:
    half_index: int
    round_no: int
    phase: str
    elapsed_ticks: int
    total_health_loss: int
    damaged_role_losses: tuple[tuple[str, int], ...]
    visible_enemy_weapon_types: tuple[str, ...]
    attribution: EvidenceAttribution = EvidenceAttribution.UNKNOWN

    def __post_init__(self) -> None:
        if self.elapsed_ticks <= 0 or self.total_health_loss <= 0:
            raise ValueError('damage observation must contain positive deltas')


@dataclass(frozen=True, slots=True)
class OpponentHalfSummary:
    half_index: int
    epoch: int
    side_key: str
    first_round: int
    last_round: int
    max_visible_walls: int = 0
    max_visible_weapons: int = 0
    max_visible_station_level: int = 0
    role_observation_count: int = 0
    task_proximity_observations: int = 0
    hostile_robot_peak: int = 0


@dataclass(frozen=True, slots=True)
class HalfTransitionObservation:
    from_half: int
    to_half: int
    from_side: str
    to_side: str
    round_no: int
    observed_wall_delta: int
    observed_weapon_delta: int
    observed_station_level_delta: int


@dataclass(frozen=True, slots=True)
class OwnHealthSample:
    unit_id: int
    role_type: str
    health: int


@dataclass(frozen=True, slots=True)
class OpponentMemory:
    structures: tuple[OpponentStructureSighting, ...] = ()
    last_visible_roles: tuple[OpponentRoleSighting, ...] = ()
    known_structure_ids: tuple[int, ...] = ()
    defense_growth: tuple[DefenseGrowthObservation, ...] = ()
    damage_patterns: tuple[DamagePatternObservation, ...] = ()
    half_summaries: tuple[OpponentHalfSummary, ...] = ()
    half_transitions: tuple[HalfTransitionObservation, ...] = ()
    visible_structure_count: int = 0
    visible_role_count: int = 0
    visible_hostile_robot_count: int = 0
    hostile_robot_peak: int = 0
    task_proximity_observations: int = 0
    task_tendency: OpponentTaskTendency = OpponentTaskTendency.UNKNOWN
    summon_attribution: EvidenceAttribution = EvidenceAttribution.UNKNOWN
    half_index: int = 0
    last_epoch: int = 0
    side_key: str = ''
    last_observation_tick: int = 0
    last_our_health: tuple[OwnHealthSample, ...] = ()
    defense_tracked_ticks: int = 0
    defense_added_structures: int = 0

    @property
    def defense_growth_per_100_rounds(self) -> Fraction:
        return (
            Fraction(
                self.defense_added_structures * 100,
                self.defense_tracked_ticks,
            )
            if self.defense_tracked_ticks
            else Fraction(0, 1)
        )

    @property
    def last_observed_damage(self) -> int:
        return self.damage_patterns[-1].total_health_loss if self.damage_patterns else 0

    def structure_log_entries(self) -> tuple[str, ...]:
        return tuple(
            (
                f'{item.unit_id}:{item.role_type}:'
                f'{item.position.x},{item.position.y}:'
                f'hp={item.health}:level={item.level}:'
                f'round={item.last_seen_round}'
            )
            for item in self.structures
        )

    def role_log_entries(self) -> tuple[str, ...]:
        return tuple(
            (
                f'{item.unit_id}:{item.role_type}:'
                f'{item.position.x},{item.position.y}:'
                f'hp={item.health}:round={item.last_seen_round}'
            )
            for item in self.last_visible_roles
        )

    def evidence_log_entries(self) -> tuple[str, ...]:
        growth = tuple(
            (
                f'growth:half={item.half_index}:round={item.round_no}:'
                f'elapsed={item.elapsed_ticks}:'
                f'ids={item.added_structure_ids}:'
                f'types={item.added_structure_types}'
            )
            for item in self.defense_growth
        )
        damage = tuple(
            (
                f'damage:half={item.half_index}:round={item.round_no}:'
                f'phase={item.phase}:loss={item.total_health_loss}:'
                f'roles={item.damaged_role_losses}:'
                f'visible_weapons={item.visible_enemy_weapon_types}:'
                f'attribution={item.attribution.value}'
            )
            for item in self.damage_patterns
        )
        halves = tuple(
            (
                f'half:{item.half_index}:epoch={item.epoch}:'
                f'side={item.side_key}:rounds={item.first_round}-{item.last_round}:'
                f'walls={item.max_visible_walls}:'
                f'weapons={item.max_visible_weapons}:'
                f'station_level={item.max_visible_station_level}:'
                f'task_proximity={item.task_proximity_observations}:'
                f'hostile_robot_peak={item.hostile_robot_peak}'
            )
            for item in self.half_summaries
        )
        transitions = tuple(
            (
                f'transition:{item.from_half}->{item.to_half}:'
                f'sides={item.from_side}->{item.to_side}:'
                f'round={item.round_no}:'
                f'wall_delta={item.observed_wall_delta}:'
                f'weapon_delta={item.observed_weapon_delta}:'
                f'station_level_delta={item.observed_station_level_delta}'
            )
            for item in self.half_transitions
        )
        return (*growth, *damage, *halves, *transitions)


PositionNormalizer = Callable[[Position], Position]


def update_opponent_memory(
    previous: OpponentMemory,
    observation: Observation,
    *,
    epoch: int,
    observation_tick: int,
    side_key: str,
    normalize: PositionNormalizer,
) -> OpponentMemory:
    effective_side_key = (
        previous.side_key
        if side_key == 'unknown' and previous.side_key
        else side_key
    )
    half_changed = bool(previous.side_key) and (
        epoch != previous.last_epoch
        or (
            previous.side_key != 'unknown'
            and effective_side_key != 'unknown'
            and effective_side_key != previous.side_key
        )
    )
    half_index = previous.half_index + (1 if half_changed else 0)
    visible_structures = tuple(
        unit
        for unit in observation.enemy.units
        if unit.health > 0 and unit.role_type in STRUCTURE_ROLES
    )
    visible_roles = tuple(
        unit
        for unit in observation.enemy.units
        if unit.health > 0 and unit.role_type not in STRUCTURE_ROLES
    )

    structures_by_id = {item.unit_id: item for item in previous.structures}
    for unit in visible_structures:
        structures_by_id[unit.unit_id] = OpponentStructureSighting(
            unit_id=unit.unit_id,
            role_type=unit.role_type.strip(),
            position=normalize(unit.position),
            health=max(0, unit.health),
            level=unit.level,
            last_seen_round=observation.time.round_no,
            observation_tick=observation_tick,
        )
    structures = tuple(
        sorted(
            structures_by_id.values(),
            key=lambda item: (
                -item.observation_tick,
                item.role_type,
                item.unit_id,
            ),
        )[:MAX_STRUCTURE_SIGHTINGS]
    )

    roles_by_id = {item.unit_id: item for item in previous.last_visible_roles}
    for unit in visible_roles:
        roles_by_id[unit.unit_id] = OpponentRoleSighting(
            unit_id=unit.unit_id,
            role_type=unit.role_type.strip(),
            position=normalize(unit.position),
            health=max(0, unit.health),
            last_seen_round=observation.time.round_no,
            observation_tick=observation_tick,
        )
    roles = tuple(
        sorted(
            roles_by_id.values(),
            key=lambda item: (-item.observation_tick, item.unit_id),
        )[:MAX_ROLE_SIGHTINGS]
    )

    previously_known = set(previous.known_structure_ids)
    visible_ids = {unit.unit_id for unit in visible_structures}
    added_ids = tuple(sorted(visible_ids - previously_known))
    elapsed_ticks = max(0, observation_tick - previous.last_observation_tick)
    ids_safe_to_commit = (
        visible_ids
        if (
            previous.last_observation_tick == 0
            or half_changed
            or elapsed_ticks > 0
        )
        else set()
    )
    known_ids = tuple(
        sorted(previously_known | ids_safe_to_commit)[:MAX_KNOWN_STRUCTURE_IDS]
    )
    growth = previous.defense_growth
    defense_tracked_ticks = previous.defense_tracked_ticks
    defense_added_structures = previous.defense_added_structures
    if (
        previous.last_observation_tick > 0
        and not half_changed
        and elapsed_ticks > 0
    ):
        defense_tracked_ticks += elapsed_ticks
        defense_added_structures += len(added_ids)
    if (
        previous.last_observation_tick > 0
        and not half_changed
        and elapsed_ticks > 0
        and added_ids
    ):
        added_types = tuple(
            sorted(
                unit.role_type
                for unit in visible_structures
                if unit.unit_id in added_ids
            )
        )
        growth = (
            *growth,
            DefenseGrowthObservation(
                half_index=half_index,
                round_no=observation.time.round_no,
                elapsed_ticks=elapsed_ticks,
                added_structure_ids=added_ids,
                added_structure_types=added_types,
            ),
        )[-MAX_GROWTH_OBSERVATIONS:]

    task_positions = tuple(
        zone.position
        for zone in observation.zones
        if 'taskpoint' in zone.neutral_type.replace('_', '').casefold()
    )
    task_proximity = any(
        unit.position.chebyshev_distance(position) <= 1
        for unit in visible_roles
        for position in task_positions
    )
    count_observation = (
        previous.last_observation_tick == 0
        or half_changed
        or elapsed_ticks > 0
    )
    task_observation_increment = int(task_proximity and count_observation)
    task_count = (
        previous.task_proximity_observations + task_observation_increment
    )
    hostile_robot_count = sum(
        robot.health > 0 and robot.target_team == observation.our.team_type
        for robot in observation.robots
    )

    current_health = tuple(
        OwnHealthSample(unit.unit_id, unit.role_type, max(0, unit.health))
        for unit in sorted(observation.our.units, key=lambda item: item.unit_id)
        if unit.health > 0
    )
    damage_patterns = previous.damage_patterns
    if not half_changed and elapsed_ticks > 0 and previous.last_our_health:
        before = {item.unit_id: item for item in previous.last_our_health}
        losses: dict[str, int] = {}
        for item in current_health:
            prior = before.get(item.unit_id)
            if prior is None:
                continue
            loss = max(0, prior.health - item.health)
            if loss:
                losses[item.role_type] = losses.get(item.role_type, 0) + loss
        total_loss = sum(losses.values())
        if total_loss:
            damage_patterns = (
                *damage_patterns,
                DamagePatternObservation(
                    half_index=half_index,
                    round_no=observation.time.round_no,
                    phase=observation.time.phase.value,
                    elapsed_ticks=elapsed_ticks,
                    total_health_loss=total_loss,
                    damaged_role_losses=tuple(sorted(losses.items())),
                    visible_enemy_weapon_types=tuple(
                        sorted(
                            {
                                unit.role_type
                                for unit in visible_structures
                                if unit.role_type in WEAPON_ROLES
                            }
                        )
                    ),
                ),
            )[-MAX_DAMAGE_PATTERNS:]

    visible_wall_count = sum(
        unit.role_type == 'wall' for unit in visible_structures
    )
    visible_weapon_count = sum(
        unit.role_type in WEAPON_ROLES for unit in visible_structures
    )
    visible_station_level = max(
        (
            unit.level or 0
            for unit in visible_structures
            if unit.role_type == 'station'
        ),
        default=0,
    )
    summaries = previous.half_summaries
    transitions = previous.half_transitions
    current_summary = OpponentHalfSummary(
        half_index=half_index,
        epoch=epoch,
        side_key=effective_side_key,
        first_round=observation.time.round_no,
        last_round=observation.time.round_no,
        max_visible_walls=visible_wall_count,
        max_visible_weapons=visible_weapon_count,
        max_visible_station_level=visible_station_level,
        role_observation_count=(
            len(visible_roles) if count_observation else 0
        ),
        task_proximity_observations=task_observation_increment,
        hostile_robot_peak=hostile_robot_count,
    )
    prior_summary = summaries[-1] if summaries else None
    if prior_summary is not None and not half_changed:
        current_summary = replace(
            prior_summary,
            side_key=(
                effective_side_key
                if prior_summary.side_key == 'unknown'
                else prior_summary.side_key
            ),
            last_round=observation.time.round_no,
            max_visible_walls=max(
                prior_summary.max_visible_walls,
                visible_wall_count,
            ),
            max_visible_weapons=max(
                prior_summary.max_visible_weapons,
                visible_weapon_count,
            ),
            max_visible_station_level=max(
                prior_summary.max_visible_station_level,
                visible_station_level,
            ),
            role_observation_count=(
                prior_summary.role_observation_count
                + (len(visible_roles) if count_observation else 0)
            ),
            task_proximity_observations=(
                prior_summary.task_proximity_observations
                + task_observation_increment
            ),
            hostile_robot_peak=max(
                prior_summary.hostile_robot_peak,
                hostile_robot_count,
            ),
        )
        summaries = (*summaries[:-1], current_summary)
    else:
        if prior_summary is not None:
            transitions = (
                *transitions,
                HalfTransitionObservation(
                    from_half=prior_summary.half_index,
                    to_half=half_index,
                    from_side=prior_summary.side_key,
                    to_side=effective_side_key,
                    round_no=observation.time.round_no,
                    observed_wall_delta=(
                        visible_wall_count - prior_summary.max_visible_walls
                    ),
                    observed_weapon_delta=(
                        visible_weapon_count - prior_summary.max_visible_weapons
                    ),
                    observed_station_level_delta=(
                        visible_station_level
                        - prior_summary.max_visible_station_level
                    ),
                ),
            )[-MAX_HALF_TRANSITIONS:]
        summaries = (*summaries, current_summary)[-MAX_HALF_SUMMARIES:]

    return OpponentMemory(
        structures=structures,
        last_visible_roles=roles,
        known_structure_ids=known_ids,
        defense_growth=growth,
        damage_patterns=damage_patterns,
        half_summaries=summaries,
        half_transitions=transitions,
        visible_structure_count=len(visible_structures),
        visible_role_count=len(visible_roles),
        visible_hostile_robot_count=hostile_robot_count,
        hostile_robot_peak=max(previous.hostile_robot_peak, hostile_robot_count),
        task_proximity_observations=task_count,
        task_tendency=OpponentTaskTendency.UNKNOWN,
        summon_attribution=EvidenceAttribution.UNKNOWN,
        half_index=half_index,
        last_epoch=epoch,
        side_key=effective_side_key,
        last_observation_tick=observation_tick,
        last_our_health=current_health,
        defense_tracked_ticks=defense_tracked_ticks,
        defense_added_structures=defense_added_structures,
    )
