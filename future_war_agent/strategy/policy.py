from dataclasses import dataclass, field
from enum import StrEnum

from .simulation.objective import DEFAULT_NIGHT_OBJECTIVE, NightObjective


class StrategyProfile(StrEnum):
    SURVIVE = "survive"
    ECONOMY = "economy"
    SCORE = "score"
    PRESSURE = "pressure"
    DESPERATION = "desperation"


@dataclass(frozen=True, slots=True)
class RuleFeatureFlags:
    enable_pressure: bool = False
    enable_bomb: bool = False
    enable_stun: bool = False
    enable_repairs: bool = False
    enable_upgrades: bool = False
    enable_enemy_wall_removal: bool = False
    enable_blind_fire: bool = False
    enable_rocket_structure_damage: bool = False
    enable_treasure_retry: bool = False
    enable_attack_enemy_station: bool = False
    enable_attack_enemy_role: bool = False
    enable_cross_team_robot_attack: bool = False
    enable_night_mining: bool = False
    enable_controller_personal_action: bool = False
    enable_task_execute_commands: bool = False


@dataclass(frozen=True, slots=True)
class BuildPlan:
    weapon_loadout: tuple[str, ...] = ("gatling", "railgun", "rocket")
    wall_site_limit: int | None = None
    build_weapons: bool = True
    build_walls: bool = True
    minimum_weapons_before_walls: int = 1
    opening_critical_wall_count: int = 3
    critical_wall_priority_boost: int = 75
    threat_wall_priority_boost: int = 30
    max_wall_job_candidates: int = 4

    def __post_init__(self) -> None:
        if any(not name.strip() for name in self.weapon_loadout):
            raise ValueError("weapon names must be non-blank")
        if len(set(self.weapon_loadout)) != len(self.weapon_loadout):
            raise ValueError("weapon loadout entries must be unique")
        if self.wall_site_limit is not None and self.wall_site_limit < 0:
            raise ValueError("wall_site_limit cannot be negative")
        if self.minimum_weapons_before_walls < 0:
            raise ValueError('minimum_weapons_before_walls cannot be negative')
        if self.opening_critical_wall_count < 0:
            raise ValueError('opening_critical_wall_count cannot be negative')
        if self.critical_wall_priority_boost < 0:
            raise ValueError('critical_wall_priority_boost cannot be negative')
        if self.threat_wall_priority_boost < 0:
            raise ValueError('threat_wall_priority_boost cannot be negative')
        if self.max_wall_job_candidates <= 0:
            raise ValueError('max_wall_job_candidates must be positive')


@dataclass(frozen=True, slots=True)
class DayPriorities:
    recall: int = 500
    emergency_item: int = 475
    build_weapon: int = 400
    build_wall: int = 350
    purchase: int = 325
    sell: int = 300
    collect: int = 200
    scout: int = 150
    preposition: int = 100

    def __post_init__(self) -> None:
        values = (
            self.recall,
            self.emergency_item,
            self.build_weapon,
            self.build_wall,
            self.purchase,
            self.sell,
            self.collect,
            self.scout,
            self.preposition,
        )
        if any(value <= 0 for value in values):
            raise ValueError("day priorities must be positive")


@dataclass(frozen=True, slots=True)
class ItemPolicy:
    medicine_name: str = "Medicine"
    medicine_health_threshold: int = 0
    medicine_stock: int = 0
    bomb_name: str | None = None
    stun_name: str | None = None
    repair_name: str | None = None
    upgrade_name: str | None = None

    def __post_init__(self) -> None:
        if not self.medicine_name.strip():
            raise ValueError("medicine_name must be non-blank")
        if self.medicine_health_threshold < 0:
            raise ValueError("medicine_health_threshold cannot be negative")
        if self.medicine_stock < 0:
            raise ValueError("medicine_stock cannot be negative")
        optional_names = (
            self.bomb_name,
            self.stun_name,
            self.repair_name,
            self.upgrade_name,
        )
        if any(name is not None and not name.strip() for name in optional_names):
            raise ValueError("configured item names must be non-blank")


DEFAULT_BUILD_PLAN = BuildPlan()
DEFAULT_DAY_PRIORITIES = DayPriorities()
DEFAULT_ITEM_POLICY = ItemPolicy()


@dataclass(frozen=True, slots=True)
class StrategicIntent:
    profile: StrategyProfile = StrategyProfile.ECONOMY
    build_plan: BuildPlan = DEFAULT_BUILD_PLAN
    day_priorities: DayPriorities = DEFAULT_DAY_PRIORITIES
    item_policy: ItemPolicy = DEFAULT_ITEM_POLICY
    feature_flags: RuleFeatureFlags = field(default_factory=RuleFeatureFlags)
    gold_reserve: int = 0
    reserve_eligible_actions: frozenset[tuple[str, str]] = frozenset()
    minimum_hold_rounds: int = 0
    allow_mining: bool = True
    allow_tasks: bool = True
    allow_scouting: bool = False
    allow_pressure: bool = False
    night_objective: NightObjective = DEFAULT_NIGHT_OBJECTIVE
    transition_reason: str = "phase2-compatible default"

    def __post_init__(self) -> None:
        if self.gold_reserve < 0:
            raise ValueError("gold_reserve cannot be negative")
        if self.minimum_hold_rounds < 0:
            raise ValueError("minimum_hold_rounds cannot be negative")
        if not self.transition_reason.strip():
            raise ValueError("transition_reason must be non-blank")


DEFAULT_STRATEGIC_INTENT = StrategicIntent()


def intent_for_profile(
    profile: StrategyProfile,
    reason: str,
    *,
    feature_flags: RuleFeatureFlags | None = None,
) -> StrategicIntent:
    flags = feature_flags if feature_flags is not None else RuleFeatureFlags()
    if profile is StrategyProfile.SURVIVE:
        return StrategicIntent(
            profile=profile,
            item_policy=ItemPolicy(
                medicine_health_threshold=100,
                medicine_stock=2,
            ),
            feature_flags=flags,
            gold_reserve=100,
            minimum_hold_rounds=3,
            allow_tasks=False,
            transition_reason=reason,
        )
    if profile is StrategyProfile.ECONOMY:
        return StrategicIntent(
            profile=profile,
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
            feature_flags=flags,
            gold_reserve=25,
            minimum_hold_rounds=4,
            allow_tasks=True,
            transition_reason=reason,
        )
    if profile is StrategyProfile.SCORE:
        return StrategicIntent(
            profile=profile,
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
            feature_flags=flags,
            gold_reserve=50,
            minimum_hold_rounds=5,
            allow_tasks=True,
            transition_reason=reason,
        )
    if profile is StrategyProfile.PRESSURE:
        return StrategicIntent(
            profile=profile,
            item_policy=ItemPolicy(
                medicine_health_threshold=80,
                medicine_stock=2,
            ),
            feature_flags=flags,
            gold_reserve=50,
            minimum_hold_rounds=5,
            allow_tasks=True,
            allow_scouting=True,
            allow_pressure=flags.enable_pressure,
            transition_reason=reason,
        )
    if profile is StrategyProfile.DESPERATION:
        return StrategicIntent(
            profile=profile,
            build_plan=BuildPlan(build_walls=False),
            item_policy=ItemPolicy(
                medicine_health_threshold=150,
                medicine_stock=0,
            ),
            feature_flags=flags,
            gold_reserve=0,
            minimum_hold_rounds=1,
            allow_tasks=True,
            allow_scouting=True,
            allow_pressure=flags.enable_pressure,
            night_objective=NightObjective(
                protect_all_controllers=False,
                protect_all_weapons=False,
                require_post_horizon_buffer=False,
                enable_score_band=True,
            ),
            transition_reason=reason,
        )
    raise ValueError(f"unsupported strategy profile: {profile!r}")
