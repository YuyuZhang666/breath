from collections import Counter
from dataclasses import dataclass

from future_war_agent.protocol.models import UnitState


@dataclass(frozen=True, slots=True)
class CoreWeaponReadiness:
    required_count: int
    ready_count: int
    missing_types: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.ready_count >= self.required_count


def core_weapon_readiness(
    units: tuple[UnitState, ...],
    weapon_loadout: tuple[str, ...],
) -> CoreWeaponReadiness:
    living_counts = Counter(
        unit.role_type for unit in units if unit.health > 0
    )
    missing: list[str] = []
    ready_count = 0
    for weapon_type in weapon_loadout:
        if living_counts[weapon_type] > 0:
            living_counts[weapon_type] -= 1
            ready_count += 1
        else:
            missing.append(weapon_type)
    return CoreWeaponReadiness(
        required_count=len(weapon_loadout),
        ready_count=ready_count,
        missing_types=tuple(missing),
    )
