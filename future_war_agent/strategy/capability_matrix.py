from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class CapabilityStatus(StrEnum):
    UNKNOWN = 'unknown'
    SUPPORTED = 'supported'
    UNSUPPORTED = 'unsupported'


class Capability(StrEnum):
    ATTACK_ENEMY_STATION = 'attack_enemy_station'
    ATTACK_ENEMY_ROLE = 'attack_enemy_role'
    ATTACK_ENEMY_WEAPON = 'attack_enemy_weapon'
    ATTACK_ENEMY_SIDE_ROBOT = 'attack_enemy_side_robot'
    CROSS_TEAM_KILL_SCORE = 'cross_team_kill_score'
    COLLECT_AT_NIGHT = 'collect_at_night'
    REMOVE_ENEMY_WALL = 'remove_enemy_wall'
    CONTROLLER_CAN_ALSO_ACT = 'controller_can_also_act'
    STUN_REFRESHES = 'stun_refreshes'
    DUPLICATE_ROCKET_TARGETS = 'duplicate_rocket_targets'
    GAME_CONTINUES_AFTER_ONE_STATION_DESTROYED = (
        'game_continues_after_one_station_destroyed'
    )


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    capability: Capability
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    source: str = 'default_unknown'
    note: str = ''

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError('capability source must be nonblank')


def _unknown_records() -> tuple[CapabilityRecord, ...]:
    return tuple(CapabilityRecord(capability) for capability in Capability)


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    records: tuple[CapabilityRecord, ...] = _unknown_records()

    def __post_init__(self) -> None:
        capabilities = tuple(record.capability for record in self.records)
        if len(capabilities) != len(Capability):
            raise ValueError('capability matrix must contain every capability')
        if len(set(capabilities)) != len(capabilities):
            raise ValueError('capability matrix contains duplicate capabilities')
        if set(capabilities) != set(Capability):
            raise ValueError('capability matrix contains an invalid capability set')

    @classmethod
    def from_manual_config(
        cls,
        values: Mapping[
            Capability | str,
            CapabilityStatus | str | CapabilityRecord,
        ],
    ) -> 'CapabilityMatrix':
        overrides: dict[Capability, CapabilityRecord] = {}
        for raw_capability, raw_value in values.items():
            try:
                capability = (
                    raw_capability
                    if isinstance(raw_capability, Capability)
                    else Capability(str(raw_capability))
                )
            except ValueError as error:
                raise ValueError(
                    f'unknown capability: {raw_capability!r}'
                ) from error
            if isinstance(raw_value, CapabilityRecord):
                if raw_value.capability is not capability:
                    raise ValueError('capability record key does not match record')
                record = raw_value
            else:
                try:
                    status = (
                        raw_value
                        if isinstance(raw_value, CapabilityStatus)
                        else CapabilityStatus(str(raw_value))
                    )
                except ValueError as error:
                    raise ValueError(
                        f'invalid status for {capability.value}: {raw_value!r}'
                    ) from error
                record = CapabilityRecord(
                    capability=capability,
                    status=status,
                    source='manual_config',
                )
            overrides[capability] = record
        return cls(
            tuple(
                overrides.get(capability, CapabilityRecord(capability))
                for capability in Capability
            )
        )

    def record(self, capability: Capability | str) -> CapabilityRecord:
        normalized = (
            capability
            if isinstance(capability, Capability)
            else Capability(str(capability))
        )
        return next(
            record
            for record in self.records
            if record.capability is normalized
        )

    def status(self, capability: Capability | str) -> CapabilityStatus:
        return self.record(capability).status

    def supports(self, capability: Capability | str) -> bool:
        return self.status(capability) is CapabilityStatus.SUPPORTED

    def supports_all(self, *capabilities: Capability | str) -> bool:
        return bool(capabilities) and all(
            self.supports(capability) for capability in capabilities
        )

    def status_counts(self) -> tuple[int, int, int]:
        return (
            sum(
                record.status is CapabilityStatus.UNKNOWN
                for record in self.records
            ),
            sum(
                record.status is CapabilityStatus.SUPPORTED
                for record in self.records
            ),
            sum(
                record.status is CapabilityStatus.UNSUPPORTED
                for record in self.records
            ),
        )

    def log_entries(self) -> tuple[str, ...]:
        return tuple(
            f'{record.capability.value}:{record.status.value}:{record.source}'
            for record in self.records
        )


UNKNOWN_CAPABILITY_MATRIX = CapabilityMatrix()
