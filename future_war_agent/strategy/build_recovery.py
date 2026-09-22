from dataclasses import dataclass

from future_war_agent.decision.actions import ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position


BUILD_FAILURE_TTL_ROUNDS = 20
MAX_BUILD_FAILURE_RECORDS = 32
MAX_BUILD_FAILURE_COOLDOWN_ROUNDS = 8


@dataclass(frozen=True, slots=True)
class BuildFailureRecord:
    name: str
    target: Position
    consecutive_failures: int
    last_failure_round: int
    cooldown_until_round: int

    @property
    def key(self) -> tuple[str, Position]:
        return self.name, self.target


@dataclass(frozen=True, slots=True)
class BuildRecoveryState:
    failures: tuple[BuildFailureRecord, ...] = ()

    def cooldown_targets(
        self,
        name: str,
        round_no: int,
    ) -> frozenset[Position]:
        return frozenset(
            record.target
            for record in self.failures
            if record.name == name
            and round_no <= record.cooldown_until_round
        )

    def reroute_targets(
        self,
        name: str,
        round_no: int,
    ) -> frozenset[Position]:
        return frozenset(
            record.target
            for record in self.failures
            if record.name == name
            and record.consecutive_failures >= 2
            and round_no - record.last_failure_round
            <= BUILD_FAILURE_TTL_ROUNDS
        )

    def log_entries(self, round_no: int) -> tuple[str, ...]:
        return tuple(
            f'{record.name}@({record.target.x},{record.target.y}):'
            f'failures={record.consecutive_failures}:'
            f'cooldown={max(0, record.cooldown_until_round - round_no + 1)}'
            for record in self.failures[:MAX_BUILD_FAILURE_RECORDS]
        )


EMPTY_BUILD_RECOVERY_STATE = BuildRecoveryState()


def update_build_recovery(
    previous: BuildRecoveryState,
    observation: Observation,
    previous_decision: Decision | None,
) -> BuildRecoveryState:
    round_no = observation.time.round_no
    active = {
        record.key: record
        for record in previous.failures
        if round_no - record.last_failure_round <= BUILD_FAILURE_TTL_ROUNDS
    }
    if previous_decision is not None:
        actual_builds = {
            (unit.role_type, unit.position)
            for unit in observation.our.units
            if unit.health > 0
        }
        for actor_id, action in sorted(previous_decision.commands.items()):
            if (
                action is None
                or action.kind is not ActionKind.BUILD
                or action.name is None
                or len(action.target_positions) != 1
            ):
                continue
            target = action.target_positions[0]
            key = action.name, target
            if key in actual_builds:
                active.pop(key, None)
                continue
            succeeded = observation.last_action_results.get(actor_id)
            if succeeded is None:
                continue
            prior = active.get(key)
            failures = (
                prior.consecutive_failures + 1
                if prior is not None
                else 1
            )
            cooldown_rounds = min(
                MAX_BUILD_FAILURE_COOLDOWN_ROUNDS,
                2 ** (failures - 1),
            )
            active[key] = BuildFailureRecord(
                name=action.name,
                target=target,
                consecutive_failures=failures,
                last_failure_round=round_no,
                cooldown_until_round=round_no + cooldown_rounds - 1,
            )
    failures = tuple(
        sorted(
            active.values(),
            key=lambda record: (
                -record.last_failure_round,
                record.name,
                record.target.x,
                record.target.y,
            ),
        )[:MAX_BUILD_FAILURE_RECORDS]
    )
    return BuildRecoveryState(failures=failures)
