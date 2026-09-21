import unittest

from future_war_agent.protocol.models import Position, Zone
from future_war_agent.strategy.memory import MatchMemoryStore
from future_war_agent.strategy.opponent_memory import (
    EvidenceAttribution,
    MAX_DAMAGE_PATTERNS,
    OpponentTaskTendency,
)
from tests.strategy_helpers import observation, robot, unit


def station(unit_id: int, x: int, y: int, *, health: int = 1000):
    return unit(unit_id, x, y, 'station', health=health, level=1)


class OpponentMemoryTests(unittest.TestCase):
    def test_visible_structures_roles_and_pressure_are_recorded_canonically(self) -> None:
        store = MatchMemoryStore()
        observed = observation(
            our_units=(station(10, 2, 2),),
            enemy_units=(
                station(90, 12, 12),
                unit(91, 11, 12, 'wall', health=300),
                unit(92, 10, 12, 'rocket', health=200, level=2),
                unit(93, 9, 9, 'worker'),
            ),
            robots=(robot(501, 7, 7, target_team='challenger'),),
        )

        memory = store.observe(observed).opponent_memory

        self.assertEqual(memory.visible_structure_count, 3)
        self.assertEqual(memory.visible_role_count, 1)
        self.assertEqual(memory.visible_hostile_robot_count, 1)
        self.assertEqual(memory.summon_attribution, EvidenceAttribution.UNKNOWN)
        self.assertEqual(
            next(item for item in memory.structures if item.unit_id == 90).position,
            Position(2, 2),
        )
        self.assertEqual(memory.last_visible_roles[0].position, Position(5, 5))

    def test_only_newly_seen_structures_count_as_defense_growth(self) -> None:
        store = MatchMemoryStore()
        ours = (station(10, 12, 12),)
        store.observe(
            observation(
                round_no=1,
                our_units=ours,
                enemy_units=(station(90, 2, 2),),
            )
        )
        memory = store.observe(
            observation(
                round_no=2,
                our_units=ours,
                enemy_units=(
                    station(90, 2, 2),
                    unit(91, 3, 2, 'wall'),
                    unit(92, 4, 2, 'gatling'),
                ),
            )
        ).opponent_memory

        self.assertEqual(len(memory.defense_growth), 1)
        self.assertEqual(memory.defense_growth[0].added_structure_ids, (91, 92))
        self.assertEqual(memory.defense_growth_per_100_rounds, 200)

        hidden = store.observe(
            observation(
                round_no=3,
                our_units=ours,
                enemy_units=(station(90, 2, 2),),
            )
        ).opponent_memory
        self.assertEqual(len(hidden.defense_growth), 1)
        self.assertEqual(hidden.defense_growth_per_100_rounds, 100)

    def test_damage_pattern_records_context_without_claiming_attribution(self) -> None:
        store = MatchMemoryStore()
        enemy = (unit(92, 4, 2, 'rocket', health=200),)
        store.observe(
            observation(
                round_no=1,
                our_units=(
                    station(10, 12, 12, health=1000),
                    unit(11, 10, 10, 'worker', health=100),
                ),
                enemy_units=enemy,
            )
        )
        memory = store.observe(
            observation(
                round_no=2,
                our_units=(
                    station(10, 12, 12, health=900),
                    unit(11, 10, 10, 'worker', health=80),
                ),
                enemy_units=enemy,
            )
        ).opponent_memory

        pattern = memory.damage_patterns[-1]
        self.assertEqual(pattern.total_health_loss, 120)
        self.assertEqual(
            pattern.damaged_role_losses,
            (('station', 100), ('worker', 20)),
        )
        self.assertEqual(pattern.visible_enemy_weapon_types, ('rocket',))
        self.assertEqual(pattern.attribution, EvidenceAttribution.UNKNOWN)

    def test_task_proximity_is_logged_but_tendency_remains_unknown(self) -> None:
        store = MatchMemoryStore()
        memory = store.observe(
            observation(
                our_units=(station(10, 12, 12),),
                enemy_units=(unit(90, 4, 5, 'pioneer'),),
                zones=(Zone(Position(5, 5), 'challengerTaskPoint2'),),
            )
        ).opponent_memory

        self.assertEqual(memory.task_proximity_observations, 1)
        self.assertEqual(memory.task_tendency, OpponentTaskTendency.UNKNOWN)

    def test_side_change_creates_half_summary_without_false_growth(self) -> None:
        store = MatchMemoryStore()
        store.observe(
            observation(
                round_no=1,
                our_units=(station(10, 2, 2),),
                enemy_units=(unit(90, 12, 12, 'wall'),),
            )
        )
        memory = store.observe(
            observation(
                round_no=2,
                our_units=(station(10, 12, 12),),
                enemy_units=(
                    unit(90, 2, 2, 'wall'),
                    unit(91, 3, 2, 'rocket'),
                ),
            )
        ).opponent_memory

        self.assertEqual(memory.half_index, 1)
        self.assertEqual(len(memory.half_summaries), 2)
        self.assertEqual(len(memory.half_transitions), 1)
        self.assertEqual(memory.defense_growth, ())
        self.assertTrue(
            any(
                entry.startswith('transition:0->1')
                for entry in memory.evidence_log_entries()
            )
        )

    def test_same_round_revision_defers_growth_until_time_advances(self) -> None:
        store = MatchMemoryStore()
        ours = (station(10, 12, 12),)
        store.observe(
            observation(
                round_no=1,
                our_units=ours,
                enemy_units=(station(90, 2, 2),),
            )
        )
        revised = store.observe(
            observation(
                round_no=1,
                our_units=ours,
                enemy_units=(
                    station(90, 2, 2),
                    unit(91, 3, 2, 'wall'),
                ),
            )
        ).opponent_memory
        advanced = store.observe(
            observation(
                round_no=2,
                our_units=ours,
                enemy_units=(
                    station(90, 2, 2),
                    unit(91, 3, 2, 'wall'),
                ),
            )
        ).opponent_memory

        self.assertEqual(revised.defense_growth, ())
        self.assertEqual(advanced.defense_growth[0].added_structure_ids, (91,))

    def test_station_visibility_does_not_create_a_false_half_transition(self) -> None:
        store = MatchMemoryStore()
        store.observe(
            observation(
                round_no=1,
                enemy_units=(unit(90, 4, 4, 'worker'),),
            )
        )
        memory = store.observe(
            observation(
                round_no=2,
                our_units=(station(10, 12, 12),),
                enemy_units=(unit(90, 4, 4, 'worker'),),
            )
        ).opponent_memory

        self.assertEqual(memory.half_index, 0)
        self.assertEqual(memory.half_transitions, ())

    def test_damage_history_is_bounded(self) -> None:
        store = MatchMemoryStore()
        health = 1000
        store.observe(
            observation(
                round_no=1,
                our_units=(station(10, 12, 12, health=health),),
            )
        )
        round_no = 2
        for _ in range(MAX_DAMAGE_PATTERNS + 5):
            health -= 1
            store.observe(
                observation(
                    round_no=round_no,
                    our_units=(station(10, 12, 12, health=health),),
                )
            )
            round_no += 1

        memory = store.get('team').opponent_memory
        self.assertEqual(len(memory.damage_patterns), MAX_DAMAGE_PATTERNS)


if __name__ == '__main__':
    unittest.main()
