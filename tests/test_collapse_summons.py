import unittest

from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.collapse_summons import (
    evaluate_collapse_summons,
    generate_collapse_summon_jobs,
)
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.opponent_memory import (
    OpponentMemory,
    OpponentStationDamageObservation,
)
from future_war_agent.strategy.policy import (
    BuildPlan,
    RuleFeatureFlags,
    StrategicIntent,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


SUMMON_SHOP = (
    ShopItem('SmallRobotSummonOrder', 20),
    ShopItem('MiddleRobotSummonOrder', 30),
    ShopItem('LargeRobotSummonOrder', 100),
    ShopItem('BossRobotSummonOrder', 200),
)


def enemy_snapshot(
    *,
    station_health=200,
    weapons=1,
    roles=1,
    walls=0,
    weapon_level=1,
):
    values = [unit(90, 12, 12, 'station', health=station_health, level=1)]
    values.extend(
        unit(100 + index, 11 - index, 12, 'gatling', level=weapon_level)
        for index in range(weapons)
    )
    values.extend(
        unit(110 + index, 10, 10 - index, 'worker')
        for index in range(roles)
    )
    values.extend(
        unit(120 + index, 11, 11 - index, 'wall', health=1000, level=1)
        for index in range(walls)
    )
    return tuple(values)


def sustained_memory(day_no=1):
    return OpponentMemory(
        station_damage_history=(
            OpponentStationDamageObservation(
                half_index=0, round_no=120, day_no=day_no,
                phase='night', elapsed_ticks=1, health_loss=50, health_after=1050,
            ),
            OpponentStationDamageObservation(
                half_index=0, round_no=121, day_no=day_no,
                phase='night', elapsed_ticks=1, health_loss=50, health_after=1000,
            ),
        ),
    )



class CollapseEvaluationTests(unittest.TestCase):
    def test_critical_weak_snapshot_selects_cheapest_breakthrough(self):
        observed = observation(
            round_no=131,
            enemy_units=enemy_snapshot(),
            weapon_shop=SUMMON_SHOP,
        )

        plans = evaluate_collapse_summons(
            observed,
            OpponentMemory(),
            available_names=frozenset(item.name for item in SUMMON_SHOP),
        )

        self.assertTrue(plans)
        self.assertEqual(plans[0].item_name, 'LargeRobotSummonOrder')
        self.assertGreaterEqual(plans[0].advance_rounds, 3)
        self.assertIn('station_critical', plans[0].evidence)

    def test_strong_defense_that_prevents_first_fall_fails_closed(self):
        observed = observation(
            round_no=131,
            enemy_units=enemy_snapshot(
                station_health=350, weapons=3, roles=3, weapon_level=3
            ),
            weapon_shop=SUMMON_SHOP,
        )

        plans = evaluate_collapse_summons(
            observed, OpponentMemory(),
            available_names=frozenset(item.name for item in SUMMON_SHOP),
        )

        self.assertEqual(plans, ())

    def test_missing_controller_snapshot_fails_closed(self):
        observed = observation(
            round_no=131,
            enemy_units=enemy_snapshot(roles=0),
            weapon_shop=SUMMON_SHOP,
        )
        self.assertEqual(
            evaluate_collapse_summons(
                observed, OpponentMemory(),
                available_names=frozenset(item.name for item in SUMMON_SHOP),
            ),
            (),
        )

    def test_unknown_station_or_weapon_level_fails_closed(self):
        unknown_station = observation(
            round_no=131,
            enemy_units=(
                unit(90, 12, 12, 'station', health=200, level=None),
                unit(91, 11, 12, 'worker'),
            ),
            weapon_shop=SUMMON_SHOP,
        )
        unknown_weapon = observation(
            round_no=131,
            enemy_units=(
                unit(90, 12, 12, 'station', health=200, level=1),
                unit(91, 11, 12, 'gatling', level=None),
                unit(92, 10, 10, 'worker'),
            ),
            weapon_shop=SUMMON_SHOP,
        )
        names = frozenset(item.name for item in SUMMON_SHOP)
        self.assertEqual(
            evaluate_collapse_summons(
                unknown_station, OpponentMemory(), available_names=names
            ),
            (),
        )
        self.assertEqual(
            evaluate_collapse_summons(
                unknown_weapon, OpponentMemory(), available_names=names
            ),
            (),
        )

    def test_no_collapse_evidence_fails_closed(self):
        observed = observation(
            round_no=131,
            enemy_units=enemy_snapshot(
                station_health=1000, weapons=3, roles=3, walls=4
            ),
            weapon_shop=SUMMON_SHOP,
        )
        self.assertEqual(
            evaluate_collapse_summons(
                observed, OpponentMemory(),
                available_names=frozenset(item.name for item in SUMMON_SHOP),
            ),
            (),
        )

    def test_previous_night_sustained_damage_can_prove_advance(self):
        observed = observation(
            round_no=131,
            enemy_units=enemy_snapshot(station_health=1000),
            weapon_shop=SUMMON_SHOP,
        )

        plans = evaluate_collapse_summons(
            observed, sustained_memory(),
            available_names=frozenset(item.name for item in SUMMON_SHOP),
        )

        self.assertTrue(plans)
        self.assertIn('previous_night_sustained_damage', plans[0].evidence)
        self.assertGreaterEqual(plans[0].advance_rounds, 3)


class CollapseJobTests(unittest.TestCase):
    def _observed(self, *, gold=250, backpack=(), complete=True):
        own_base = observation(
            round_no=131,
            our_units=(
                unit(1, 5, 5, 'station', health=1500, level=1),
                unit(2, 2, 2, 'worker', backpack=backpack),
                unit(3, 3, 2, 'worker'),
                unit(4, 4, 2, 'pioneer'),
            ),
            enemy_units=enemy_snapshot(),
            gold=gold,
            zones=(Zone(Position(1, 1), 'weaponShop'),),
            weapon_shop=SUMMON_SHOP,
        )
        initial = WorldGrid.from_observation(own_base)
        plan = BuildPlan(wall_site_limit=12)
        layout = build_defensive_layout(initial, plan)
        defenses = ()
        if complete:
            defenses = tuple(
                unit(
                    10 + index, site.position.x, site.position.y,
                    site.weapon_type, health=1000, level=1,
                )
                for index, site in enumerate(layout.weapon_sites)
            ) + tuple(
                unit(30 + index, pos.x, pos.y, 'wall', health=1000, level=1)
                for index, pos in enumerate(layout.critical_wall_sites)
            )
        observed = observation(
            round_no=131,
            our_units=own_base.our.units + defenses,
            enemy_units=own_base.enemy.units,
            gold=gold,
            zones=own_base.zones,
            weapon_shop=SUMMON_SHOP,
        )
        world = WorldGrid.from_observation(observed)
        return observed, world, build_defensive_layout(world, plan), plan

    def _intent(self, plan, reserve=100):
        return StrategicIntent(
            build_plan=plan,
            feature_flags=RuleFeatureFlags(enable_collapse_summons=True),
            gold_reserve=reserve,
        )

    def test_core_survival_build_must_finish_first(self):
        observed, world, layout, plan = self._observed(complete=False)
        jobs = generate_collapse_summon_jobs(
            observed, world, layout, self._intent(plan), OpponentMemory()
        )
        self.assertTrue(all(not values for values in jobs.values()))

    def test_purchase_uses_only_gold_above_dynamic_reserve(self):
        observed, world, layout, plan = self._observed(gold=199)
        jobs = generate_collapse_summon_jobs(
            observed, world, layout, self._intent(plan), OpponentMemory()
        )
        self.assertTrue(all(not values for values in jobs.values()))

        rich, rich_world, rich_layout, rich_plan = self._observed(gold=200)
        rich_jobs = generate_collapse_summon_jobs(
            rich, rich_world, rich_layout, self._intent(rich_plan), OpponentMemory()
        )
        job = next(job for values in rich_jobs.values() for job in values)
        self.assertEqual(job.name, 'LargeRobotSummonOrder')
        self.assertLess(job.priority, self._intent(rich_plan).day_priorities.purchase)

    def test_held_qualifying_order_is_used_without_new_purchase(self):
        observed, world, layout, plan = self._observed(
            gold=100, backpack=('LargeRobotSummonOrder',)
        )
        jobs = generate_collapse_summon_jobs(
            observed, world, layout, self._intent(plan), OpponentMemory()
        )
        job = next(job for values in jobs.values() for job in values)
        self.assertEqual(job.kind.value, 'use_item')
        self.assertEqual(job.name, 'LargeRobotSummonOrder')

    def test_feature_flag_can_disable_all_summon_jobs(self):
        observed, world, layout, plan = self._observed()
        jobs = generate_collapse_summon_jobs(
            observed, world, layout, StrategicIntent(build_plan=plan), OpponentMemory()
        )
        self.assertTrue(all(not values for values in jobs.values()))


if __name__ == '__main__':
    unittest.main()
