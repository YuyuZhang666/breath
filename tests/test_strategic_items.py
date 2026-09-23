import unittest

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.joint import candidates_for_jobs
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.policy import (
    BuildPlan,
    RuleFeatureFlags,
    StrategicIntent,
    StrategyProfile,
)
from future_war_agent.strategy.strategic_items import (
    apply_emergency_combat_items,
    generate_strategic_item_jobs,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


class StrategicItemTests(unittest.TestCase):
    def _defended(
        self,
        *,
        round_no=55,
        station_health=900,
        worker_backpack=(),
        gold=225,
        shop=(),
        damaged_wall=False,
        damaged_weapon=False,
        worker_position=(2, 2),
    ):
        base = observation(
            round_no=round_no,
            our_units=(
                unit(1, 5, 5, 'station', health=station_health, level=1),
                unit(
                    2,
                    worker_position[0],
                    worker_position[1],
                    'worker',
                    backpack=worker_backpack,
                ),
                unit(3, 3, 2, 'worker'),
                unit(4, 4, 2, 'pioneer'),
            ),
            gold=gold,
            weapon_shop=shop,
            zones=(Zone(Position(1, 1), 'weaponShop'),),
        )
        initial_world = WorldGrid.from_observation(base)
        layout = build_defensive_layout(
            initial_world,
            BuildPlan(wall_site_limit=12),
        )
        weapons = tuple(
            unit(
                10 + index,
                site.position.x,
                site.position.y,
                site.weapon_type,
                health=(400 if damaged_weapon and index == 0 else 1000),
                level=1,
            )
            for index, site in enumerate(layout.weapon_sites)
        )
        walls = tuple(
            unit(
                30 + index,
                position.x,
                position.y,
                'wall',
                health=(300 if damaged_wall and index == 0 else 1000),
                level=1,
            )
            for index, position in enumerate(layout.critical_wall_sites)
        )
        observed = observation(
            round_no=round_no,
            our_units=base.our.units + weapons + walls,
            gold=gold,
            weapon_shop=shop,
            zones=base.zones,
        )
        world = WorldGrid.from_observation(observed)
        return observed, world, build_defensive_layout(
            world,
            BuildPlan(wall_site_limit=12),
        )

    def test_upgrade_jobs_wait_for_core_defense(self):
        observed = observation(
            round_no=55,
            our_units=(
                unit(1, 5, 5, 'station', health=500, level=1),
                unit(2, 2, 2, 'worker'),
            ),
            gold=300,
            weapon_shop=(ShopItem('StationUpgradeVoucher1', 100),),
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            feature_flags=RuleFeatureFlags(enable_upgrades=True),
        )

        jobs = generate_strategic_item_jobs(
            observed, world, build_defensive_layout(world), intent
        )

        self.assertTrue(all(not role_jobs for role_jobs in jobs.values()))

    def test_station_upgrade_purchase_respects_dynamic_gold_reserve(self):
        shop = (ShopItem('StationUpgradeVoucher1', 100),)
        observed, world, layout = self._defended(shop=shop)
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            build_plan=BuildPlan(wall_site_limit=12),
            feature_flags=RuleFeatureFlags(enable_upgrades=True),
            gold_reserve=100,
        )

        jobs = generate_strategic_item_jobs(observed, world, layout, intent)

        purchases = [
            job for role_jobs in jobs.values() for job in role_jobs
            if job.kind.value == 'buy'
        ]
        self.assertEqual(len(purchases), 1)
        self.assertEqual(purchases[0].name, 'StationUpgradeVoucher1')

        poor, poor_world, poor_layout = self._defended(
            gold=199, shop=shop
        )
        poor_jobs = generate_strategic_item_jobs(
            poor, poor_world, poor_layout, intent
        )
        self.assertTrue(all(not role_jobs for role_jobs in poor_jobs.values()))

    def test_held_upgrade_is_targeted_and_legal(self):
        observed, world, layout = self._defended(
            worker_backpack=('StationUpgradeVoucher1',),
            worker_position=(4, 5),
        )
        holder = world.unit_by_id(2)
        self.assertIsNotNone(holder)
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            build_plan=BuildPlan(wall_site_limit=12),
            feature_flags=RuleFeatureFlags(enable_upgrades=True),
        )
        jobs = generate_strategic_item_jobs(observed, world, layout, intent)
        job = jobs[2][0]
        self.assertTrue(job.targeted)
        self.assertLess(job.priority, intent.day_priorities.build_wall)

        candidates = candidates_for_jobs(observed, world, holder, jobs[2])
        direct = next(
            candidate.action
            for candidate in candidates
            if candidate.action is not None
            and candidate.action.kind is ActionKind.USE
        )
        action = Action.use('StationUpgradeVoucher1', Position(5, 5))
        self.assertEqual(direct, action)
        validated = validate_decision(
            observed, Decision(commands={2: action})
        )
        self.assertEqual(validated.commands[2], action)

    def test_damaged_weapon_gets_upgrade_but_full_weapon_does_not(self):
        shop = (ShopItem('WeaponUpgradeVoucher1', 100),)
        observed, world, layout = self._defended(
            station_health=1500,
            damaged_weapon=True,
            shop=shop,
        )
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            build_plan=BuildPlan(wall_site_limit=12),
            feature_flags=RuleFeatureFlags(enable_upgrades=True),
            gold_reserve=100,
        )
        jobs = generate_strategic_item_jobs(observed, world, layout, intent)
        purchase = next(job for values in jobs.values() for job in values)
        self.assertEqual(purchase.name, 'WeaponUpgradeVoucher1')

        full, full_world, full_layout = self._defended(
            station_health=1500, shop=shop
        )
        full_jobs = generate_strategic_item_jobs(
            full, full_world, full_layout, intent
        )
        self.assertTrue(all(not values for values in full_jobs.values()))

    def test_repair_only_targets_damaged_critical_wall(self):
        observed, world, layout = self._defended(
            worker_backpack=('WallFixer',),
            damaged_wall=True,
        )
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            build_plan=BuildPlan(wall_site_limit=12),
            feature_flags=RuleFeatureFlags(enable_repairs=True),
        )

        jobs = generate_strategic_item_jobs(observed, world, layout, intent)

        repair = next(job for values in jobs.values() for job in values)
        self.assertEqual(repair.name, 'WallFixer')
        self.assertIn(repair.target, layout.critical_wall_sites)

        full, full_world, full_layout = self._defended(
            worker_backpack=('WallFixer',),
        )
        full_jobs = generate_strategic_item_jobs(
            full, full_world, full_layout, intent
        )
        self.assertTrue(all(not values for values in full_jobs.values()))

    def test_stun_uses_idle_role_without_replacing_attack(self):
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 5, 5, 'station', health=1000, level=1),
                unit(2, 4, 4, 'worker', backpack=('DizzyWeapon',)),
                unit(3, 4, 5, 'pioneer'),
                unit(9, 6, 5, 'gatling', attack_range=5, level=1),
            ),
            robots=(robot(20, 8, 5, role_type='bossRobot'),),
        )
        attack = Action.attack(3, (Position(8, 5),))
        baseline = Decision(commands={9: attack})
        intent = StrategicIntent(
            feature_flags=RuleFeatureFlags(enable_stun=True),
        )

        result = apply_emergency_combat_items(observed, baseline, intent)

        self.assertEqual(result.commands[9], attack)
        self.assertEqual(result.commands[2].name, 'DizzyWeapon')
        self.assertEqual(result.commands[2].target_positions, (Position(8, 5),))

    def test_bomb_never_displaces_controller_attack(self):
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 5, 5, 'station', health=1000, level=1),
                unit(2, 4, 4, 'worker', backpack=('Bomb',)),
                unit(9, 6, 5, 'gatling', attack_range=5, level=1),
            ),
            robots=(robot(20, 8, 5, health=40),),
        )
        baseline = Decision(
            commands={9: Action.attack(2, (Position(8, 5),))}
        )
        intent = StrategicIntent(
            feature_flags=RuleFeatureFlags(enable_bomb=True),
        )

        self.assertEqual(
            apply_emergency_combat_items(observed, baseline, intent),
            baseline,
        )

    def test_idle_role_uses_bomb_for_immediate_kill(self):
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 5, 5, 'station', health=1000, level=1),
                unit(2, 4, 4, 'worker', backpack=('Bomb',)),
            ),
            robots=(robot(20, 8, 5, health=40),),
        )
        result = apply_emergency_combat_items(
            observed,
            Decision(),
            StrategicIntent(
                feature_flags=RuleFeatureFlags(enable_bomb=True),
            ),
        )
        self.assertEqual(result.commands[2].name, 'Bomb')

    def test_high_risk_stock_prefers_stun_then_bomb(self):
        shop = (
            ShopItem('DizzyWeapon', 100),
            ShopItem('Bomb', 100),
        )
        observed, world, layout = self._defended(
            station_health=1500, shop=shop, gold=250
        )
        intent = StrategicIntent(
            profile=StrategyProfile.SURVIVE,
            build_plan=BuildPlan(wall_site_limit=12),
            feature_flags=RuleFeatureFlags(
                enable_stun=True,
                enable_bomb=True,
            ),
            gold_reserve=100,
        )
        jobs = generate_strategic_item_jobs(observed, world, layout, intent)
        first = next(job for values in jobs.values() for job in values)
        self.assertEqual(first.name, 'DizzyWeapon')
        self.assertLess(first.priority, intent.day_priorities.purchase)

        stocked, stocked_world, stocked_layout = self._defended(
            station_health=1500,
            shop=shop,
            gold=250,
            worker_backpack=('DizzyWeapon',),
        )
        stocked_jobs = generate_strategic_item_jobs(
            stocked, stocked_world, stocked_layout, intent
        )
        second = next(job for values in stocked_jobs.values() for job in values)
        self.assertEqual(second.name, 'Bomb')

    def test_three_busy_controllers_prevent_emergency_item_use(self):
        target = Position(8, 5)
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 5, 5, 'station', health=1000, level=1),
                unit(2, 4, 4, 'worker', backpack=('DizzyWeapon',)),
                unit(3, 4, 5, 'worker', backpack=('Bomb',)),
                unit(4, 5, 4, 'pioneer'),
                unit(10, 6, 4, 'gatling', attack_range=5, level=1),
                unit(11, 6, 5, 'railgun', attack_range=6, level=1),
                unit(12, 6, 6, 'rocket', attack_range=10, level=1),
            ),
            robots=(robot(20, target.x, target.y, health=40),),
        )
        baseline = Decision(commands={
            10: Action.attack(2, (target,)),
            11: Action.attack(3, (target,)),
            12: Action.attack(4, (target,)),
        })
        intent = StrategicIntent(
            feature_flags=RuleFeatureFlags(
                enable_stun=True, enable_bomb=True
            ),
        )
        self.assertEqual(
            apply_emergency_combat_items(observed, baseline, intent),
            baseline,
        )

    def test_targeted_emergency_item_requires_target_in_validator(self):
        observed = observation(
            round_no=71,
            our_units=(
                unit(1, 5, 5, 'station', level=1),
                unit(2, 4, 4, 'worker', backpack=('Bomb',)),
            ),
        )
        rejected = validate_decision(
            observed, Decision(commands={2: Action.use('Bomb')})
        )
        accepted = validate_decision(
            observed,
            Decision(commands={2: Action.use('Bomb', Position(8, 8))}),
        )
        self.assertNotIn(2, rejected.commands)
        self.assertIn(2, accepted.commands)


if __name__ == '__main__':
    unittest.main()
