import unittest

from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.strategy.jobs import JobKind, generate_day_jobs
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.policy import (
    DayPriorities,
    ItemPolicy,
    StrategicIntent,
)
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class DayJobTests(unittest.TestCase):
    def jobs(self, observed):
        world = WorldGrid.from_observation(observed)
        return generate_day_jobs(observed, world, build_defensive_layout(world))

    def test_twilight_recall_outranks_ordinary_jobs(self) -> None:
        observed = observation(
            round_no=70,
            our_units=(
                unit(10010, 1, 1, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
            ),
            zones=(Zone(Position(2, 2), "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.RECALL)

    def test_affordable_missing_weapon_is_a_build_job(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, "worker"),
                unit(10013, 5, 5, "station", level=1),
            ),
            gold=25,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.BUILD_WEAPON, {job.kind for job in jobs[10010]})

    def test_full_backpack_generates_unload_job(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    10010,
                    2,
                    2,
                    "worker",
                    backpack_capacity=2,
                    backpack=("iron", "iron"),
                ),
            ),
            zones=(Zone(Position(5, 5), "vendor"),),
            vendor_shop=(ShopItem("iron", 4),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].kind, JobKind.SELL)
        self.assertEqual(jobs[10010][0].quantity, 2)

    def test_low_gold_with_minerals_generates_proactive_sale(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker", backpack=("copper",)),
                unit(10013, 5, 5, "station", level=1),
            ),
            zones=(Zone(Position(5, 2), "vendor"),),
            vendor_shop=(ShopItem("copper", 8),),
            gold=10,
        )

        jobs = self.jobs(observed)

        self.assertIn(JobKind.SELL, {job.kind for job in jobs[10010]})

    def test_stone_is_preferred_while_wall_is_missing(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 2, 2, "worker"),
                unit(10013, 7, 7, "station", level=1),
                unit(10020, 6, 6, "gatling", level=1),
                unit(10030, 7, 6, "railgun", level=1),
                unit(10040, 8, 6, "rocket", level=1),
            ),
            zones=(
                Zone(Position(3, 2), "stone"),
                Zone(Position(2, 3), "copper"),
            ),
            vendor_shop=(ShopItem("stone", 1), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        mining = next(job for job in jobs[10010] if job.kind is JobKind.COLLECT)
        self.assertEqual(mining.name, "stone")

    def test_price_per_distance_selects_mineral_after_defense(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, "worker"),),
            zones=(
                Zone(Position(3, 1), "iron"),
                Zone(Position(6, 1), "copper"),
            ),
            vendor_shop=(ShopItem("iron", 2), ShopItem("copper", 20)),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].name, "copper")

    def test_two_workers_may_receive_same_adjacent_mine(self) -> None:
        mine = Position(4, 4)
        observed = observation(
            our_units=(
                unit(10010, 3, 4, "worker"),
                unit(10012, 4, 3, "worker"),
            ),
            zones=(Zone(mine, "stone"),),
        )

        jobs = self.jobs(observed)

        self.assertEqual(jobs[10010][0].target, mine)
        self.assertEqual(jobs[10012][0].target, mine)

    def test_gold_reserve_prevents_weapon_build_job(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            StrategicIntent(gold_reserve=25),
        )

        self.assertNotIn(JobKind.BUILD_WEAPON, {job.kind for job in jobs[10010]})

    def test_intent_priorities_flow_into_generated_jobs(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 3, 3, 'worker'),
                unit(10013, 5, 5, 'station', level=1),
            ),
            gold=25,
        )
        world = WorldGrid.from_observation(observed)

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            StrategicIntent(day_priorities=DayPriorities(build_weapon=777)),
        )

        weapon_job = next(
            job for job in jobs[10010] if job.kind is JobKind.BUILD_WEAPON
        )
        self.assertEqual(weapon_job.priority, 777)

    def test_medicine_purchase_respects_stock_and_reserve(self) -> None:
        observed = observation(
            our_units=(unit(10010, 1, 1, 'worker'),),
            zones=(Zone(Position(2, 2), 'weaponShop'),),
            weapon_shop=(ShopItem('Medicine', 10),),
            gold=40,
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            gold_reserve=25,
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            intent,
        )

        purchase = next(job for job in jobs[10010] if job.kind is JobKind.BUY)
        self.assertEqual((purchase.name, purchase.quantity), ('Medicine', 1))

    def test_medicine_stock_target_caps_concurrent_purchase_jobs(self) -> None:
        observed = observation(
            our_units=(
                unit(10010, 1, 2, 'worker'),
                unit(10011, 2, 1, 'worker'),
            ),
            zones=(Zone(Position(2, 2), 'weaponShop'),),
            weapon_shop=(ShopItem('Medicine', 10),),
            gold=40,
        )
        world = WorldGrid.from_observation(observed)
        intent = StrategicIntent(
            item_policy=ItemPolicy(
                medicine_health_threshold=60,
                medicine_stock=1,
            ),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            intent,
        )

        purchases = [
            job
            for role_jobs in jobs.values()
            for job in role_jobs
            if job.kind is JobKind.BUY and job.name == 'Medicine'
        ]
        self.assertEqual(len(purchases), 1)


if __name__ == "__main__":
    unittest.main()
