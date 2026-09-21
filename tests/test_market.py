import unittest

from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import ShopItem, WorldNews, Zone
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.jobs import JobKind, generate_day_jobs
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.market import (
    MarketMemory,
    MarketView,
    PriceDirection,
    parse_market_signals,
)
from future_war_agent.strategy.world import WorldGrid
from future_war_agent.strategy.session import SessionStore
from tests.strategy_helpers import observation, unit
from future_war_agent.protocol.models import Position


class MarketMemoryTests(unittest.TestCase):
    def test_explicit_supply_news_predicts_next_day(self) -> None:
        signals = parse_market_signals(
            'Iron mine collapse will disrupt supply for 2 days.',
            current_day=1,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].mineral, 'iron')
        self.assertIs(signals[0].direction, PriceDirection.UP)
        self.assertEqual((signals[0].start_day, signals[0].end_day), (2, 3))

    def test_price_ground_truth_updates_rule_reliability(self) -> None:
        memory = MarketMemory()
        first = observation(
            round_no=1,
            vendor_shop=(ShopItem('iron', 10),),
            world_news=WorldNews(
                official_news='Iron shortage will last 2 days.',
            ),
        )
        state = memory.observe(first)

        self.assertIn('iron', memory.view(first, state).hold_items)

        second = observation(
            round_no=131,
            vendor_shop=(ShopItem('iron', 15),),
            world_news=first.world_news,
        )
        updated = memory.observe(second, state)

        self.assertEqual(len(updated.signals), 1)
        self.assertTrue(updated.signals[0].settled)
        self.assertEqual(updated.reliability[0].correct, 1)
        self.assertEqual(updated.reliability[0].wrong, 0)
        self.assertEqual(len(updated.processed_news), 1)

    def test_wrong_prediction_lowers_rule_reliability(self) -> None:
        memory = MarketMemory()
        first = observation(
            round_no=1,
            vendor_shop=(ShopItem('copper', 10),),
            world_news=WorldNews(
                official_news='Copper shortage will last 1 day.',
            ),
        )
        state = memory.observe(first)
        second = observation(
            round_no=131,
            vendor_shop=(ShopItem('copper', 5),),
        )

        updated = memory.observe(second, state)

        self.assertEqual(updated.reliability[0].correct, 0)
        self.assertEqual(updated.reliability[0].wrong, 1)

    def test_unknown_and_malformed_news_stays_unknown(self) -> None:
        samples = (
            '',
            '\x00',
            'market rumors',
            'iron mine collapse',
            '铜矿也许会变化三天',
            '🔥' * 1024,
        )

        for sample in samples:
            with self.subTest(sample=sample[:20]):
                self.assertEqual(
                    parse_market_signals(sample, current_day=1),
                    (),
                )

    def test_hold_signal_keeps_iron_and_sells_other_material(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    1,
                    1,
                    1,
                    'worker',
                    backpack_capacity=2,
                    backpack=('iron', 'copper'),
                ),
            ),
            zones=(Zone(Position(2, 1), 'vendor'),),
            vendor_shop=(ShopItem('iron', 10), ShopItem('copper', 10)),
        )
        world = WorldGrid.from_observation(observed)
        view = MarketView(
            next_day_directions=(
                ('iron', PriceDirection.UP),
                ('copper', PriceDirection.DOWN),
            ),
            hold_items=frozenset({'iron'}),
            sell_items=frozenset({'copper'}),
        )

        jobs = generate_day_jobs(
            observed,
            world,
            build_defensive_layout(world),
            market_view=view,
        )

        sale = next(job for job in jobs[1] if job.kind is JobKind.SELL)
        self.assertEqual(sale.name, 'copper')

    def test_engine_retains_market_learning_across_round_gap(self) -> None:
        sessions = SessionStore()
        engine = StrategyEngine(
            phase2_planner=lambda *_args, **_kwargs: Decision(),
            session_store=sessions,
        )
        units = (unit(10, 5, 5, 'station', health=1000, level=1),)
        first = observation(
            round_no=1,
            our_units=units,
            vendor_shop=(ShopItem('iron', 10),),
            world_news=WorldNews(
                official_news='Iron shortage will last 2 days.',
            ),
        )
        second = observation(
            round_no=131,
            our_units=units,
            vendor_shop=(ShopItem('iron', 15),),
        )

        engine.plan(first)
        engine.plan(second)
        retained = sessions.get('team').market_state

        self.assertEqual(len(retained.signals), 1)
        self.assertEqual(retained.reliability[0].correct, 1)


if __name__ == '__main__':
    unittest.main()
