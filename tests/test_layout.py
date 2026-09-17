import unittest

from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.policy import BuildPlan
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, unit


class DefensiveLayoutTests(unittest.TestCase):
    def test_selects_three_distinct_weapon_sites_in_loadout_order(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, "station", level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertEqual(
            tuple(site.weapon_type for site in layout.weapon_sites),
            ("gatling", "railgun", "rocket"),
        )
        self.assertEqual(len({site.position for site in layout.weapon_sites}), 3)

    def test_wall_ring_has_one_entrance_and_no_weapon_overlap(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, "station", level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertIsNotNone(layout.entrance)
        self.assertNotIn(layout.entrance, layout.wall_sites)
        self.assertTrue(
            set(layout.wall_sites).isdisjoint(
                site.position for site in layout.weapon_sites
            )
        )

    def test_layout_is_clipped_at_map_edge(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                width=8,
                height=8,
                our_units=(unit(10013, 0, 0, "station", level=1),),
            )
        )

        layout = build_defensive_layout(world)

        positions = (
            tuple(site.position for site in layout.weapon_sites)
            + layout.wall_sites
        )
        self.assertTrue(all(world.in_bounds(position) for position in positions))

    def test_missing_station_returns_empty_layout(self) -> None:
        layout = build_defensive_layout(WorldGrid.from_observation(observation()))

        self.assertEqual(layout.weapon_sites, ())
        self.assertEqual(layout.wall_sites, ())
        self.assertIsNone(layout.entrance)

    def test_custom_build_plan_controls_loadout_and_wall_cap(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, 'station', level=1),))
        )

        layout = build_defensive_layout(
            world,
            BuildPlan(
                weapon_loadout=('rocket', 'gatling'),
                wall_site_limit=2,
            ),
        )

        self.assertEqual(
            tuple(site.weapon_type for site in layout.weapon_sites),
            ('rocket', 'gatling'),
        )
        self.assertEqual(len(layout.wall_sites), 2)


if __name__ == "__main__":
    unittest.main()
