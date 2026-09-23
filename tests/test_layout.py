import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.layout import build_defensive_layout
from future_war_agent.strategy.policy import BuildPlan
from future_war_agent.strategy.world import WorldGrid
from tests.strategy_helpers import observation, robot, unit


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

    def test_wall_ring_has_one_temporary_gate_and_no_weapon_overlap(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, "station", level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertIsNotNone(layout.entrance)
        self.assertEqual(layout.wall_sites[-1], layout.entrance)
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

    def test_entrance_is_on_rear_side_and_is_planned_last(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, 'station', level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertEqual(layout.entrance, Position(3, 3))
        self.assertEqual(layout.wall_sites[-1], layout.entrance)

    def test_visible_east_attack_reorders_critical_walls(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                our_units=(unit(10013, 5, 5, 'station', level=1),),
                robots=(robot(1, 12, 7), robot(2, 11, 8)),
            )
        )

        layout = build_defensive_layout(world)

        self.assertEqual(
            layout.critical_wall_sites,
            (
                Position(6, 3),
                Position(7, 3),
                Position(8, 4),
                Position(8, 5),
                Position(8, 6),
                Position(8, 7),
                Position(8, 8),
                Position(7, 8),
                Position(6, 8),
            ),
        )

    def test_recent_east_attack_reorders_daytime_rebuild_sites(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, 'station', level=1),))
        )

        layout = build_defensive_layout(
            world,
            recent_threat_positions=(Position(12, 7), Position(11, 8)),
        )

        self.assertEqual(
            layout.critical_wall_sites,
            (
                Position(6, 3),
                Position(7, 3),
                Position(8, 4),
                Position(8, 5),
                Position(8, 6),
                Position(8, 7),
                Position(8, 8),
                Position(7, 8),
                Position(6, 8),
            ),
        )

    def test_daily_anchor_overrides_transient_current_threat_direction(self) -> None:
        world = WorldGrid.from_observation(
            observation(
                our_units=(unit(10013, 5, 5, 'station', level=1),),
                robots=(robot(1, 1, 7), robot(2, 2, 8)),
            )
        )

        layout = build_defensive_layout(
            world,
            recent_threat_positions=(Position(12, 7), Position(11, 8)),
        )

        self.assertEqual(
            layout.critical_wall_sites,
            (
                Position(6, 3),
                Position(7, 3),
                Position(8, 4),
                Position(8, 5),
                Position(8, 6),
                Position(8, 7),
                Position(8, 8),
                Position(7, 8),
                Position(6, 8),
            ),
        )

    def test_corner_bases_put_five_critical_walls_toward_map_center(
        self,
    ) -> None:
        cases = (
            (2, 2, 5),
            (2, 11, 5),
            (11, 2, 9),
            (11, 11, 9),
        )
        for station_x, station_y, front_x in cases:
            with self.subTest(station=(station_x, station_y)):
                world = WorldGrid.from_observation(
                    observation(
                        our_units=(
                            unit(
                                10013,
                                station_x,
                                station_y,
                                'station',
                                level=1,
                            ),
                        ),
                    )
                )

                layout = build_defensive_layout(world)

                self.assertEqual(len(layout.critical_wall_sites), 9)
                self.assertEqual(
                    len(set(layout.critical_wall_sites)),
                    9,
                )
                self.assertEqual(
                    sum(
                        position.x == front_x
                        for position in layout.critical_wall_sites
                    ),
                    5,
                )
                self.assertNotIn(
                    layout.entrance,
                    layout.critical_wall_sites,
                )
                self.assertTrue(all(
                    world.is_wall_build_site(position)
                    for position in layout.critical_wall_sites
                ))

    def test_every_planned_weapon_keeps_a_distinct_controller_site(self) -> None:
        world = WorldGrid.from_observation(
            observation(our_units=(unit(10013, 5, 5, 'station', level=1),))
        )

        layout = build_defensive_layout(world)

        self.assertEqual(len(layout.controller_sites), len(layout.weapon_sites))
        self.assertEqual(len(set(layout.controller_sites)), len(layout.controller_sites))
        self.assertTrue(set(layout.controller_sites).isdisjoint(layout.wall_sites))
        self.assertNotIn(layout.entrance, layout.controller_sites)
        for weapon_site, controller_site in zip(
            layout.weapon_sites,
            layout.controller_sites,
        ):
            self.assertEqual(
                weapon_site.position.chebyshev_distance(controller_site),
                1,
            )


if __name__ == "__main__":
    unittest.main()
