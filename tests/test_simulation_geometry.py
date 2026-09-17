import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.geometry import (
    bresenham_cells,
    center_intersection_cells,
    is_legal_cone,
    supercover_cells,
)


class SimulationGeometryTests(unittest.TestCase):
    def test_horizontal_and_vertical_rays_exclude_start_and_include_end(self) -> None:
        functions = (
            bresenham_cells,
            supercover_cells,
            center_intersection_cells,
        )
        for function in functions:
            with self.subTest(function=function.__name__, direction="horizontal"):
                self.assertEqual(
                    function(Position(0, 0), Position(3, 0)),
                    (Position(1, 0), Position(2, 0), Position(3, 0)),
                )
            with self.subTest(function=function.__name__, direction="vertical"):
                self.assertEqual(
                    function(Position(2, 3), Position(2, 0)),
                    (Position(2, 2), Position(2, 1), Position(2, 0)),
                )

    def test_bresenham_handles_diagonal_shallow_and_steep_slopes(self) -> None:
        cases = (
            (
                Position(0, 0),
                Position(3, 3),
                (Position(1, 1), Position(2, 2), Position(3, 3)),
            ),
            (
                Position(0, 0),
                Position(4, 2),
                (
                    Position(1, 1),
                    Position(2, 1),
                    Position(3, 2),
                    Position(4, 2),
                ),
            ),
            (
                Position(0, 0),
                Position(2, 4),
                (
                    Position(1, 1),
                    Position(1, 2),
                    Position(2, 3),
                    Position(2, 4),
                ),
            ),
        )
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(bresenham_cells(start, end), expected)

    def test_supercover_includes_corner_touch_cells_in_traversal_order(self) -> None:
        self.assertEqual(
            supercover_cells(Position(0, 0), Position(3, 1)),
            (
                Position(1, 0),
                Position(1, 1),
                Position(2, 0),
                Position(2, 1),
                Position(3, 1),
            ),
        )

    def test_center_intersection_excludes_corner_only_contacts(self) -> None:
        self.assertEqual(
            center_intersection_cells(Position(0, 0), Position(3, 1)),
            (Position(1, 0), Position(2, 1), Position(3, 1)),
        )

    def test_all_rays_are_stable_without_duplicate_cells(self) -> None:
        start = Position(6, 6)
        end = Position(1, 3)
        for function in (
            bresenham_cells,
            supercover_cells,
            center_intersection_cells,
        ):
            ray = function(start, end)
            with self.subTest(function=function.__name__):
                self.assertNotIn(start, ray)
                self.assertEqual(ray[-1], end)
                self.assertEqual(len(ray), len(set(ray)))

    def test_cone_accepts_ninety_degrees_and_rejects_more(self) -> None:
        origin = Position(5, 5)

        self.assertTrue(
            is_legal_cone(origin, (Position(8, 5), Position(5, 8)))
        )
        self.assertFalse(
            is_legal_cone(origin, (Position(8, 5), Position(4, 8)))
        )

    def test_cone_rejects_origin_as_a_target(self) -> None:
        self.assertFalse(is_legal_cone(Position(2, 2), (Position(2, 2),)))


if __name__ == "__main__":
    unittest.main()
