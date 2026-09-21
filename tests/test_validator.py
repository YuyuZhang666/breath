import json
import unittest
from dataclasses import replace
from pathlib import Path

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.decision.validator import validate_decision
from future_war_agent.protocol.models import Position, ShopItem, Zone
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime


FIXTURE = Path(__file__).parent / "fixtures" / "request.json"


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observed = parse_observation(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def test_keeps_valid_night_attack(self) -> None:
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(6, 20),)),
            }
        )

        validated = validate_decision(self.observed, decision)

        self.assertIn(10020, validated.commands)

    def test_drops_day_attack_but_keeps_valid_move(self) -> None:
        day = replace(self.observed, time=TurnTime.from_round(1))
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(4, 4),)),
                10010: Action.move(Position(5, 24)),
            }
        )

        validated = validate_decision(day, decision)

        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_drops_out_of_bounds_target(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(41, 0))})

        self.assertEqual(validate_decision(self.observed, decision).commands, {})

    def test_drops_attack_when_controller_has_personal_action(self) -> None:
        decision = Decision(
            commands={
                10020: Action.attack(10010, (Position(6, 20),)),
                10010: Action.move(Position(5, 24)),
            }
        )

        validated = validate_decision(self.observed, decision)

        self.assertNotIn(10020, validated.commands)
        self.assertIn(10010, validated.commands)

    def test_upgraded_gatling_requires_exact_level_sized_cone(self) -> None:
        weapon = next(
            unit for unit in self.observed.our.units if unit.unit_id == 10020
        )
        observed = replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=tuple(
                    replace(unit, level=2)
                    if unit.unit_id == weapon.unit_id
                    else unit
                    for unit in self.observed.our.units
                ),
            ),
        )
        legal = Decision(
            commands={
                10020: Action.attack(
                    10010,
                    (Position(6, 20), Position(7, 20)),
                )
            }
        )
        undersized = Decision(
            commands={10020: Action.attack(10010, (Position(6, 20),))}
        )

        self.assertIn(10020, validate_decision(observed, legal).commands)
        self.assertNotIn(10020, validate_decision(observed, undersized).commands)

    def test_upgraded_rocket_uses_level_targets_and_railgun_stays_single(self) -> None:
        weapon = next(
            unit for unit in self.observed.our.units if unit.unit_id == 10020
        )
        cases = (
            (
                replace(weapon, role_type='rocket', level=3),
                (Position(5, 20), Position(6, 20), Position(7, 20)),
                True,
            ),
            (
                replace(weapon, role_type='rocket', level=3),
                (Position(6, 20),),
                False,
            ),
            (
                replace(weapon, role_type='railgun', level=3),
                (Position(6, 20),),
                True,
            ),
            (
                replace(weapon, role_type='railgun', level=3),
                (Position(6, 20), Position(7, 20)),
                False,
            ),
        )
        for changed, targets, expected in cases:
            with self.subTest(role_type=changed.role_type, targets=targets):
                observed = replace(
                    self.observed,
                    our=replace(
                        self.observed.our,
                        units=tuple(
                            changed if unit.unit_id == weapon.unit_id else unit
                            for unit in self.observed.our.units
                        ),
                    ),
                )
                retained = 10020 in validate_decision(
                    observed,
                    Decision(commands={10020: Action.attack(10010, targets)}),
                ).commands
                self.assertEqual(retained, expected)

    def test_drops_attack_on_cooldown_or_outside_live_range(self) -> None:
        weapon = next(
            unit for unit in self.observed.our.units if unit.unit_id == 10020
        )
        decision = Decision(
            commands={10020: Action.attack(10010, (Position(6, 20),))}
        )
        for changed in (
            replace(weapon, cooldown=1),
            replace(weapon, attack_range=2),
        ):
            with self.subTest(changed=changed):
                observed = replace(
                    self.observed,
                    our=replace(
                        self.observed.our,
                        units=tuple(
                            changed if unit.unit_id == weapon.unit_id else unit
                            for unit in self.observed.our.units
                        ),
                    ),
                )
                self.assertNotIn(
                    10020,
                    validate_decision(observed, decision).commands,
                )

    def test_drops_structurally_incomplete_action(self) -> None:
        decision = Decision(commands={10010: Action.move(Position(5, 24))})
        malformed = replace(decision.commands[10010], target_positions=())

        validated = validate_decision(
            self.observed,
            Decision(commands={10010: malformed}),
        )

        self.assertEqual(validated.commands, {})

    def test_treasure_requires_items_in_pioneer_backpack(self) -> None:
        source = next(
            unit for unit in self.observed.our.units if unit.role_type == 'worker'
        )
        pioneer = replace(source, role_type='pioneer', backpack=())
        observed = replace(
            self.observed,
            our=replace(
                self.observed.our,
                units=tuple(
                    pioneer if unit.unit_id == source.unit_id else unit
                    for unit in self.observed.our.units
                ),
            ),
        )
        nearby_target = Position(pioneer.position.x + 1, pioneer.position.y)
        action = Action.summon_treasure(nearby_target, ('StarSand',))
        missing = validate_decision(
            observed,
            Decision(commands={pioneer.unit_id: action}),
        )
        stocked_observation = replace(
            observed,
            our=replace(
                observed.our,
                units=tuple(
                    replace(unit, backpack=('StarSand',))
                    if unit.unit_id == pioneer.unit_id
                    else unit
                    for unit in observed.our.units
                ),
            ),
        )
        stocked = validate_decision(
            stocked_observation,
            Decision(commands={pioneer.unit_id: action}),
        )

        self.assertNotIn(pioneer.unit_id, missing.commands)
        self.assertIn(pioneer.unit_id, stocked.commands)

        distant = validate_decision(
            stocked_observation,
            Decision(
                commands={
                    pioneer.unit_id: Action.summon_treasure(
                        Position(0, 0),
                        ('StarSand',),
                    )
                }
            ),
        )
        self.assertNotIn(pioneer.unit_id, distant.commands)

    def test_buy_requires_shop_inventory_capacity_and_aggregate_gold(self) -> None:
        workers = tuple(
            unit for unit in self.observed.our.units if unit.role_type == 'worker'
        )[:2]
        observed = replace(
            self.observed,
            zones=(Zone(workers[0].position, 'vendor'),),
            vendor_shop=(ShopItem('StarSand', 60),),
            our=replace(
                self.observed.our,
                gold=100,
                units=tuple(
                    replace(unit, position=workers[0].position)
                    if unit.unit_id in {worker.unit_id for worker in workers}
                    else unit
                    for unit in self.observed.our.units
                ),
            ),
        )
        decision = Decision(
            commands={worker.unit_id: Action.buy('StarSand', 1) for worker in workers}
        )

        validated = validate_decision(observed, decision)

        self.assertEqual(len(validated.commands), 1)


if __name__ == "__main__":
    unittest.main()
