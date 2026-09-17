import unittest

from future_war_agent.protocol.models import Position
from future_war_agent.strategy.simulation.state import (
    SimRobot,
    SimState,
    SimStructure,
    SimWeapon,
)
from future_war_agent.strategy.simulation.weapons import (
    WeaponAttack,
    generate_weapon_attacks,
    weapon_damage,
)


class SimulationWeaponTests(unittest.TestCase):
    def test_gatling_requires_level_sized_targets_inside_one_cone(self) -> None:
        weapon = self._weapon("gatling", level=2, attack_power=10, attack_range=8)
        state = self._state(
            weapon,
            (
                self._robot(1, 4, 1),
                self._robot(2, 1, 4),
                self._robot(3, 0, 4),
            ),
        )

        attacks = generate_weapon_attacks(state, weapon, controller_id=7)

        self.assertTrue(attacks)
        self.assertLessEqual(len(attacks), 3)
        self.assertTrue(all(len(attack.targets) == 2 for attack in attacks))
        self.assertNotIn(
            (Position(4, 1), Position(0, 4)),
            tuple(attack.targets for attack in attacks),
        )

    def test_gatling_hits_nearest_robot_for_fixed_ten_damage(self) -> None:
        weapon = self._weapon("gatling", level=1, attack_power=10, attack_range=8)
        state = self._state(
            weapon,
            (self._robot(1, 3, 1), self._robot(2, 6, 1)),
        )
        attack = WeaponAttack(weapon.unit_id, 7, (Position(6, 1),))

        self.assertEqual(dict(weapon_damage(state, attack)), {1: 10})

    def test_railgun_spends_energy_in_ray_order_and_stops_at_zero(self) -> None:
        weapon = self._weapon("railgun", level=1, attack_power=25, attack_range=8)
        state = self._state(
            weapon,
            (
                self._robot(1, 2, 1, health=10),
                self._robot(2, 4, 1, health=20),
                self._robot(3, 6, 1, health=30),
            ),
        )
        attack = WeaponAttack(weapon.unit_id, 7, (Position(6, 1),))

        self.assertEqual(dict(weapon_damage(state, attack)), {1: 10, 2: 15})

    def test_rocket_stacks_fixed_center_and_splash_damage(self) -> None:
        weapon = self._weapon("rocket", level=2, attack_power=20, attack_range=8)
        state = self._state(
            weapon,
            (
                self._robot(1, 3, 3, health=50),
                self._robot(2, 4, 3, health=50),
                self._robot(3, 3, 4, health=50),
            ),
        )
        attack = WeaponAttack(
            weapon.unit_id,
            7,
            (Position(3, 3), Position(4, 3)),
        )

        self.assertEqual(
            dict(weapon_damage(state, attack)),
            {1: 30, 2: 30, 3: 20},
        )

    def test_weapon_cooldown_and_range_gate_candidates(self) -> None:
        cooling = self._weapon(
            "rocket",
            level=1,
            attack_power=20,
            attack_range=8,
            cooldown=1,
        )
        short_range = self._weapon(
            "railgun",
            level=1,
            attack_power=20,
            attack_range=2,
        )

        self.assertEqual(
            generate_weapon_attacks(
                self._state(cooling, (self._robot(1, 3, 1),)),
                cooling,
                controller_id=7,
            ),
            (),
        )
        self.assertEqual(
            generate_weapon_attacks(
                self._state(short_range, (self._robot(1, 4, 1),)),
                short_range,
                controller_id=7,
            ),
            (),
        )

    def test_candidates_are_stably_truncated_to_three(self) -> None:
        weapon = self._weapon("rocket", level=2, attack_power=20, attack_range=8)
        state = self._state(
            weapon,
            (self._robot(1, 3, 3), self._robot(2, 5, 4)),
        )

        first = generate_weapon_attacks(state, weapon, controller_id=7)
        second = generate_weapon_attacks(state, weapon, controller_id=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, tuple(sorted(first, key=lambda item: item.stable_key)))

    def test_conservative_ray_folding_requires_all_interpretations(self) -> None:
        weapon = self._weapon("gatling", level=1, attack_power=10, attack_range=8)
        state = self._state(
            weapon,
            (self._robot(1, 3, 2), self._robot(2, 2, 1)),
        )
        attack = WeaponAttack(weapon.unit_id, 7, (Position(3, 2),))

        self.assertEqual(dict(weapon_damage(state, attack)), {})

    def test_no_candidate_when_level_exceeds_distinct_legal_cells(self) -> None:
        weapon = self._weapon("gatling", level=2, attack_power=10, attack_range=8)
        state = self._state(weapon, (self._robot(1, 3, 1),))

        self.assertEqual(
            generate_weapon_attacks(state, weapon, controller_id=7),
            (),
        )

    @staticmethod
    def _weapon(
        role_type: str,
        *,
        level: int,
        attack_power: int,
        attack_range: int,
        cooldown: int = 0,
    ) -> SimWeapon:
        return SimWeapon(
            unit_id=100,
            role_type=role_type,
            position=Position(1, 1),
            health=100,
            attack_power=attack_power,
            attack_range=attack_range,
            level=level,
            cooldown=cooldown,
        )

    @staticmethod
    def _robot(
        robot_id: int,
        x: int,
        y: int,
        *,
        health: int = 40,
        attack_power: int = 5,
        kill_score: int = 1,
    ) -> SimRobot:
        return SimRobot(
            robot_id=robot_id,
            role_type="smallRobot",
            position=Position(x, y),
            health=health,
            attack_power=attack_power,
            attack_range=3,
            kill_score=kill_score,
            waits_this_turn=False,
        )

    @staticmethod
    def _state(weapon: SimWeapon, robots: tuple[SimRobot, ...]) -> SimState:
        station = SimStructure(
            unit_id=1,
            role_type="station",
            position=Position(8, 8),
            occupied_cells=frozenset(
                {Position(8, 8), Position(9, 8), Position(8, 9), Position(9, 9)}
            ),
            health=500,
            level=1,
        )
        return SimState(
            round_no=71,
            width=12,
            height=12,
            remaining_night_turns=60,
            team_type="challenger",
            static_blocked=station.occupied_cells | frozenset({weapon.position}),
            roles=(),
            station=station,
            walls=(),
            weapons=(weapon,),
            robots=robots,
        )


if __name__ == "__main__":
    unittest.main()
