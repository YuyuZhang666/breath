import random

from tests.strategy_helpers import observation, robot, unit


def seeded_night_observation(seed: int):
    randomizer = random.Random(seed)
    positions = (
        (4, 4),
        (4, 8),
        (4, 12),
        (4, 16),
        (8, 4),
        (8, 16),
        (12, 4),
        (12, 16),
        (16, 4),
        (16, 8),
        (16, 12),
        (16, 16),
    )
    selected = randomizer.sample(positions, k=6)
    robot_types = ('smallRobot', 'mediumRobot', 'largeRobot', 'bossRobot')
    health_by_type = {
        'smallRobot': 40,
        'mediumRobot': 80,
        'largeRobot': 160,
        'bossRobot': 400,
    }
    robots = tuple(
        robot(
            30_000 + index,
            x,
            y,
            role_type=(
                role_type := randomizer.choices(
                    robot_types,
                    weights=(6, 3, 2, 1),
                    k=1,
                )[0]
            ),
            health=health_by_type[role_type],
        )
        for index, (x, y) in enumerate(selected)
    )
    return observation(
        round_no=71,
        width=21,
        height=21,
        our_units=(
            unit(101, 7, 10, 'worker'),
            unit(102, 10, 7, 'worker'),
            unit(103, 13, 10, 'pioneer'),
            unit(200, 10, 10, 'station', health=1000, level=1),
            unit(
                300,
                8,
                10,
                'gatling',
                health=1000,
                attack_power=35,
                attack_range=10,
                level=3,
            ),
            unit(
                301,
                10,
                8,
                'railgun',
                health=1000,
                attack_power=90,
                attack_range=12,
                level=2,
            ),
            unit(
                302,
                12,
                10,
                'rocket',
                health=1000,
                attack_power=55,
                attack_range=10,
                level=3,
            ),
        ),
        robots=robots,
    )
