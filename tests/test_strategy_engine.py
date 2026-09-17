import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from math import ceil
from pathlib import Path
from statistics import median

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.director import StrategicDirector
from future_war_agent.strategy.policy import (
    DEFAULT_STRATEGIC_INTENT,
    ItemPolicy,
    StrategicIntent,
)
from future_war_agent.strategy.simulation.candidates import SimJointAction
from future_war_agent.strategy.simulation.certificate import (
    ScenarioOutcome,
    build_certificate,
)
from future_war_agent.strategy.simulation.config import Phase3Config
from future_war_agent.strategy.simulation.errors import (
    DeadlineExceeded,
    UnsupportedSimulation,
)
from future_war_agent.strategy.simulation.objective import (
    DEFAULT_NIGHT_OBJECTIVE,
)
from future_war_agent.strategy.simulation.search import (
    SearchResult,
    SearchStats,
)


FIXTURES = Path(__file__).parent / "fixtures"
DAY_FIXTURE = FIXTURES / "phase3_day_request.json"
NIGHT_FIXTURE = FIXTURES / "phase3_night_request.json"
UNIFORM_WEIGHTS = (
    Fraction(1, 4),
    Fraction(1, 4),
    Fraction(1, 4),
    Fraction(1, 4),
)


def load_observation(path: Path) -> Observation:
    return parse_observation(json.loads(path.read_text(encoding="utf-8")))


def search_result() -> SearchResult:
    outcome = ScenarioOutcome(
        weight=Fraction(1, 4),
        station_health=1000,
        surviving_controlled_role_count=3,
        surviving_controller_count=3,
        controller_losses=0,
        surviving_key_weapon_count=3,
        key_weapon_losses=0,
        wall_losses=0,
        weapon_losses=0,
        minimum_controlled_role_health=100,
        surviving_wall_non_key_weapon_value=1000,
        owned_kill_score=3,
        remaining_threat=0,
        remaining_one_turn_damage=0,
        ended_with_night=False,
    )
    return SearchResult(
        decision=Decision(
            commands={
                300: Action.attack(
                    101,
                    (Position(5, 4), Position(5, 5)),
                )
            }
        ),
        simulation_action=SimJointAction(),
        certificate=build_certificate(
            (outcome, outcome, outcome, outcome),
            DEFAULT_NIGHT_OBJECTIVE,
        ),
        stats=SearchStats(1, 1, 4, 1),
    )


class Phase2Spy:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.decisions: list[Decision] = []
        self.intents: list[StrategicIntent] = []

    def __call__(
        self,
        observation: Observation,
        *,
        intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
    ) -> Decision:
        self.observations.append(observation)
        self.intents.append(intent)
        decision = Decision(prompt=f"phase2-{len(self.observations)}")
        self.decisions.append(decision)
        return decision


class DirectorSpy:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[Observation] = []
        self.delegate = StrategicDirector()

    def select(self, observation: Observation, **kwargs):
        self.calls.append(observation)
        if self.failure is not None:
            raise self.failure
        return self.delegate.select(observation, **kwargs)


class SearchSpy:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, object]] = []
        self.result = search_result()
        self.delay_seconds = 0.0

    def __call__(
        self,
        observation: Observation,
        scenario_weights: tuple[Fraction, Fraction, Fraction, Fraction],
        **kwargs: object,
    ) -> SearchResult:
        self.calls.append(
            {
                "observation": observation,
                "weights": scenario_weights,
                **kwargs,
            }
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.failure is not None:
            raise self.failure
        return self.result


class StrategyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = load_observation(DAY_FIXTURE)
        self.night = load_observation(NIGHT_FIXTURE)

    def make_engine(
        self,
        *,
        failure: Exception | None = None,
        objective_provider=None,
        reconciler=None,
        director=None,
    ) -> tuple[StrategyEngine, Phase2Spy, SearchSpy]:
        phase2 = Phase2Spy()
        search = SearchSpy(failure)
        kwargs = {}
        if objective_provider is not None:
            kwargs["objective_provider"] = objective_provider
        if reconciler is not None:
            kwargs["scenario_reconciler"] = reconciler
        if director is not None:
            kwargs['director'] = director
        engine = StrategyEngine(
            config=Phase3Config(watchdog_seconds=0.8),
            phase2_planner=phase2,
            night_searcher=search,
            clock=lambda: 100.0,
            **kwargs,
        )
        return engine, phase2, search

    def test_first_mid_night_observation_uses_and_caches_phase_2(self) -> None:
        engine, phase2, search = self.make_engine()

        first = engine.plan(self.night)
        duplicate = engine.plan(self.night)

        self.assertIs(first, phase2.decisions[0])
        self.assertIs(duplicate, first)
        self.assertEqual(len(phase2.observations), 1)
        self.assertEqual(search.calls, [])

    def test_consecutive_day_to_night_uses_phase_3_and_objective(self) -> None:
        objectives: list[Observation] = []

        def objective_provider(observation: Observation):
            objectives.append(observation)
            return DEFAULT_NIGHT_OBJECTIVE

        engine, phase2, search = self.make_engine(
            objective_provider=objective_provider
        )

        day_decision = engine.plan(self.day)
        night_decision = engine.plan(self.night)

        self.assertIs(day_decision, phase2.decisions[0])
        self.assertIs(night_decision, search.result.decision)
        self.assertEqual(objectives, [self.night])
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(search.calls[0]["weights"], UNIFORM_WEIGHTS)
        self.assertEqual(search.calls[0]["deadline"], 100.8)
        self.assertIs(search.calls[0]["clock"], engine._clock)

    def test_duplicate_night_request_performs_one_search(self) -> None:
        engine, _, search = self.make_engine()
        engine.plan(self.day)

        first = engine.plan(self.night)
        second = engine.plan(self.night)

        self.assertIs(second, first)
        self.assertEqual(len(search.calls), 1)

    def test_duplicate_request_runs_director_once(self) -> None:
        director = DirectorSpy()
        engine, phase2, _ = self.make_engine(director=director)

        first = engine.plan(self.day)
        duplicate = engine.plan(self.day)

        self.assertIs(duplicate, first)
        self.assertEqual(len(director.calls), 1)
        self.assertEqual(len(phase2.intents), 1)

    def test_director_failure_uses_default_phase2_intent(self) -> None:
        director = DirectorSpy(RuntimeError('director failed'))
        engine, phase2, search = self.make_engine(director=director)

        decision = engine.plan(self.day)

        self.assertIs(decision, phase2.decisions[0])
        self.assertIs(phase2.intents[0], DEFAULT_STRATEGIC_INTENT)
        self.assertEqual(search.calls, [])

    def test_emergency_medicine_bypasses_phase3_search(self) -> None:
        injured_id = next(
            role.unit_id
            for role in self.night.our.units
            if role.role_type in {'worker', 'pioneer'}
        )
        injured_night = replace(
            self.night,
            our=replace(
                self.night.our,
                units=tuple(
                    replace(
                        role,
                        health=50,
                        backpack=role.backpack + ('Medicine',),
                    )
                    if role.unit_id == injured_id
                    else role
                    for role in self.night.our.units
                ),
            ),
        )
        engine, phase2, search = self.make_engine()

        engine.plan(self.day)
        decision = engine.plan(injured_night)

        self.assertIs(decision, phase2.decisions[-1])
        self.assertEqual(search.calls, [])
        self.assertGreater(
            phase2.intents[-1].item_policy.medicine_health_threshold,
            0,
        )

    def test_same_round_revision_replans_without_reconciliation(self) -> None:
        reconciled = (
            Fraction(1, 10),
            Fraction(1, 5),
            Fraction(3, 10),
            Fraction(2, 5),
        )
        reconciliation_calls: list[tuple[object, Observation]] = []

        def reconciler(previous, current, *, config):
            reconciliation_calls.append((previous, current))
            return reconciled

        engine, _, search = self.make_engine(reconciler=reconciler)
        engine.plan(self.day)
        engine.plan(self.night)
        revised = replace(
            self.night,
            our=replace(self.night.our, gold=self.night.our.gold + 1),
        )

        engine.plan(revised)

        self.assertEqual(len(search.calls), 2)
        self.assertEqual(len(reconciliation_calls), 1)
        self.assertEqual(search.calls[0]["weights"], reconciled)
        self.assertEqual(search.calls[1]["weights"], reconciled)

    def test_discontinuities_use_phase_2_and_do_not_search(self) -> None:
        station = next(
            unit for unit in self.night.our.units if unit.role_type == "station"
        )
        cases = {
            "gap": (
                self.day,
                replace(self.night, time=TurnTime.from_round(72)),
            ),
            "rollback": (
                replace(self.night, time=TurnTime.from_round(72)),
                self.night,
            ),
            "map signature": (
                self.day,
                replace(self.night, width=self.night.width + 1),
            ),
            "station signature": (
                self.day,
                replace(
                    self.night,
                    our=replace(
                        self.night.our,
                        units=tuple(
                            replace(unit, unit_id=station.unit_id + 1)
                            if unit.unit_id == station.unit_id
                            else unit
                            for unit in self.night.our.units
                        ),
                    ),
                ),
            ),
        }
        for label, (previous, current) in cases.items():
            with self.subTest(label=label):
                engine, phase2, search = self.make_engine()
                engine.plan(previous)

                decision = engine.plan(current)

                self.assertIs(decision, phase2.decisions[-1])
                self.assertEqual(search.calls, [])

    def test_blank_team_id_never_creates_a_cached_session(self) -> None:
        engine, phase2, search = self.make_engine()
        blank = replace(
            self.night,
            our=replace(self.night.our, team_id="   "),
        )

        first = engine.plan(blank)
        second = engine.plan(blank)

        self.assertIsNot(first, second)
        self.assertEqual(len(phase2.observations), 2)
        self.assertEqual(search.calls, [])

    def test_team_sessions_are_isolated(self) -> None:
        engine, _, search = self.make_engine()
        day_b = replace(
            self.day,
            our=replace(self.day.our, team_id="phase3-team-b"),
        )
        night_b = replace(
            self.night,
            our=replace(self.night.our, team_id="phase3-team-b"),
        )

        engine.plan(self.day)
        engine.plan(day_b)
        decision_a = engine.plan(self.night)
        decision_b = engine.plan(night_b)

        self.assertIs(decision_a, search.result.decision)
        self.assertIs(decision_b, search.result.decision)
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(
            [call["observation"].our.team_id for call in search.calls],
            ["phase3-team", "phase3-team-b"],
        )

    def test_phase_3_failures_fall_back_atomically_and_cache_result(self) -> None:
        failures = (
            UnsupportedSimulation("unsupported"),
            DeadlineExceeded("late"),
            RuntimeError("unexpected"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                engine, phase2, search = self.make_engine(failure=failure)
                engine.plan(self.day)

                fallback = engine.plan(self.night)
                duplicate = engine.plan(self.night)

                self.assertIs(fallback, phase2.decisions[1])
                self.assertIs(duplicate, fallback)
                self.assertEqual(len(phase2.observations), 2)
                self.assertIs(phase2.observations[1], self.night)
                self.assertEqual(len(search.calls), 1)

    def test_phase_2_exception_is_not_swallowed(self) -> None:
        def broken(
            _: Observation,
            *,
            intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
        ) -> Decision:
            del intent
            raise ValueError("phase 2 failure")

        engine = StrategyEngine(phase2_planner=broken)

        with self.assertRaisesRegex(ValueError, "phase 2 failure"):
            engine.plan(self.day)

    def test_concurrent_duplicates_execute_one_search(self) -> None:
        engine, _, search = self.make_engine()
        search.delay_seconds = 0.05
        engine.plan(self.day)

        with ThreadPoolExecutor(max_workers=4) as executor:
            decisions = tuple(executor.map(engine.plan, (self.night,) * 4))

        self.assertEqual(len(search.calls), 1)
        self.assertTrue(all(item is decisions[0] for item in decisions))

    def test_supported_night_latency_stays_below_external_limit(self) -> None:
        fresh_samples: list[float] = []
        for _ in range(20):
            engine = StrategyEngine()
            engine.plan(self.day)
            started = time.perf_counter()
            engine.plan(self.night)
            fresh_samples.append(time.perf_counter() - started)
            session = engine._sessions.get(self.night.our.team_id)
            self.assertIsNotNone(session)
            self.assertIsNotNone(
                session.simulation_action if session is not None else None
            )

        cached_engine = StrategyEngine()
        cached_engine.plan(self.day)
        cached_engine.plan(self.night)
        cached_session = cached_engine._sessions.get(self.night.our.team_id)
        self.assertIsNotNone(cached_session)
        self.assertIsNotNone(
            cached_session.simulation_action
            if cached_session is not None
            else None
        )
        cached_samples: list[float] = []
        for _ in range(100):
            started = time.perf_counter()
            cached_engine.plan(self.night)
            cached_samples.append(time.perf_counter() - started)

        self.assertTrue(
            all(sample < 5.0 for sample in fresh_samples),
            max(fresh_samples),
        )
        self.assertTrue(
            all(sample < 5.0 for sample in cached_samples),
            max(cached_samples),
        )
        fresh_p99 = sorted(fresh_samples)[
            ceil(len(fresh_samples) * 0.99) - 1
        ]
        cached_p99 = sorted(cached_samples)[
            ceil(len(cached_samples) * 0.99) - 1
        ]
        print(
            "Phase 3 latency: "
            f"fresh median={median(fresh_samples) * 1000:.2f}ms "
            f"p99={fresh_p99 * 1000:.2f}ms; "
            f"cached median={median(cached_samples) * 1000:.3f}ms "
            f"p99={cached_p99 * 1000:.3f}ms"
        )


if __name__ == "__main__":
    unittest.main()
