import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from math import ceil
from pathlib import Path
from statistics import median
from unittest.mock import patch

from future_war_agent.decision.actions import Action
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Observation, Position, Zone
from future_war_agent.protocol.parser import parse_observation
from future_war_agent.protocol.time import TurnTime
from future_war_agent.strategy.build_recovery import BuildRecoveryState
from future_war_agent.strategy.compute import ComputeGovernor, ComputeGovernorConfig
from future_war_agent.strategy.build_recovery import BuildRecoveryState
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.director import StrategicDirector
from future_war_agent.strategy.forecast import (
    RiskLevel,
    refresh_night_forecast,
)
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
from future_war_agent.strategy.simulation.config import Phase3Config, Phase3Level
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
from future_war_agent.strategy.world import WorldGrid
from future_war_agent.telemetry import TelemetryRecorder
from tests.strategy_helpers import observation, robot, unit


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

    def test_night_pipeline_constructs_current_world_grid_once(self) -> None:
        engine, _, _ = self.make_engine()

        with patch.object(
            WorldGrid,
            'from_observation',
            wraps=WorldGrid.from_observation,
        ) as build_world:
            engine.plan(self.night)

        self.assertEqual(build_world.call_count, 1)

    def test_phase2_receives_daily_anchor_and_historical_wall_sites(self) -> None:
        captured: list[tuple[tuple[Position, ...], frozenset[Position]]] = []

        def phase2(
            observed,
            *,
            fortification_threats=(),
            previously_built_wall_sites=frozenset(),
            **kwargs,
        ):
            del observed, kwargs
            captured.append(
                (fortification_threats, previously_built_wall_sites)
            )
            return Decision()

        engine = StrategyEngine(phase2_planner=phase2)
        wall_position = Position(8, 8)
        engine.plan(
            observation(
                round_no=71,
                our_units=(
                    unit(10, 5, 5, 'station', level=1),
                    unit(20, 8, 8, 'wall', level=1),
                ),
                robots=(robot(501, 12, 7), robot(502, 11, 8)),
            )
        )

        engine.plan(
            observation(
                round_no=131,
                our_units=(unit(10, 5, 5, 'station', level=1),),
            )
        )

        self.assertEqual(
            captured[-1][0],
            (Position(11, 8), Position(12, 7)),
        )
        self.assertIn(wall_position, captured[-1][1])

    def test_failed_build_result_reaches_next_round_phase2(self) -> None:
        target = Position(9, 6)
        captured: list[BuildRecoveryState] = []

        def phase2(observed, *, build_recovery, **kwargs):
            del kwargs
            captured.append(build_recovery)
            if observed.time.round_no == 1:
                return Decision(
                    commands={10010: Action.build('rocket', target)}
                )
            return Decision()

        engine = StrategyEngine(phase2_planner=phase2)
        units = (
            unit(10010, 3, 3, 'worker'),
            unit(10013, 7, 7, 'station', level=1),
        )

        engine.plan(observation(round_no=1, our_units=units))
        engine.plan(
            observation(
                round_no=2,
                our_units=units,
                last_action_results={10010: False},
            )
        )

        self.assertEqual(captured[0].failures, ())
        self.assertEqual(len(captured[1].failures), 1)
        failure = captured[1].failures[0]
        self.assertEqual(failure.name, 'rocket')
        self.assertEqual(failure.target, target)
        self.assertEqual(failure.consecutive_failures, 1)

    def test_safe_active_night_task_reserves_pioneer_from_fire_control(
        self,
    ) -> None:
        captured: list[tuple[object, ...] | None] = []

        def phase2(
            observation,
            *,
            intent=DEFAULT_STRATEGIC_INTENT,
            controller_assignments=None,
        ):
            del observation, intent
            captured.append(controller_assignments)
            return Decision(prompt='phase2')

        def safe_forecaster(observation, **kwargs):
            refresh = refresh_night_forecast(observation, **kwargs)
            return replace(
                refresh,
                forecast=replace(
                    refresh.forecast,
                    risk_level=RiskLevel.SAFE,
                ),
            )

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_forecaster=safe_forecaster,
            clock=lambda: 100.0,
        )
        observed = replace(
            self.night,
            phase_task='Return the exact answer.',
        )

        decision = engine.plan(observed)

        self.assertNotIn(103, {item.role_id for item in captured[0]})
        self.assertTrue(decision.prompt)
        session = engine._sessions.get(observed.our.team_id)
        self.assertTrue(session.task_state.active_task_type)

    def test_opening_hard_recall_skips_task_and_treasure_overlays(
        self,
    ) -> None:
        class OverlaySpy:
            def __init__(self) -> None:
                self.apply_calls = 0

            def reconcile(self, observation, state):
                del observation
                return state

            def apply(self, *args, **kwargs):
                del args, kwargs
                self.apply_calls += 1
                raise AssertionError('opening recall must not be overlaid')

        task_agent = OverlaySpy()
        treasure_agent = OverlaySpy()
        engine = StrategyEngine(
            task_agent=task_agent,
            treasure_agent=treasure_agent,
        )
        observed = observation(
            round_no=66,
            our_units=(
                unit(10, 1, 1, 'worker'),
                unit(11, 2, 1, 'worker'),
                unit(20, 10, 10, 'station', health=500, level=1),
            ),
            zones=(Zone(Position(1, 2), 'stone'),),
        )

        decision = engine.plan(observed)

        self.assertTrue(decision.commands)
        self.assertEqual(task_agent.apply_calls, 0)
        self.assertEqual(treasure_agent.apply_calls, 0)

    def test_critical_active_night_task_releases_pioneer_to_fire_control(
        self,
    ) -> None:
        forecast_assignments: list[tuple[object, ...] | None] = []
        phase2_assignments: list[tuple[object, ...] | None] = []

        def phase2(
            observation,
            *,
            intent=DEFAULT_STRATEGIC_INTENT,
            controller_assignments=None,
        ):
            del observation, intent
            phase2_assignments.append(controller_assignments)
            return Decision(prompt='phase2')

        def critical_forecaster(observation, **kwargs):
            forecast_assignments.append(kwargs.get('controller_assignments'))
            refresh = refresh_night_forecast(observation, **kwargs)
            return replace(
                refresh,
                forecast=replace(
                    refresh.forecast,
                    risk_level=RiskLevel.CRITICAL,
                ),
            )

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_forecaster=critical_forecaster,
            clock=lambda: 100.0,
        )
        observed = replace(
            self.night,
            phase_task='Return the exact answer.',
        )

        decision = engine.plan(observed)

        self.assertNotIn(
            103,
            {item.role_id for item in forecast_assignments[0]},
        )
        self.assertIn(
            103,
            {item.role_id for item in phase2_assignments[0]},
        )
        self.assertEqual(decision.prompt, 'phase2')
        session = engine._sessions.get(observed.our.team_id)
        self.assertEqual(session.task_state.active_task_type, '')
        self.assertEqual(session.task_state.abandoned_task_fingerprint, '')

    def test_active_night_task_fails_closed_when_assignment_cache_fails(
        self,
    ) -> None:
        class BrokenAssignmentCache:
            def resolve(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError('assignment unavailable')

        def phase2(
            observation,
            *,
            intent=DEFAULT_STRATEGIC_INTENT,
            controller_assignments=None,
        ):
            del observation, intent, controller_assignments
            return Decision(prompt='phase2')

        def safe_forecaster(observation, **kwargs):
            refresh = refresh_night_forecast(observation, **kwargs)
            return replace(
                refresh,
                forecast=replace(
                    refresh.forecast,
                    risk_level=RiskLevel.SAFE,
                ),
            )

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_forecaster=safe_forecaster,
            controller_assignment_cache=BrokenAssignmentCache(),
            clock=lambda: 100.0,
        )
        observed = replace(
            self.night,
            phase_task='Return the exact answer.',
        )

        decision = engine.plan(observed)

        self.assertEqual(decision.prompt, 'phase2')

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
        self.assertEqual(search.calls[0]["deadline"], 100.08)
        self.assertEqual(search.calls[0]["level"].value, "lite")
        self.assertIs(search.calls[0]["baseline_decision"], phase2.decisions[1])
        self.assertIs(search.calls[0]["clock"], engine._clock)
        assignments = search.calls[0]['controller_assignments']
        self.assertTrue(assignments)
        self.assertEqual(
            len({item.role_id for item in assignments}),
            len(assignments),
        )

    def test_duplicate_night_request_performs_one_search(self) -> None:
        engine, _, search = self.make_engine()
        engine.plan(self.day)

        first = engine.plan(self.night)
        second = engine.plan(self.night)

        self.assertIs(second, first)
        self.assertEqual(len(search.calls), 1)

    def test_stable_night_uses_phase2_5_without_rollout(self) -> None:
        engine, phase2, search = self.make_engine()
        stable_night = replace(self.night, robots=())

        engine.plan(self.day)
        decision = engine.plan(stable_night)

        self.assertIs(decision, phase2.decisions[-1])
        self.assertEqual(search.calls, [])

    def test_full_failure_degrades_to_lite_before_phase2_5(self) -> None:
        phase2 = Phase2Spy()
        result = search_result()
        attempts: list[Phase3Level] = []
        telemetry = TelemetryRecorder()

        def searcher(*args, level=Phase3Level.FULL, **kwargs):
            del args, kwargs
            attempts.append(level)
            if level is Phase3Level.FULL:
                raise DeadlineExceeded('full timeout')
            return result

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_searcher=searcher,
            clock=lambda: 100.0,
            telemetry=telemetry,
            compute_governor=ComputeGovernor(
                ComputeGovernorConfig(night_full_forecast_enabled=True)
            ),
        )
        station_id = next(
            item.unit_id for item in self.night.our.units if item.role_type == 'station'
        )
        damaged_night = replace(
            self.night,
            our=replace(
                self.night.our,
                units=tuple(
                    replace(item, health=item.health - 100)
                    if item.unit_id == station_id
                    else item
                    for item in self.night.our.units
                ),
            ),
        )

        engine.plan(self.day)
        decision = engine.plan(damaged_night)

        self.assertIs(decision, result.decision)
        self.assertEqual(attempts, [Phase3Level.FULL, Phase3Level.LITE])
        self.assertEqual(len(phase2.observations), 2)
        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.phase3_level, 'full')
        self.assertEqual(sample.phase3_effective_level, 'full')
        self.assertEqual(sample.phase3_executed_level, 'lite')
        self.assertEqual(sample.phase3_skip_reason, 'none')
        self.assertEqual(
            sample.phase3_completed_root_count,
            result.stats.roots_evaluated,
        )
        self.assertEqual(sample.phase3_fallback_count, 1)
        self.assertTrue(sample.watchdog_hit)
        self.assertTrue(sample.fallback_used)

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

    def test_director_failure_atomically_skips_phase3_on_eligible_night(self) -> None:
        director = DirectorSpy(RuntimeError('director failed'))
        engine, phase2, search = self.make_engine(director=director)
        engine.plan(self.day)

        decision = engine.plan(self.night)

        self.assertIs(decision, phase2.decisions[-1])
        self.assertIs(phase2.intents[-1], DEFAULT_STRATEGIC_INTENT)
        self.assertEqual(search.calls, [])

    def test_latest_phase3_certificate_survives_non_search_turn(self) -> None:
        engine, _, search = self.make_engine()
        engine.plan(self.day)
        engine.plan(self.night)
        injured_id = next(
            role.unit_id
            for role in self.night.our.units
            if role.role_type in {'worker', 'pioneer'}
        )
        emergency_revision = replace(
            self.night,
            our=replace(
                self.night.our,
                gold=self.night.our.gold + 1,
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

        engine.plan(emergency_revision)

        session = engine._sessions.get(self.night.our.team_id)
        self.assertIs(session.certificate, search.result.certificate)
        self.assertEqual(len(search.calls), 1)

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

    def test_destroyed_own_station_skips_forecast_and_phase3(self) -> None:
        phase2 = Phase2Spy()
        search = SearchSpy()
        telemetry = TelemetryRecorder()

        def forecaster(*args, **kwargs):
            del args, kwargs
            raise AssertionError('forecast must not run after station destruction')

        engine = StrategyEngine(
            phase2_planner=phase2,
            night_searcher=search,
            night_forecaster=forecaster,
            telemetry=telemetry,
            clock=lambda: 100.0,
        )
        destroyed = replace(
            self.night,
            our=replace(
                self.night.our,
                units=tuple(
                    replace(item, health=0)
                    if item.role_type == 'station'
                    else item
                    for item in self.night.our.units
                ),
            ),
        )

        engine.plan(self.day)
        engine.plan(destroyed)

        sample = telemetry.snapshot()[-1]
        self.assertEqual(search.calls, [])
        self.assertFalse(sample.own_station_alive)
        self.assertEqual(sample.own_station_status, 'destroyed')
        self.assertEqual(sample.performance_segment, 'post_station_loss')
        self.assertEqual(sample.forecast_mode, 'skipped')
        self.assertEqual(sample.forecast_reason, 'own_station_destroyed')
        self.assertEqual(sample.phase3_executed_level, 'none')
        self.assertEqual(sample.phase3_skip_reason, 'own_station_destroyed')
        self.assertGreater(
            phase2.intents[-1].item_policy.medicine_health_threshold,
            0,
        )

    def test_forecast_telemetry_proves_current_observation_was_used(
        self,
    ) -> None:
        telemetry = TelemetryRecorder()
        engine = StrategyEngine(telemetry=telemetry, clock=lambda: 100.0)

        engine.plan(self.day)
        engine.plan(self.night)

        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.performance_segment, 'night_alive')
        self.assertTrue(sample.forecast_observation_signature)
        self.assertEqual(
            sample.forecast_model_input_signature,
            sample.forecast_observation_signature,
        )
        self.assertTrue(sample.forecast_input_current)
        self.assertGreater(sample.forecast_observed_station_hp, 0)
        self.assertGreaterEqual(sample.forecast_hostile_robot_count, 0)
        self.assertGreaterEqual(sample.forecast_ready_weapon_count, 0)

    def test_lowercase_inventory_medicine_bypasses_phase3_search(self) -> None:
        injured_id = next(
            role.unit_id
            for role in self.night.our.units
            if role.role_type in {"worker", "pioneer"}
        )
        injured_night = replace(
            self.night,
            our=replace(
                self.night.our,
                units=tuple(
                    replace(
                        role,
                        health=50,
                        backpack=role.backpack + ("medicine",),
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

    def test_same_round_revision_without_new_event_skips_rollout(self) -> None:
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

        self.assertEqual(len(search.calls), 1)
        self.assertEqual(len(reconciliation_calls), 1)
        self.assertEqual(search.calls[0]["weights"], reconciled)

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

    def test_governor_updates_inside_outer_http_telemetry_context(self) -> None:
        phase2 = Phase2Spy()
        search = SearchSpy()
        telemetry = TelemetryRecorder()
        governor = ComputeGovernor()
        engine = StrategyEngine(
            config=Phase3Config(watchdog_seconds=0.8),
            phase2_planner=phase2,
            night_searcher=search,
            clock=lambda: 100.0,
            telemetry=telemetry,
            compute_governor=governor,
        )
        outer = telemetry.begin()
        try:
            engine.plan(self.day)
            engine.plan(self.night)
        finally:
            telemetry.finish(outer)

        state = governor.state_for(self.night.our.team_id)
        self.assertGreater(state.ewma_root_rollout_ms, 0)
        self.assertGreater(state.last_phase3_ms, 0)

    def test_governor_watchdog_cooldown_skips_phase3(self) -> None:
        governor = ComputeGovernor()
        governor.observe_turn(
            self.day.our.team_id,
            round_no=70,
            day_no=1,
            phase2_5_ms=1.0,
            phase3_ms=0.0,
            roots_evaluated=0,
            scenarios_per_root=0,
            exact_horizon=0,
            watchdog_hit=True,
        )
        phase2 = Phase2Spy()
        search = SearchSpy()
        engine = StrategyEngine(
            phase2_planner=phase2,
            night_searcher=search,
            clock=lambda: 100.0,
            compute_governor=governor,
        )

        engine.plan(self.day)
        decision = engine.plan(self.night)

        self.assertIs(decision, phase2.decisions[-1])
        self.assertEqual(search.calls, [])

    def test_emergency_reserve_skips_reconciliation_and_new_task_work(self) -> None:
        class TaskSpy:
            def __init__(self) -> None:
                self.calls = 0
                self.reconcile_calls = 0

            def reconcile(self, observation, state):
                del observation
                self.reconcile_calls += 1
                return state

            def apply(self, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                raise AssertionError('TaskAgent must not run inside reserve')

        task_agent = TaskSpy()
        engine = StrategyEngine(
            phase2_planner=Phase2Spy(),
            task_agent=task_agent,
            clock=lambda: 2.6,
        )

        engine.plan(self.day, request_started_at=0.0)

        self.assertEqual(task_agent.calls, 0)
        self.assertEqual(task_agent.reconcile_calls, 0)

    def test_emergency_reserve_still_records_current_forecast_inputs(
        self,
    ) -> None:
        telemetry = TelemetryRecorder()
        engine = StrategyEngine(
            phase2_planner=Phase2Spy(),
            telemetry=telemetry,
            clock=lambda: 2.6,
        )

        engine.plan(self.night, request_started_at=0.0)

        sample = telemetry.snapshot()[-1]
        self.assertEqual(sample.performance_segment, 'night_alive')
        self.assertTrue(sample.forecast_observation_signature)
        self.assertTrue(sample.forecast_input_changed)
        self.assertFalse(sample.forecast_input_current)
        self.assertGreater(sample.forecast_observed_station_hp, 0)

    def test_emergency_reserve_defers_success_feedback_until_retry(self) -> None:
        engine = StrategyEngine(
            phase2_planner=Phase2Spy(),
            clock=lambda: 2.6,
        )
        task_text = 'Return the sum of 20 and 22.'
        prompted = replace(
            self.day,
            time=TurnTime.from_round(1),
            phase_task=task_text,
        )
        answered = replace(
            self.day,
            time=TurnTime.from_round(2),
            phase_task=task_text,
            llm_response='42',
        )
        feedback = replace(
            self.day,
            time=TurnTime.from_round(3),
            phase_task='',
            llm_response='',
            last_action_results={103: True},
        )

        engine.plan(prompted)
        engine.plan(answered)
        engine.plan(feedback, request_started_at=0.0)

        session = engine._sessions.get(self.day.our.team_id)
        self.assertEqual(session.task_state.sops, ())

        engine.plan(feedback)

        session = engine._sessions.get(self.day.our.team_id)
        self.assertEqual(session.task_state.sops[0].answer, '42')

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

    def test_phase_2_exception_falls_back_to_empty_legal_decision(self) -> None:
        def broken(
            _: Observation,
            *,
            intent: StrategicIntent = DEFAULT_STRATEGIC_INTENT,
        ) -> Decision:
            del intent
            raise ValueError("phase 2 failure")

        engine = StrategyEngine(phase2_planner=broken)

        decision = engine.plan(self.day)

        self.assertEqual(decision, Decision())

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
