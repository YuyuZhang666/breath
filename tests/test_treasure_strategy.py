import unittest
from dataclasses import replace

from future_war_agent.decision.actions import Action, ActionKind
from future_war_agent.decision.decision import Decision
from future_war_agent.protocol.models import Position, ShopItem, WorldNews, Zone
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.strategy.policy import (
    RuleFeatureFlags,
    StrategyProfile,
    intent_for_profile,
)
from future_war_agent.strategy.treasure import (
    DEFAULT_TREASURE_REWARD,
    TreasureAgent,
    TreasureCandidate,
    TreasureReward,
    TreasureState,
    parse_treasure_clues,
)
from future_war_agent.strategy.session import SessionStore
from tests.strategy_helpers import observation, unit


SCORE_INTENT = intent_for_profile(StrategyProfile.SCORE, 'test')


def candidate(**changes: object) -> TreasureCandidate:
    base = TreasureCandidate(
        position=Position(2, 2),
        items=('StarSand',),
        opening_days=(1,),
        confidence=0.9,
        evidence_revision=1,
        support_evidence=('fixture',),
    )
    return replace(base, **changes)


class TreasureStrategyTests(unittest.TestCase):
    def test_observed_reward_is_not_labeled_official(self) -> None:
        self.assertEqual(DEFAULT_TREASURE_REWARD.score, 200)
        self.assertEqual(DEFAULT_TREASURE_REWARD.gold, 100)
        self.assertEqual(DEFAULT_TREASURE_REWARD.source, 'battle_log')
        self.assertLessEqual(DEFAULT_TREASURE_REWARD.confidence, 1.0)

        learned = TreasureAgent(
            reward=TreasureReward(180, 75, 'battle_log', 0.6),
        ).reconcile(observation())
        self.assertEqual((learned.reward.score, learned.reward.gold), (180, 75))
        self.assertEqual(learned.reward.source, 'battle_log')

    def test_structured_folk_legend_is_parsed_without_guessing_prose(self) -> None:
        clues = parse_treasure_clues(
            '{"treasure":{"x":2,"y":3,"items":["StarSand"],'
            '"days":[1],"confidence":0.9}}',
            width=10,
            height=10,
        )

        self.assertEqual(len(clues), 1)
        self.assertEqual(clues[0].position, Position(2, 3))
        self.assertEqual(parse_treasure_clues('maybe somewhere west', 10, 10), ())

    def test_result_codes_update_only_pending_candidate(self) -> None:
        base = TreasureState(
            candidates=(candidate(),),
            pending_candidate_key=candidate().key,
            pending_round=1,
            pending_day=1,
        )
        agent = TreasureAgent()

        wrong_place = agent.reconcile(
            observation(round_no=2, last_summon_treasure_result=2),
            base,
        )
        wrong_items = agent.reconcile(
            observation(round_no=2, last_summon_treasure_result=3),
            base,
        )
        depleted = agent.reconcile(
            observation(round_no=2, last_summon_treasure_result=4),
            base,
        )
        ignored = agent.reconcile(
            observation(round_no=2, last_summon_treasure_result=0),
            base,
        )

        self.assertEqual(wrong_place.candidates[0].invalid_days, frozenset())
        self.assertIn((2, 2, 1), wrong_place.failed_position_times)
        self.assertEqual(wrong_items.candidates[0].item_conflicts, 1)
        self.assertIn(('starsand',), wrong_items.failed_item_multisets)
        self.assertTrue(depleted.candidates[0].depleted)
        self.assertEqual(ignored.candidates, base.candidates)
        self.assertEqual(ignored.pending_candidate_key, '')

    def test_explicit_natural_language_clue_is_parsed_conservatively(
        self,
    ) -> None:
        clues = parse_treasure_clues(
            'Coordinate: (2, 3); items: StarSand, MoonDust; '
            'day 2; confidence: 0.9',
            width=10,
            height=10,
        )

        self.assertEqual(len(clues), 1)
        self.assertEqual(clues[0].position, Position(2, 3))
        self.assertEqual(clues[0].items, ('StarSand', 'MoonDust'))
        self.assertEqual(clues[0].opening_days, (2,))
        self.assertEqual(parse_treasure_clues('somewhere west', 10, 10), ())

    def test_result_two_blocks_only_exact_attempt_turn(self) -> None:
        attempted = candidate(attempt_count=1, last_attempt_evidence_revision=1)
        pending = TreasureState(
            candidates=(attempted,),
            pending_candidate_key=attempted.key,
            pending_round=1,
            pending_day=1,
            evidence_revision=1,
            attempt_count=1,
            last_attempt_evidence_revision=1,
        )
        failed = TreasureAgent().reconcile(
            observation(round_no=2, last_summon_treasure_result=2),
            pending,
        )
        refreshed_candidate = replace(
            failed.candidates[0],
            confidence=0.9,
            evidence_revision=2,
        )
        refreshed = replace(
            failed,
            candidates=(refreshed_candidate,),
            evidence_revision=2,
        )
        enabled = intent_for_profile(
            StrategyProfile.SCORE,
            'verified retry',
            feature_flags=RuleFeatureFlags(enable_treasure_retry=True),
        )
        observed = observation(
            round_no=3,
            our_units=(
                unit(
                    2,
                    2,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            enabled,
            previous_state=refreshed,
        )

        self.assertEqual(
            result.decision.commands[2].kind,
            ActionKind.SUMMON_TREASURE,
        )

    def test_result_three_blocks_item_multiset_across_new_candidates(
        self,
    ) -> None:
        attempted = candidate(
            items=('MoonDust', 'StarSand'),
            attempt_count=1,
            last_attempt_evidence_revision=1,
        )
        pending = TreasureState(
            candidates=(attempted,),
            pending_candidate_key=attempted.key,
            pending_round=1,
            pending_day=1,
            evidence_revision=1,
            attempt_count=1,
            last_attempt_evidence_revision=1,
        )
        failed = TreasureAgent().reconcile(
            observation(round_no=2, last_summon_treasure_result=3),
            pending,
        )
        alternate = candidate(
            position=Position(3, 2),
            items=('StarSand', 'MoonDust'),
            evidence_revision=2,
        )
        refreshed = replace(
            failed,
            candidates=(alternate, *failed.candidates),
            evidence_revision=2,
        )
        enabled = intent_for_profile(
            StrategyProfile.SCORE,
            'verified retry',
            feature_flags=RuleFeatureFlags(enable_treasure_retry=True),
        )
        observed = observation(
            round_no=3,
            our_units=(
                unit(
                    2,
                    3,
                    1,
                    'pioneer',
                    backpack=('StarSand', 'MoonDust'),
                    backpack_capacity=10,
                ),
            ),
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            enabled,
            previous_state=refreshed,
        )

        self.assertNotIn(2, result.decision.commands)

    def test_saturated_negative_evidence_disables_automatic_attempts(
        self,
    ) -> None:
        observed = observation(
            our_units=(
                unit(
                    2,
                    2,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )
        state = TreasureState(
            candidates=(candidate(),),
            negative_evidence_saturated=True,
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)

    def test_initial_high_confidence_attempt_is_emitted_once(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    2,
                    2,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )
        state = TreasureState(candidates=(candidate(),))

        first = TreasureAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )
        second = TreasureAgent().apply(
            observation(
                round_no=2,
                our_units=(
                    unit(
                        2,
                        2,
                        1,
                        'pioneer',
                        backpack=('StarSand',),
                        backpack_capacity=10,
                    ),
                ),
            ),
            Decision(),
            SCORE_INTENT,
            previous_state=first.state,
        )

        self.assertEqual(first.decision.commands[2].kind, ActionKind.SUMMON_TREASURE)
        self.assertNotIn(2, second.decision.commands)

    def test_retry_requires_flag_and_new_evidence(self) -> None:
        observed = observation(
            round_no=3,
            our_units=(
                unit(
                    2,
                    2,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )
        attempted = candidate(attempt_count=1, last_attempt_evidence_revision=1)
        enabled = intent_for_profile(
            StrategyProfile.SCORE,
            'verified retry',
            feature_flags=RuleFeatureFlags(enable_treasure_retry=True),
        )

        disabled_result = TreasureAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=TreasureState(
                candidates=(attempted,),
                evidence_revision=1,
                attempt_count=1,
                last_attempt_evidence_revision=1,
            ),
        )
        stale_result = TreasureAgent().apply(
            observed,
            Decision(),
            enabled,
            previous_state=TreasureState(
                candidates=(attempted,),
                evidence_revision=1,
                attempt_count=1,
                last_attempt_evidence_revision=1,
            ),
        )
        fresh_result = TreasureAgent().apply(
            observed,
            Decision(),
            enabled,
            previous_state=TreasureState(
                candidates=(replace(attempted, evidence_revision=2),),
                evidence_revision=2,
                attempt_count=1,
                last_attempt_evidence_revision=1,
            ),
        )

        self.assertNotIn(2, disabled_result.decision.commands)
        self.assertNotIn(2, stale_result.decision.commands)
        self.assertEqual(
            fresh_result.decision.commands[2].kind,
            ActionKind.SUMMON_TREASURE,
        )

    def test_safe_gold_reserve_blocks_bundle_purchase(self) -> None:
        observed = observation(
            gold=120,
            our_units=(unit(2, 1, 1, 'pioneer', backpack_capacity=10),),
            vendor_shop=(ShopItem('StarSand', 90), ShopItem('MoonDust', 30)),
        )
        state = TreasureState(
            candidates=(candidate(items=('StarSand', 'MoonDust')),),
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)

    def test_semantically_identical_news_does_not_create_new_evidence(self) -> None:
        compact = '{"treasure":{"x":2,"y":2,"items":["StarSand"],"days":[1],"confidence":0.9}}'
        spaced = '{ "treasure": { "confidence": 0.9, "days": [1], "items": ["StarSand"], "y": 2, "x": 2 } }'
        agent = TreasureAgent()

        first = agent.reconcile(
            observation(world_news=WorldNews(folk_legends=compact))
        )
        second = agent.reconcile(
            observation(round_no=2, world_news=WorldNews(folk_legends=spaced)),
            first,
        )

        self.assertEqual(second.evidence_revision, first.evidence_revision)
        self.assertEqual(second.candidates, first.candidates)

    def test_preexisting_alternate_candidate_is_not_new_retry_evidence(self) -> None:
        attempted = candidate(attempt_count=1, last_attempt_evidence_revision=2)
        alternate = candidate(
            position=Position(3, 2),
            evidence_revision=2,
        )
        state = TreasureState(
            candidates=(attempted, alternate),
            pending_candidate_key=attempted.key,
            pending_round=1,
            pending_day=1,
            evidence_revision=2,
            attempt_count=1,
            last_attempt_evidence_revision=2,
        )
        enabled = intent_for_profile(
            StrategyProfile.SCORE,
            'verified retry',
            feature_flags=RuleFeatureFlags(enable_treasure_retry=True),
        )
        observed = observation(
            round_no=2,
            last_summon_treasure_result=2,
            our_units=(
                unit(
                    2,
                    3,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            enabled,
            previous_state=state,
        )

        self.assertNotIn(2, result.decision.commands)
        self.assertEqual(result.state.last_failure_code, 2)

    def test_zero_reward_does_not_fund_item_purchase(self) -> None:
        observed = observation(
            gold=200,
            zones=(Zone(Position(2, 1), 'vendor'),),
            our_units=(unit(2, 1, 1, 'pioneer', backpack_capacity=10),),
            vendor_shop=(ShopItem('StarSand', 50),),
        )
        state = TreasureState(
            reward=TreasureReward(0, 0, 'disabled', 1.0),
            candidates=(candidate(),),
        )

        result = TreasureAgent(
            reward=state.reward,
        ).apply(observed, Decision(), SCORE_INTENT, previous_state=state)

        self.assertNotIn(2, result.decision.commands)

    def test_equal_cost_shop_paths_are_deterministic(self) -> None:
        observed = observation(
            gold=200,
            zones=(
                Zone(Position(1, 3), 'vendor'),
                Zone(Position(5, 3), 'vendor'),
            ),
            our_units=(unit(2, 3, 3, 'pioneer', backpack_capacity=10),),
            vendor_shop=(ShopItem('StarSand', 50),),
        )

        result = TreasureAgent().apply(
            observed,
            Decision(),
            SCORE_INTENT,
            previous_state=TreasureState(candidates=(candidate(),)),
        )

        self.assertIn(2, result.decision.commands)
        self.assertIn(
            result.decision.commands[2].kind,
            {ActionKind.MOVE, ActionKind.BUY},
        )

    def test_treasure_replaces_only_ordinary_pioneer_move(self) -> None:
        observed = observation(
            our_units=(
                unit(
                    2,
                    1,
                    1,
                    'pioneer',
                    backpack=('StarSand',),
                    backpack_capacity=10,
                ),
            ),
        )
        base = Decision(commands={2: Action.move(Position(1, 2))})
        state = TreasureState(candidates=(candidate(),))

        replaced = TreasureAgent().apply(
            observed,
            base,
            SCORE_INTENT,
            previous_state=state,
        )
        task_protected = TreasureAgent().apply(
            observed,
            base,
            SCORE_INTENT,
            previous_state=state,
            task_active=True,
        )

        self.assertEqual(
            replaced.decision.commands[2].kind,
            ActionKind.SUMMON_TREASURE,
        )
        self.assertEqual(task_protected.decision.commands[2].kind, ActionKind.MOVE)

    def test_engine_retains_custom_reward_and_state_across_round_gap(self) -> None:
        sessions = SessionStore()
        reward = TreasureReward(17, 23, 'battle_log', 0.5)
        engine = StrategyEngine(
            phase2_planner=lambda *_args, **_kwargs: Decision(),
            treasure_agent=TreasureAgent(reward=reward),
            session_store=sessions,
        )
        units = (
            unit(10, 5, 5, 'station', health=1000, level=1),
            unit(11, 3, 3, 'gatling', level=1),
            unit(12, 4, 3, 'railgun', level=1),
            unit(13, 5, 3, 'rocket', level=1),
            unit(14, 1, 1, 'pioneer'),
        )
        rumor = WorldNews(
            folk_legends='{"treasure":{"x":2,"y":2,"items":["StarSand"],"days":[2],"confidence":0.9}}'
        )

        engine.plan(observation(round_no=1, our_units=units, world_news=rumor))
        engine.plan(observation(round_no=3, our_units=units))
        retained = sessions.get('team').treasure_state

        self.assertEqual(retained.reward, reward)
        self.assertEqual(len(retained.candidates), 1)


if __name__ == '__main__':
    unittest.main()
