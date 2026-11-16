"""regime_value 领域测试：时间变化混杂与多阶段策略目标、决策时可见状态与
吸收状态、序贯支持不足与估值适用范围、策略价值与完整历程复核。"""
import copy
import random
import tempfile
import unittest

from app.catalog import local_packet
from app.engine import analyze
from app.store import ResultStore


def make_packet(roster, episodes, outcomes, rewards=None, stages=(0, 1), week=8,
                revision="regime-test"):
    rewards = rewards if rewards is not None else [
        {"learner": row["learner"], "after_stage": row["stage"],
         "observed_time": row["stage"] + 0.5, "reward": 50.0, "missing_reward": False}
        for row in episodes]
    return {"revision": revision, "synthetic": True,
            "roster": [{"learner": learner, "site": 0, "initial_need": need}
                       for learner, need in roster],
            "episodes": episodes, "rewards": rewards, "outcomes": outcomes,
            "decision_schedule": {"stages": list(stages), "outcome_week": week}}


def episode(learner, stage, need, action, probabilities, terminal=None, attendance=None):
    return {"learner": learner, "stage": stage,
            "observed_at_decision": {"need": need,
                                     "prior_attendance": 2 * stage if attendance is None else attendance},
            "action": action, "action_probabilities": probabilities,
            "terminal_reason": terminal}


ALWAYS_PHONE = {"name": "always-phone",
                "rules": [{"stage": 0, "action": "phone"}, {"stage": 1, "action": "phone"}]}


class ConfoundingTests(unittest.TestCase):
    """时间变化混杂与多阶段策略目标。"""

    def confounded_packet(self):
        # 阶段 1 的 need 由阶段 0 动作决定；结局同时依赖两阶段动作。
        episodes = [
            episode("p0", 0, 0, "phone", {"none": 0.5, "phone": 0.5}),
            episode("p0", 1, 0, "phone", {"none": 0.5, "phone": 0.5}, terminal="completed"),
            episode("p1", 0, 0, "phone", {"none": 0.5, "phone": 0.5}),
            episode("p1", 1, 0, "none", {"none": 0.5, "phone": 0.5}, terminal="completed"),
            episode("p2", 0, 0, "none", {"none": 0.5, "phone": 0.5}),
            episode("p2", 1, 1, "phone", {"none": 0.5, "phone": 0.5}, terminal="completed"),
            episode("p3", 0, 0, "none", {"none": 0.5, "phone": 0.5}),
            episode("p3", 1, 1, "none", {"none": 0.5, "phone": 0.5}, terminal="completed"),
        ]
        outcomes = [{"learner": "p0", "week": 8, "score": 20.0},
                    {"learner": "p1", "week": 8, "score": 10.0},
                    {"learner": "p2", "week": 8, "score": 15.0},
                    {"learner": "p3", "week": 8, "score": 5.0}]
        return make_packet([("p0", 0), ("p1", 0), ("p2", 0), ("p3", 0)],
                           episodes, outcomes)

    def test_sequential_weighting_recovers_policy_value_not_final_support(self):
        result = analyze(self.confounded_packet(), "regime_value",
                         {"policy": ALWAYS_PHONE, "bootstrap_reps": 0})["result"]
        estimate = result["regime_estimate"]
        self.assertEqual(estimate["status"], "ok")
        # 仅 p0 全程与规则一致，权重 1/(0.5*0.5)=4，HT 分量 = 4*20/4 = 20。
        self.assertAlmostEqual(estimate["estimators"]["ipw_ht_component"], 20.0, places=9)
        self.assertAlmostEqual(estimate["estimators"]["ipw_ht_observed_share"], 1.0, places=9)
        # 业务上“最终拿过哪种支持”的粗比较：(20+15)/2=17.5，不等于策略价值。
        naive_final_phone = (20.0 + 15.0) / 2
        self.assertNotAlmostEqual(estimate["estimators"]["ipw_ht_component"],
                                  naive_final_phone, places=6)
        self.assertEqual(result["identification"]["status"], "identified")
        self.assertEqual(result["identification"]["consistent_learners"], 1)

    def test_aipw_telescopes_under_deterministic_matching_policy(self):
        # 行为策略确定（倾向 0/1）且规则与之完全一致时，AIPW 递归恒等于观测均值。
        episodes = [
            episode("p0", 0, 0, "phone", {"none": 0.0, "phone": 1.0}),
            episode("p0", 1, 0, "phone", {"none": 0.0, "phone": 1.0}, terminal="completed"),
            episode("p1", 0, 0, "phone", {"none": 0.0, "phone": 1.0}),
            episode("p1", 1, 0, "phone", {"none": 0.0, "phone": 1.0}, terminal="completed"),
            episode("p2", 0, 1, "none", {"none": 1.0, "phone": 0.0}),
            episode("p2", 1, 1, "none", {"none": 1.0, "phone": 0.0}, terminal="completed"),
            episode("p3", 0, 1, "none", {"none": 1.0, "phone": 0.0}),
            episode("p3", 1, 1, "none", {"none": 1.0, "phone": 0.0}, terminal="completed"),
        ]
        outcomes = [{"learner": "p0", "week": 8, "score": 20.0},
                    {"learner": "p1", "week": 8, "score": 10.0},
                    {"learner": "p2", "week": 8, "score": 15.0},
                    {"learner": "p3", "week": 8, "score": 5.0}]
        packet = make_packet([("p0", 0), ("p1", 0), ("p2", 1), ("p3", 1)],
                             episodes, outcomes)
        policy = {"name": "mirror", "rules": [
            {"stage": 0, "when": {"initial_need": {"eq": 0}}, "action": "phone"},
            {"stage": 1, "when": {"need": {"eq": 0}}, "action": "phone"},
            {"stage": 0, "action": "none"}, {"stage": 1, "action": "none"}]}
        estimate = analyze(packet, "regime_value",
                           {"policy": policy, "bootstrap_reps": 0})["result"]["regime_estimate"]
        for key in ("aipw_component", "ipw_ht_component", "ipw_hajek_component"):
            self.assertAlmostEqual(estimate["estimators"][key], 12.5, places=9)


class DecisionTimeStateTests(unittest.TestCase):
    """决策时可见状态与吸收状态。"""

    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_policy_cannot_reference_future_or_outcome_fields(self):
        for field in ("reward", "score", "terminal_reason", "outcome"):
            policy = {"rules": [{"stage": 0, "when": {field: {"gte": 1}}, "action": "peer"},
                                {"stage": 0, "action": "phone"},
                                {"stage": 1, "action": "phone"},
                                {"stage": 2, "action": "phone"}]}
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze(self.packet, "regime_value", {"policy": policy})

    def test_policy_requires_fallback_and_known_action(self):
        with self.assertRaises(ValueError):
            analyze(self.packet, "regime_value",
                    {"policy": {"rules": [{"stage": 0, "action": "phone"},
                                          {"stage": 1, "action": "phone"}]}})
        with self.assertRaises(ValueError):
            analyze(self.packet, "regime_value",
                    {"policy": {"rules": [{"stage": 0, "action": "sms"},
                                          {"stage": 1, "action": "phone"},
                                          {"stage": 2, "action": "phone"}]}})

    def test_absorbing_completion_keeps_full_initial_population(self):
        packet = local_packet("data/cases/boundary-1.json")
        result = analyze(packet, "regime_value", {"bootstrap_reps": 0})["result"]
        self.assertEqual(result["data_used"]["included"], 150)
        self.assertEqual(result["data_used"]["journey_endings"], {"completed": 150})
        self.assertEqual(result["data_used"]["outcome"]["missing"], 27)
        # 全部学员在首次支持后完成：一致性只看阶段 0；Hajek 分量等于一致学员
        # 以 1/倾向 加权的均值（boundary-1 中 phone 倾向非常数）。
        consistent = [row for row in packet["episodes"] if row["action"] == "phone"]
        scores = {row["learner"]: row["score"] for row in packet["outcomes"]}
        weights = [1.0 / row["action_probabilities"]["phone"] for row in consistent]
        y_tilde = [scores[row["learner"]] or 0.0 for row in consistent]
        expected = sum(w * y for w, y in zip(weights, y_tilde)) / sum(weights)
        self.assertAlmostEqual(result["regime_estimate"]["estimators"]["ipw_hajek_component"],
                               expected, places=9)

    def test_incomplete_and_anchorless_journeys_excluded_with_reasons(self):
        episodes = [
            episode("p0", 0, 0, "phone", {"none": 0.5, "phone": 0.5}, terminal="completed"),
            episode("p1", 0, 0, "phone", {"none": 0.5, "phone": 0.5}),
            episode("p1", 1, 0, "phone", {"none": 0.5, "phone": 0.5}),  # 未到下一阶段
            episode("p2", 1, 0, "phone", {"none": 0.5, "phone": 0.5}, terminal="completed"),
        ]
        outcomes = [{"learner": learner, "week": 8, "score": 50.0}
                    for learner in ("p0", "p1", "p2", "p3")]
        packet = make_packet([("p0", 0), ("p1", 0), ("p2", 0), ("p3", 0)],
                             episodes, outcomes, stages=(0, 1, 2))
        result = analyze(packet, "regime_value", {"bootstrap_reps": 0})["result"]
        used = result["data_used"]
        self.assertEqual(used["included"], 1)
        self.assertEqual({entry["reason"]: entry["learners"] for entry in used["excluded"]},
                         {"无任何决策记录，无法核对规则一致性": 1,
                          "历程在未到最后决策阶段前中断且未标记终态（未到下一阶段）": 1,
                          "首个决策记录不在第一决策阶段": 1})
        self.assertEqual(used["journey_endings"],
                         {"completed": 1, "never_decided": 1, "not_anchored": 1,
                          "ongoing_incomplete": 1})
        self.assertIsNotNone(result["estimand"]["exclusions_note"])


class SupportBoundaryTests(unittest.TestCase):
    """序贯支持不足与估值适用范围。"""

    def test_never_supported_action_is_boundary_not_benefit(self):
        packet = local_packet("data/cases/boundary-2.json")
        policy = {"name": "always-peer",
                  "rules": [{"stage": stage, "action": "peer"} for stage in (0, 1, 2)]}
        result = analyze(packet, "regime_value",
                         {"policy": policy, "bootstrap_reps": 0})["result"]
        self.assertEqual(result["identification"]["status"], "not_identified")
        estimate = result["regime_estimate"]
        self.assertEqual(estimate["status"], "not_identified")
        self.assertNotIn("outcome_component", estimate)
        self.assertTrue(result["identification"]["applicability_boundary"])
        self.assertEqual(result["comparison"]["status"], "not_available")

    def test_partial_coverage_is_flagged_not_extrapolated(self):
        packet = local_packet("data/cases/boundary-2.json")
        policy = {"name": "peer-for-high-need", "rules": [
            {"stage": 0, "when": {"need": {"gte": 2}}, "action": "peer"},
            {"stage": 0, "action": "phone"}, {"stage": 1, "action": "phone"},
            {"stage": 2, "action": "phone"}]}
        result = analyze(packet, "regime_value",
                         {"policy": policy, "bootstrap_reps": 0})["result"]
        self.assertEqual(result["identification"]["status"], "partially_identified")
        self.assertLess(result["identification"]["supported_share"], 1.0)
        self.assertTrue(result["identification"]["applicability_boundary"])
        self.assertEqual(result["regime_estimate"]["status"], "ok")
        self.assertIn("partial_coverage",
                      {entry["kind"] for entry in result["diagnostics"]})

    def test_all_outcomes_missing_is_not_estimable(self):
        packet = local_packet("data/cases/boundary-3.json")
        result = analyze(packet, "regime_value", {"bootstrap_reps": 0})["result"]
        estimate = result["regime_estimate"]
        self.assertEqual(estimate["status"], "not_estimable")
        self.assertIsNone(estimate["conditional_mean_if_observed"])
        self.assertIsNone(estimate["value_bounds"])
        self.assertEqual(result["current_process"]["outcome_observed_share"], 0.0)
        # 过程量仍在但不得充当终期收益。
        self.assertTrue(all(stage["observed"] > 0
                            for stage in result["process_measures"]["stages"]))
        self.assertEqual(result["comparison"]["status"], "not_available")


class JourneyAuditTests(unittest.TestCase):
    """策略价值与完整历程复核。"""

    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_observation_order_does_not_change_result(self):
        shuffled = copy.deepcopy(self.packet)
        random.Random(20261116).shuffle(shuffled["episodes"])
        random.Random(7).shuffle(shuffled["rewards"])
        random.Random(3).shuffle(shuffled["outcomes"])
        baseline = analyze(self.packet, "regime_value", {"bootstrap_reps": 100})
        reordered = analyze(shuffled, "regime_value", {"bootstrap_reps": 100})
        self.assertEqual(baseline["result"], reordered["result"])

    def test_result_is_deterministic(self):
        first = analyze(self.packet, "regime_value")
        second = analyze(self.packet, "regime_value")
        self.assertEqual(first, second)

    def test_rule_audit_covers_every_initial_learner_and_stores(self):
        policy = {"name": "need-adaptive", "rules": [
            {"stage": 0, "when": {"need": {"gte": 2}}, "action": "peer",
             "note": "高需求先配同伴"},
            {"stage": 0, "action": "phone", "note": "其余电话提醒"},
            {"stage": 1, "when": {"previous_action": {"eq": "phone"}}, "action": "peer",
             "note": "电话后仍未改善则转同伴"},
            {"stage": 1, "action": "none"},
            {"stage": 2, "action": "none"}]}
        envelope = analyze(self.packet, "regime_value", {"policy": policy})
        result = envelope["result"]
        self.assertEqual(result["policy"]["name"], "need-adaptive")
        self.assertEqual(result["policy"]["provenance"], "caller")
        notes = [rule.get("note") for rule in result["policy"]["rules"]]
        self.assertIn("高需求先配同伴", notes)
        learners = result["rule_audit"]["learners"]
        self.assertEqual(len(learners), 150)
        for record in learners:
            self.assertEqual(len(record["prescribed"]), len(record["observed"]))
        usage = result["rule_audit"]["usage"]
        per_stage = {}
        for entry in usage:
            per_stage[entry["stage"]] = per_stage.get(entry["stage"], 0) + entry["prescribed_learners"]
        self.assertEqual(per_stage, {0: 150, 1: 123, 2: 99})
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory)
            identifier = store.put(envelope)
            self.assertEqual(store.get(identifier), envelope)

    def test_comparison_and_uncertainty_blocks(self):
        result = analyze(self.packet, "regime_value")["result"]
        self.assertEqual(result["comparison"]["status"], "ok")
        uncertainty = result["uncertainty"]
        self.assertEqual(uncertainty["seed"], 20261116)
        self.assertEqual(uncertainty["reps"], 500)
        for key in ("regime_component_ci", "current_component_ci", "difference_component_ci"):
            low, high = uncertainty[key]
            self.assertLessEqual(low, high)

    def test_score_range_override_is_recorded(self):
        result = analyze(self.packet, "regime_value",
                         {"score_range": [0, 100], "bootstrap_reps": 0})["result"]
        bounds = result["regime_estimate"]["value_bounds"]
        self.assertIn("options.score_range", bounds["basis"])
        component = result["regime_estimate"]["outcome_component"]
        share = result["regime_estimate"]["outcome_observed_share"]
        self.assertAlmostEqual(bounds["low"], component, places=9)
        self.assertAlmostEqual(bounds["high"], component + (1 - share) * 100, places=9)

    def test_invalid_options_rejected(self):
        for options in ({"support_threshold": 2}, {"support_threshold": 0},
                        {"bootstrap_reps": -1}, {"bootstrap_reps": 1.5},
                        {"bootstrap_seed": "x"}, {"score_range": [10, 5]},
                        {"score_range": [0]}, {"unknown_key": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                analyze(self.packet, "regime_value", options)


if __name__ == "__main__":
    unittest.main()
