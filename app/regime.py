"""随学习状态调整的支持规则：regime_value 策略价值评估。

在同一初始学员群体上，用事件记录中的决策时状态、动作倾向与固定期结局，
评估一套可随学习状态调整的支持规则，并与现行流程对照：

- 规则只能引用决策时可见字段，结构上阻止把未来结局带回前面的决定；
- completed / withdrawn / lost / 未到下一阶段分类处理，失访结局不以零或
  过程测量顶替，而以可识别边界表示；
- 历史几乎没人走过的路径不硬算收益，给出适用边界与原因；
- 规则、逐学员判定与估计依据全部写入可保存结果，供业务复核。

统计方法：以记录的行为倾向（action_probabilities）为权重的序贯逆概率加权
（IPW-Horvitz-Thompson 与 Hajek），以及向后递归 Q 回归的增广 IPW（AIPW）。
倾向来自记录而非模型拟合；Q 回归只用决策时可见特征。缺测结局不做点估计，
用分数范围给出全人群均值的可识别边界。
"""

from collections import Counter, defaultdict

import numpy as np

from app.protocol import finite
from app.trajectories import learner_paths

CONDITION_FIELDS = ("need", "prior_attendance", "initial_need", "site", "previous_action")
OPERATORS = ("eq", "ne", "gte", "lte", "in")
OPTION_KEYS = ("policy", "support_threshold", "score_range", "bootstrap_reps", "bootstrap_seed")
DEFAULT_SUPPORT_THRESHOLD = 0.05
DEFAULT_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 20261116
MAX_BOOTSTRAP_REPS = 5000
DISAGREEMENT_TOLERANCE = 1.0
EXAMPLE_STATES_PER_BOUNDARY = 3


def regime_value(packet, options):
    """评估一套随状态调整的支持规则，并与现行流程在同一初始群体上对照。"""
    config = _check_options(options)
    schedule = packet["decision_schedule"]
    stages = sorted(schedule["stages"])
    if not stages:
        raise ValueError("decision schedule has no stages")
    week = schedule["outcome_week"]
    actions = sorted({action for row in packet["episodes"] for action in row["action_probabilities"]})
    policy = _compile_policy(options.get("policy"), stages, actions)
    journeys = _build_journeys(packet, policy, stages, week, actions)

    included = [row for row in journeys if row["included"]]
    excluded = [row for row in journeys if not row["included"]]
    endings = Counter(row["ending"] for row in journeys)
    outcome_observed = sum(1 for row in journeys if row["outcome_observed"])

    score_range, score_range_basis = _score_range(packet, config)
    support, boundary = _support_diagnostics(included, stages, config["support_threshold"])
    identification = _identification(included, support, boundary)

    result = {
        "estimand": {
            "target_population": "roster 中全部初始学员",
            "outcome": f"固定第 {week} 周 score",
            "parameter": "E[规则下固定期结局]（同一初始群体）；缺测结局以可识别边界表示，不作点填补",
            "exclusions_note": _exclusions_note(excluded),
        },
        "policy": policy,
        "data_used": {
            "initial_roster": len(journeys),
            "included": len(included),
            "excluded": [{"reason": reason, "learners": count}
                         for reason, count in sorted(Counter(row["exclusion_reason"] for row in excluded).items())],
            "journey_endings": dict(sorted(endings.items())),
            "episodes": sum(len(row["episodes"]) for row in journeys),
            "outcome": {"week": week, "observed": outcome_observed,
                        "missing": len(journeys) - outcome_observed},
        },
        "identification": identification,
        "current_process": _current_process(included, score_range, score_range_basis),
        "process_measures": _process_measures(packet, stages),
        "assumptions": _assumptions(score_range, score_range_basis, identification["status"]),
    }

    if identification["status"] == "not_identified":
        result["regime_estimate"] = {"status": "not_identified",
                                     "reasons": identification["reasons"]}
        result["comparison"] = {"status": "not_available",
                                "reason": "规则价值不可识别，不与现行流程作数值对照"}
        result["uncertainty"] = {"status": "skipped", "reason": "规则价值不可识别"}
    else:
        estimate = _estimate(included, stages, score_range, score_range_basis)
        result["regime_estimate"] = estimate
        result["diagnostics"] = _diagnostics(included, estimate, identification)
        if estimate["status"] != "ok":
            result["comparison"] = {"status": "not_available",
                                    "reason": "规则价值不可点估计（结局全部缺测），不与现行流程作数值对照"}
            result["uncertainty"] = {"status": "skipped", "reason": "结局全部缺测"}
        else:
            result["comparison"] = _comparison(estimate, result["current_process"])
            result["uncertainty"] = _uncertainty(included, stages, config)
    result.setdefault("diagnostics", _diagnostics(included, result.get("regime_estimate"), identification))
    result["rule_audit"] = _rule_audit(journeys, support)
    result["interpretation"] = _interpretation(result)
    return result


# ---------------------------------------------------------------- 输入校验

def _check_options(options):
    unknown = sorted(set(options) - set(OPTION_KEYS))
    if unknown:
        raise ValueError(f"unknown options: {', '.join(unknown)}")
    threshold = finite(options.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD), "support_threshold")
    if not 0 < threshold < 1:
        raise ValueError("support_threshold must be in (0, 1)")
    reps = options.get("bootstrap_reps", DEFAULT_BOOTSTRAP_REPS)
    if isinstance(reps, bool) or not isinstance(reps, int) or not 0 <= reps <= MAX_BOOTSTRAP_REPS:
        raise ValueError(f"bootstrap_reps must be an integer in [0, {MAX_BOOTSTRAP_REPS}]")
    seed = options.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    score_range = options.get("score_range")
    if score_range is not None:
        if not isinstance(score_range, (list, tuple)) or len(score_range) != 2:
            raise ValueError("score_range must be [low, high]")
        low, high = finite(score_range[0], "score_range low"), finite(score_range[1], "score_range high")
        if not low < high:
            raise ValueError("score_range low must be below high")
        score_range = (float(low), float(high))
    return {"support_threshold": float(threshold), "bootstrap_reps": reps,
            "bootstrap_seed": seed, "score_range": score_range}


def _compile_policy(spec, stages, actions):
    if spec is None:
        return {"name": "default-always-phone", "provenance": "default",
                "rules": [{"index": index, "stage": stage, "action": "phone",
                           "note": "默认基线情景：各阶段均电话提醒；业务规则请经 options.policy 提供"}
                          for index, stage in enumerate(stages)]}
    if not isinstance(spec, dict):
        raise ValueError("policy must be an object")
    unknown = sorted(set(spec) - {"name", "rules"})
    if unknown:
        raise ValueError(f"unknown policy keys: {', '.join(unknown)}")
    name = spec.get("name", "caller-policy")
    if not isinstance(name, str):
        raise ValueError("policy.name must be a string")
    rules = spec.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("policy.rules must be a non-empty list")
    compiled, fallback = [], set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rule {index}: must be an object")
        unknown = sorted(set(rule) - {"stage", "action", "when", "note"})
        if unknown:
            raise ValueError(f"rule {index}: unknown keys {', '.join(unknown)}")
        stage = rule.get("stage")
        if isinstance(stage, bool) or not isinstance(stage, int) or stage not in stages:
            raise ValueError(f"rule {index}: stage must be one of {stages}")
        action = rule.get("action")
        if action not in actions:
            raise ValueError(f"rule {index}: unknown action {action!r}")
        when = rule.get("when")
        if when is None:
            if stage in fallback:
                raise ValueError(f"rule {index}: multiple fallback rules for stage {stage}")
            fallback.add(stage)
        else:
            _check_when(when, index, actions)
        note = rule.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"rule {index}: note must be a string")
        entry = {"index": index, "stage": stage, "action": action}
        if when is not None:
            entry["when"] = when
        if note is not None:
            entry["note"] = note
        compiled.append(entry)
    missing = [stage for stage in stages if stage not in fallback]
    if missing:
        raise ValueError(f"policy needs a fallback rule (without when) for stages {missing}")
    return {"name": name, "provenance": "caller", "rules": compiled}


def _check_when(when, index, actions):
    if not isinstance(when, dict) or not when:
        raise ValueError(f"rule {index}: when must be a non-empty object")
    for field, conditions in when.items():
        if field not in CONDITION_FIELDS:
            raise ValueError(f"rule {index}: {field!r} 不是决策时可见字段，"
                             f"允许 {', '.join(CONDITION_FIELDS)}")
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError(f"rule {index}: conditions for {field} must be a non-empty object")
        for operator, target in conditions.items():
            if operator not in OPERATORS:
                raise ValueError(f"rule {index}: unknown operator {operator!r}")
            if field == "previous_action":
                if operator in ("gte", "lte"):
                    raise ValueError(f"rule {index}: previous_action only supports eq/ne/in")
                values = target if operator == "in" else [target]
                if operator == "in" and not isinstance(target, list):
                    raise ValueError(f"rule {index}: in expects a list")
                if not all(isinstance(value, str) and value in actions for value in values):
                    raise ValueError(f"rule {index}: previous_action values must be known actions")
            elif operator == "in":
                if not isinstance(target, list) or not target:
                    raise ValueError(f"rule {index}: in expects a non-empty list")
                for value in target:
                    finite(value, f"rule {index} condition")
            else:
                finite(target, f"rule {index} condition")


def _matches(when, context):
    for field, conditions in when.items():
        value = context.get(field)
        for operator, target in conditions.items():
            if operator == "eq" and not value == target:
                return False
            if operator == "ne" and not value != target:
                return False
            if operator == "gte" and not (value is not None and value >= target):
                return False
            if operator == "lte" and not (value is not None and value <= target):
                return False
            if operator == "in" and not (value is not None and value in target):
                return False
    return True


# ---------------------------------------------------------------- 历程重建

def _build_journeys(packet, policy, stages, week, actions):
    paths = learner_paths(packet)
    outcomes = {}
    for row in packet["outcomes"]:
        if row["learner"] in outcomes:
            raise ValueError("duplicate outcome row")
        if row["week"] != week:
            raise ValueError("outcome week differs from decision schedule")
        if row["score"] is not None:
            finite(row["score"], "score")
        outcomes[row["learner"]] = row["score"]
    roster = {row["learner"]: row for row in packet["roster"]}
    if set(outcomes) != set(roster):
        raise ValueError("final outcome roster differs from initial roster")
    rules_by_stage = defaultdict(list)
    for rule in policy["rules"]:
        rules_by_stage[rule["stage"]].append(rule)
    sites = sorted({row["site"] for row in packet["roster"]})
    feature_spec = {"actions": actions, "sites": sites}

    journeys = []
    for learner in packet["roster"]:
        identity = learner["learner"]
        baseline = {"site": finite(learner["site"], "site"),
                    "initial_need": finite(learner["initial_need"], "initial_need")}
        rows = paths.get(identity, [])
        episodes = []
        previous_action = None
        for row in rows:
            if row["stage"] not in rules_by_stage:
                raise ValueError("episode stage outside decision schedule")
            observed = row["observed_at_decision"]
            context = {"need": finite(observed["need"], "need"),
                       "prior_attendance": finite(observed["prior_attendance"], "prior_attendance"),
                       "initial_need": baseline["initial_need"], "site": baseline["site"],
                       "previous_action": previous_action}
            rule = next(rule for rule in rules_by_stage[row["stage"]]
                        if "when" not in rule or _matches(rule["when"], context))
            episodes.append({
                "stage": row["stage"], "context": context, "action": row["action"],
                "rule_index": rule["index"], "prescribed": rule["action"],
                "propensity_observed": row["action_probabilities"][row["action"]],
                "propensity_prescribed": row["action_probabilities"][rule["action"]],
            })
            previous_action = row["action"]

        ending, included, exclusion_reason = _classify(rows, stages)
        score = outcomes[identity]
        weight = 1.0
        for episode in episodes:
            weight /= episode["propensity_observed"]
        stage_features = {}
        for episode in episodes:
            stage = episode["stage"]
            stage_features[stage] = {
                "observed": _features(feature_spec, episode["context"], baseline, episode["action"]),
                "prescribed": _features(feature_spec, episode["context"], baseline, episode["prescribed"]),
                "match": episode["action"] == episode["prescribed"],
                "propensity_observed": episode["propensity_observed"],
            }
        journeys.append({
            "learner": identity, "ending": ending, "included": included,
            "exclusion_reason": exclusion_reason, "episodes": episodes,
            "stages": stage_features,
            "consistent": all(episode["action"] == episode["prescribed"] for episode in episodes),
            "supported": all(episode["propensity_prescribed"] > 0 for episode in episodes),
            "weight": weight,
            "outcome_observed": score is not None,
            "y_tilde": float(score) if score is not None else 0.0,
            "obs": 1.0 if score is not None else 0.0,
        })
    return journeys


def _classify(rows, stages):
    """区分 completed / withdrawn / lost / 未到下一阶段 / 从未决策。"""
    if not rows:
        return "never_decided", False, "无任何决策记录，无法核对规则一致性"
    if rows[0]["stage"] != stages[0]:
        return "not_anchored", False, "首个决策记录不在第一决策阶段"
    terminal = rows[-1]["terminal_reason"]
    if terminal is not None:
        return terminal, True, None
    if rows[-1]["stage"] == stages[-1]:
        return "active_through_final_stage", True, None
    return "ongoing_incomplete", False, "历程在未到最后决策阶段前中断且未标记终态（未到下一阶段）"


def _features(spec, context, baseline, action):
    need = float(context["need"])
    row = [1.0, need, float(context["prior_attendance"]), float(baseline["initial_need"])]
    row += [1.0 if baseline["site"] == site else 0.0 for site in spec["sites"][1:]]
    dummies = [1.0 if action == candidate else 0.0 for candidate in spec["actions"][1:]]
    row += dummies
    row += [need * dummy for dummy in dummies]
    return row


# ---------------------------------------------------------------- 识别与诊断

def _score_range(packet, config):
    if config["score_range"] is not None:
        return config["score_range"], "调用方经 options.score_range 给定"
    observed = [row["score"] for row in packet["outcomes"] if row["score"] is not None]
    if not observed:
        return None, "无观测分数且未给 options.score_range，无法构成缺测边界"
    return (float(min(observed)), float(max(observed))), "观测分数范围（可用 options.score_range 覆盖）"


def _support_diagnostics(included, stages, threshold):
    """逐阶段逐规则核对所开动作的历史支持度，产出适用边界。"""
    support, boundary = [], []
    for stage in stages:
        groups = defaultdict(list)
        for journey in included:
            for episode in journey["episodes"]:
                if episode["stage"] == stage:
                    groups[(episode["rule_index"], episode["prescribed"])].append(episode)
        for (rule_index, action), episodes in sorted(groups.items()):
            propensities = [episode["propensity_prescribed"] for episode in episodes]
            zero = [episode for episode in episodes if episode["propensity_prescribed"] == 0]
            below = [episode for episode in episodes
                     if 0 < episode["propensity_prescribed"] < threshold]
            entry = {"stage": stage, "rule_index": rule_index, "action": action,
                     "prescribed_learners": len(episodes),
                     "min_logged_propensity": float(min(propensities)),
                     "zero_support_learners": len(zero),
                     "below_threshold_learners": len(below)}
            support.append(entry)
            if zero or below:
                examples = [episode["context"] for episode in (zero + below)[:EXAMPLE_STATES_PER_BOUNDARY]]
                boundary.append({**entry, "example_states": examples})
    return support, boundary


def _identification(included, support, boundary):
    if not included:
        return {"status": "not_identified", "support": support, "applicability_boundary": boundary,
                "supported_learners": 0, "consistent_learners": 0, "included_learners": 0,
                "supported_share": 0.0,
                "reasons": ["没有可用于估值的完整历程"]}
    supported = [row for row in included if row["supported"]]
    consistent = [row for row in included if row["consistent"]]
    share = len(supported) / len(included)
    reasons = []
    if not supported or not consistent:
        status = "not_identified"
        if not supported:
            reasons.append("规则所开动作在历史记录中支持为零的路径覆盖全部适用学员，"
                           "没有可对照的经验，不硬算收益；详见 applicability_boundary")
        if not consistent:
            reasons.append("没有任何学员的实际历程与规则一致，序贯加权没有可用样本")
    elif share < 1:
        status = "partially_identified"
        reasons.append("部分学员的规则处方在历史上支持为零或低于阈值，"
                       "数值仅适用于有支持的子人群，不作为全人群估值；详见 applicability_boundary")
    else:
        status = "identified"
    return {"status": status, "support": support, "applicability_boundary": boundary,
            "supported_learners": len(supported), "consistent_learners": len(consistent),
            "included_learners": len(included), "supported_share": share, "reasons": reasons}


def _exclusions_note(excluded):
    if not excluded:
        return None
    counts = Counter(row["exclusion_reason"] for row in excluded)
    detail = "；".join(f"{reason}（{count} 人）" for reason, count in sorted(counts.items()))
    return f"以下初始学员不进入估值，目标人群相应缩小：{detail}"


# ---------------------------------------------------------------- 估计量

def _aipw_component(rows, stages, key):
    """序贯增广 IPW：向后递归 Q 回归，倾向用记录值而非拟合值。"""
    values = [row[key] for row in rows]
    for stage in reversed(stages):
        indices = [index for index, row in enumerate(rows) if stage in row["stages"]]
        if not indices:
            continue
        design = np.array([rows[index]["stages"][stage]["observed"] for index in indices])
        target = np.array([values[index] for index in indices])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        for index in indices:
            node = rows[index]["stages"][stage]
            q_observed = float(np.dot(node["observed"], beta))
            q_prescribed = float(np.dot(node["prescribed"], beta))
            augmentation = ((values[index] - q_observed) / node["propensity_observed"]
                            if node["match"] else 0.0)
            values[index] = q_prescribed + augmentation
    return sum(values) / len(values)


def _ipw(rows, key):
    total = sum(row["weight"] * row[key] for row in rows if row["consistent"])
    weights = sum(row["weight"] for row in rows if row["consistent"])
    return total / len(rows), (total / weights if weights else None)


def _weight_stats(rows):
    weights = [row["weight"] for row in rows if row["consistent"]]
    if not weights:
        return {"max_weight": None, "effective_sample_size": 0.0}
    ess = sum(weights) ** 2 / sum(weight ** 2 for weight in weights)
    return {"max_weight": float(max(weights)), "effective_sample_size": float(ess)}


def _bounds(component, share, score_range, basis):
    if score_range is None:
        return None
    share = min(max(share, 0.0), 1.0)
    low, high = score_range
    return {"low": component + (1 - share) * low, "high": component + (1 - share) * high,
            "basis": f"缺测结局按分数范围 [{low}, {high}] 计入：{basis}"}


def _estimate(included, stages, score_range, score_range_basis):
    component = _aipw_component(included, stages, "y_tilde")
    share = _aipw_component(included, stages, "obs")
    ht_component, hajek_component = _ipw(included, "y_tilde")
    ht_share, hajek_share = _ipw(included, "obs")
    estimate = {
        "outcome_component": component,
        "outcome_observed_share": share,
        "conditional_mean_if_observed": component / share if share > 0 else None,
        "value_bounds": _bounds(component, share, score_range, score_range_basis),
        "estimators": {"aipw_component": component, "ipw_ht_component": ht_component,
                       "ipw_hajek_component": hajek_component,
                       "aipw_observed_share": share, "ipw_ht_observed_share": ht_share,
                       "ipw_hajek_observed_share": hajek_share},
        "weights": {**_weight_stats(included), "included_learners": len(included)},
    }
    if share == 0:
        estimate["status"] = "not_estimable"
        estimate["reasons"] = ["固定期结局全部缺测（失访），过程测量不自动充当终期收益，"
                               "条件均值与点估值不可得；仅报告可识别分量与边界"]
    else:
        estimate["status"] = "ok"
    return estimate


def _current_process(included, score_range, score_range_basis):
    if not included:
        return {"status": "not_available", "reason": "没有可用于对照的完整历程"}
    component = sum(row["y_tilde"] for row in included) / len(included)
    share = sum(row["obs"] for row in included) / len(included)
    return {"status": "ok",
            "description": "现行流程的描述性汇总：同一初始群体的观测历程，非反事实估值",
            "outcome_component": component,
            "outcome_observed_share": share,
            "conditional_mean_if_observed": component / share if share > 0 else None,
            "value_bounds": _bounds(component, share, score_range, score_range_basis)}


def _comparison(estimate, current):
    regime_conditional = estimate["conditional_mean_if_observed"]
    current_conditional = current["conditional_mean_if_observed"]
    difference_conditional = (regime_conditional - current_conditional
                              if regime_conditional is not None and current_conditional is not None
                              else None)
    return {"status": "ok",
            "description": "同一初始群体上规则与现行流程的对照；差值不确定度见 uncertainty",
            "difference_outcome_component":
                estimate["outcome_component"] - current["outcome_component"],
            "difference_observed_share":
                estimate["outcome_observed_share"] - current["outcome_observed_share"],
            "difference_conditional_mean": difference_conditional}


def _uncertainty(included, stages, config):
    reps = config["bootstrap_reps"]
    if reps == 0:
        return {"status": "skipped", "reason": "bootstrap_reps=0，调用方选择不计算不确定度"}
    rng = np.random.RandomState(config["bootstrap_seed"])
    size = len(included)
    draws = {"regime_component": [], "current_component": [], "difference_component": [],
             "difference_conditional": []}
    skipped_conditional = 0
    for _ in range(reps):
        sample = [included[index] for index in rng.randint(0, size, size=size)]
        regime_component = _aipw_component(sample, stages, "y_tilde")
        regime_share = _aipw_component(sample, stages, "obs")
        current_component = sum(row["y_tilde"] for row in sample) / size
        current_share = sum(row["obs"] for row in sample) / size
        draws["regime_component"].append(regime_component)
        draws["current_component"].append(current_component)
        draws["difference_component"].append(regime_component - current_component)
        if regime_share > 0 and current_share > 0:
            draws["difference_conditional"].append(
                regime_component / regime_share - current_component / current_share)
        else:
            skipped_conditional += 1

    def interval(values):
        low, high = np.percentile(np.array(values), [2.5, 97.5])
        return [float(low), float(high)]

    result = {"method": "对学员配对的百分位自助法（同一重抽样下计算规则与现行流程）",
              "reps": reps, "seed": config["bootstrap_seed"],
              "regime_component_ci": interval(draws["regime_component"]),
              "current_component_ci": interval(draws["current_component"]),
              "difference_component_ci": interval(draws["difference_component"]),
              "difference_conditional_mean_ci": (interval(draws["difference_conditional"])
                                                 if draws["difference_conditional"] else None),
              "monte_carlo_note": f"{reps} 次重抽样，蒙特卡洛误差量级 O(1/sqrt(reps))"}
    if skipped_conditional:
        result["conditional_skipped_reps"] = skipped_conditional
    if reps < 200:
        result["precision_warning"] = "重抽样次数较少，区间仅供粗略参考"
    return result


# ---------------------------------------------------------------- 结果组织

def _process_measures(packet, stages):
    rewards = {}
    for row in packet["rewards"]:
        key = (row["learner"], row["after_stage"])
        if key in rewards:
            raise ValueError("duplicate reward row")
        if not row["missing_reward"]:
            finite(row["reward"], "reward")
        rewards[key] = row
    by_stage = []
    ordered = sorted(packet["episodes"], key=lambda row: (row["learner"], row["stage"]))
    for stage in stages:
        observed, missing, unrecorded = [], 0, 0
        for episode in ordered:
            if episode["stage"] != stage:
                continue
            row = rewards.get((episode["learner"], stage))
            if row is None:
                unrecorded += 1
            elif row["missing_reward"]:
                missing += 1
            else:
                observed.append(row["reward"])
        by_stage.append({"stage": stage, "observed": len(observed), "missing": missing,
                         "unrecorded": unrecorded,
                         "mean_observed": float(sum(observed) / len(observed)) if observed else None})
    return {"note": "阶段后过程测量，仅作描述，未计入策略价值；缺失不以零代替",
            "stages": by_stage}


def _assumptions(score_range, score_range_basis, status):
    assumptions = [
        "记录的 action_probabilities 视为给定当时决策状态的真实行为倾向（序贯可忽略性）；"
        "未记录的混杂无法排除",
        "规则只能引用决策时可见字段（need、prior_attendance、initial_need、site、"
        "previous_action），结构上阻止把未来结局带回前面的决定",
        "缺测结局不作零或均值填补；全人群均值以可识别边界表示",
        "个体之间无相互干扰，记录与潜在结局一致（consistency）",
    ]
    if score_range is not None:
        assumptions.append(f"缺测结局边界采用分数范围 [{score_range[0]}, {score_range[1]}]：{score_range_basis}")
    else:
        assumptions.append("无可用分数范围，缺测结局边界不给出")
    if status == "partially_identified":
        assumptions.append("历史支持不足时，数值仅适用于有支持的子人群，不外推为全人群估值")
    return assumptions


def _diagnostics(included, estimate, identification):
    diagnostics = []
    if identification["status"] == "partially_identified":
        diagnostics.append({"kind": "partial_coverage",
                            "detail": "规则部分路径缺乏历史支持，估值仅适用于有支持子人群"})
    if not estimate or estimate.get("status") == "not_identified":
        return diagnostics
    weights = estimate["weights"]
    consistent = identification["consistent_learners"]
    if consistent and weights["effective_sample_size"] < max(10.0, 0.05 * len(included)):
        diagnostics.append({"kind": "low_effective_sample",
                            "detail": f"有效样本量 {weights['effective_sample_size']:.1f}，"
                                      f"权重集中，估值方差可能很大"})
    if estimate["status"] == "ok":
        gap = abs(estimate["estimators"]["aipw_component"]
                  - estimate["estimators"]["ipw_ht_component"])
        if gap > DISAGREEMENT_TOLERANCE:
            diagnostics.append({"kind": "estimator_disagreement",
                                "detail": f"AIPW 与 IPW-HT 分量相差 {gap:.3f} 分，"
                                          f"超出 {DISAGREEMENT_TOLERANCE} 分容差，请复核权重与模型"})
    return diagnostics


def _rule_audit(journeys, support):
    learners = []
    for journey in journeys:
        record = {"learner": journey["learner"], "ending": journey["ending"],
                  "included": journey["included"],
                  "prescribed": [episode["prescribed"] for episode in journey["episodes"]],
                  "observed": [episode["action"] for episode in journey["episodes"]],
                  "consistent": journey["consistent"], "supported": journey["supported"],
                  "weight": journey["weight"], "outcome_observed": journey["outcome_observed"]}
        if journey["exclusion_reason"] is not None:
            record["exclusion_reason"] = journey["exclusion_reason"]
        learners.append(record)
    return {"usage": support, "learners": learners}


def _interpretation(result):
    used = result["data_used"]
    identification = result["identification"]
    parts = [
        f"在 {used['initial_roster']} 名初始学员（{used['included']} 名进入估值）上评估规则"
        f"「{result['policy']['name']}」，目标为第 {used['outcome']['week']} 周结局。"
    ]
    estimate = result.get("regime_estimate", {})
    current = result.get("current_process", {})
    if estimate.get("status") == "ok" and current.get("status") == "ok":
        parts.append(
            f"规则下结局分量估计 {estimate['outcome_component']:.2f}"
            f"（现行流程 {current['outcome_component']:.2f}，"
            f"差值 {result['comparison']['difference_outcome_component']:+.2f}）；"
            f"结局可观测比例 {estimate['outcome_observed_share']:.2f}。")
        if estimate["value_bounds"] and current["value_bounds"]:
            parts.append(
                f"因失访缺测，全人群均值只能界定：规则 "
                f"[{estimate['value_bounds']['low']:.2f}, {estimate['value_bounds']['high']:.2f}]，"
                f"现行 [{current['value_bounds']['low']:.2f}, {current['value_bounds']['high']:.2f}]。")
        uncertainty = result.get("uncertainty", {})
        if uncertainty.get("difference_component_ci"):
            low, high = uncertainty["difference_component_ci"]
            parts.append(f"分量差值 95% 自助区间 [{low:+.2f}, {high:+.2f}]"
                         f"（{uncertainty['reps']} 次重抽样，种子 {uncertainty['seed']}）。")
    elif estimate.get("status") == "not_estimable":
        parts.append("固定期结局全部失访缺测，过程测量不充当终期收益，规则价值不可点估计。")
    else:
        parts.append("规则价值不可识别：" + "；".join(identification["reasons"]) + "。")
    if identification["status"] == "partially_identified":
        parts.append("注意：部分处方路径历史支持不足，数值仅适用于有支持子人群，"
                     "适用边界见 identification.applicability_boundary。")
    parts.append("逐学员判定与依据见 rule_audit，供业务复核为何继续或更换支持。")
    return "".join(parts)
