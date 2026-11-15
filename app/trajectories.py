from collections import Counter, defaultdict
from app.protocol import table_mean


def learner_paths(packet):
    paths = defaultdict(list)
    for episode in packet["episodes"]:
        paths[episode["learner"]].append(episode)
    result = {}
    for learner in packet["roster"]:
        identity = learner["learner"]
        rows = sorted(paths[identity], key=lambda row: row["stage"])
        for index, row in enumerate(rows[:-1]):
            if row["terminal_reason"] is not None:
                raise ValueError("decision after terminal support state")
            if rows[index + 1]["stage"] != row["stage"] + 1:
                raise ValueError("nonconsecutive decision stages")
        result[identity] = rows
    return result


def trajectory_profile(packet):
    paths = learner_paths(packet)
    outcomes = {row["learner"]: row for row in packet["outcomes"]}
    if set(outcomes) != set(paths):
        raise ValueError("final outcome roster differs from initial roster")
    ending_counts = Counter()
    products = []
    for learner, rows in paths.items():
        ending_counts[rows[-1]["terminal_reason"] if rows else "never_decided"] += 1
        probability = 1.0
        for row in rows:
            probability *= row["action_probabilities"][row["action"]]
        products.append(probability)
    return {"ending_counts": dict(ending_counts), "initial_roster": len(paths),
            "observed_final_scores": sum(row["score"] is not None for row in outcomes.values()),
            "observed_final_mean": table_mean([row["score"] for row in outcomes.values()]),
            "logged_path_probability_range": [min(products), max(products)] if products else None,
            "outcome_week": packet["decision_schedule"]["outcome_week"],
            "interpretation": "Logged path probabilities describe observed support, not policy weights."}
