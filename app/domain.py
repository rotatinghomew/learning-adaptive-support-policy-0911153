from app.trajectories import trajectory_profile
from collections import Counter
from app.protocol import finite, unique


def validate(packet):
    if not packet["revision"]:
        raise ValueError("missing revision")
    unique(packet["roster"], "learner")
    roster = {row["learner"] for row in packet["roster"]}
    seen = set()
    for row in packet["episodes"]:
        identity = (row["learner"], row["stage"])
        if identity in seen or row["learner"] not in roster:
            raise ValueError("duplicate or unknown episode")
        seen.add(identity)
        probabilities = row["action_probabilities"]
        if abs(sum(probabilities.values()) - 1) > 1e-9 or any(finite(value, "probability") < 0 for value in probabilities.values()):
            raise ValueError("invalid action distribution")
        if probabilities.get(row["action"], 0) <= 0:
            raise ValueError("logged action had no support")


def path_inventory(packet, options):
    stages = []
    for stage in sorted({row["stage"] for row in packet["episodes"]}):
        selected = [row for row in packet["episodes"] if row["stage"] == stage]
        stages.append({"stage": stage, "learners": len(selected),
                       "actions": dict(Counter(row["action"] for row in selected)),
                       "endings": dict(Counter(row["terminal_reason"] or "continuing" for row in selected))})
    return {"source_profile": trajectory_profile(packet), "initial_learners": len(packet["roster"]), "stages": stages,
            "interpretation": "Observed paths are not counterfactual strategy values."}


ANALYSES = {"path_inventory": path_inventory}
