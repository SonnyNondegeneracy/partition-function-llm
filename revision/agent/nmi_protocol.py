"""Parser and reward for the revision runs (direct or reasoning answers; state, edge or mixed reward)."""
import re


def score_response(text, truncated, label):
    bad = {"target": None, "reward": 0.0, "u": 0.0, "epsilon": None}
    if truncated:
        return {**bad, "invalid_reason": "truncated"}
    alphabet = "".join(sorted(label["mapping"]))
    if label.get("parser", "direct") == "direct":
        m = re.fullmatch(rf"\s*ANSWER: ([{alphabet}])\s*", text)
        key = m.group(1) if m else None
    else:  # reasoning, then exactly one final marker at the end
        m = re.fullmatch(rf"(?s)(.*?)ANSWER: ([{alphabet}])\s*", text)
        key = m.group(2) if m and "ANSWER:" not in m.group(1) else None
    target = label["mapping"].get(key)
    if target is None or int(target) == int(label["source"]):
        return {**bad, "invalid_reason": "format"}
    target = int(target)
    if label.get("reward_kind", "state") == "mix":  # E5: R = 1 + u_j + alpha * eps_ij
        u = float(label["state_rewards"][str(target)]); e = float(label["edge_epsilon"][str(target)])
        return {"target": target, "invalid_reason": None, "reward": 1.0 + u + float(label["alpha"]) * e,
                "u": u, "epsilon": e}
    if label.get("reward_kind", "state") == "edge":
        return {"target": target, "invalid_reason": None, "reward": float(label["edge_rewards"][str(target)]),
                "u": 0.0, "epsilon": float(label["edge_epsilon"][str(target)])}
    u = float(label["state_rewards"][str(target)])
    return {"target": target, "invalid_reason": None, "reward": 1.0 + u, "u": u, "epsilon": None}


def check_parser():
    lab = {"source": 0, "mapping": {"A": 1, "B": 2}, "state_rewards": {"1": 0.5, "2": 1.0}}
    assert score_response("ANSWER: B", False, lab)["reward"] == 2.0
    assert score_response("ANSWER: B extra", False, lab)["target"] is None
    c = {**lab, "parser": "cot"}
    assert score_response("B looks best because ...\nANSWER: B\n", False, c)["target"] == 2
    assert score_response("ANSWER: A\nANSWER: B", False, c)["target"] is None
    assert score_response("no marker", False, c)["target"] is None
    assert score_response("ANSWER: B", True, c)["invalid_reason"] == "truncated"
    e = {**lab, "reward_kind": "edge", "edge_rewards": {"1": 0, "2": 2}, "edge_epsilon": {"1": -1, "2": 1}}
    assert score_response("ANSWER: B", False, e)["reward"] == 2.0
    x = {**lab, "reward_kind": "mix", "alpha": 0.5, "edge_epsilon": {"1": -1, "2": 1}}
    assert score_response("ANSWER: B", False, x)["reward"] == 2.5
    print("NMI_PARSER_OK")


if __name__ == "__main__":
    check_parser()
