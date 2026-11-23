"""Supplementary parsers; the original direct parser remains unchanged."""
from __future__ import annotations

import re


def score_response(text, truncated, label):
    if truncated:
        return {"target": None, "invalid_reason": "truncated", "reward": 0., "u": 0.}
    final = text
    if label.get("thinking"):
        # Qwen's official generation prompt has already opened <think>.
        if final.count("</think>") != 1 or "<think>" in final:
            return {"target": None, "invalid_reason": "reasoning_boundary", "reward": 0., "u": 0.}
        _, final = final.split("</think>")
    alphabet = "".join(label["mapping"])
    match = re.fullmatch(rf"\s*ANSWER: ([{re.escape(alphabet)}])\s*", final)
    key = match.group(1) if match else None
    target = label["mapping"].get(key)
    if target is None or target == label["source"]:
        return {"target": None, "invalid_reason": "format", "reward": 0., "u": 0.}
    u = float(label["state_rewards"][str(target)])
    return {"target": int(target), "invalid_reason": None, "reward": 1. + u, "u": u}


def check_parser():
    label = {"source": 0, "mapping": {"A": 1}, "state_rewards": {"1": .5}}
    assert score_response("ANSWER: A", False, label)["reward"] == 1.5
    assert score_response("ANSWER: A extra", False, label)["target"] is None
    assert score_response("ANSWER: A", True, label)["target"] is None
    label["thinking"] = True
    assert score_response("Consider A.\n</think>\nANSWER: A", False, label)["reward"] == 1.5
    for text in ["ANSWER: A", "<think>x</think>ANSWER: A", "x</think></think>ANSWER: A"]:
        assert score_response(text, False, label)["target"] is None
