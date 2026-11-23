"""Frozen direct T1 parser and reward contract.

Only the answer marker and registered candidate alphabet are accepted.  A
self-transition is invalid because the frozen task excludes it from the edge
graph; invalid mass remains in the source denominator.
"""
from __future__ import annotations

import re


def score_response(text: str, truncated: bool, label: dict) -> dict:
    if truncated:
        return {"target": None, "invalid_reason": "truncated", "reward": 0.0, "u": 0.0}
    alphabet = "".join(label["mapping"])
    match = re.fullmatch(rf"\s*ANSWER: ([{re.escape(alphabet)}])\s*", text)
    key = match.group(1) if match else None
    target = label["mapping"].get(key) if key is not None else None
    if target is None or int(target) == int(label["source"]):
        return {"target": None, "invalid_reason": "format", "reward": 0.0, "u": 0.0}
    u = float(label["state_rewards"][str(target)])
    return {"target": int(target), "invalid_reason": None, "reward": 1.0 + u, "u": u}


def check_parser() -> None:
    label = {"source": 0, "mapping": {"A": 1, "B": 0}, "state_rewards": {"1": 0.5, "0": 0.0}}
    assert score_response("ANSWER: A", False, label)["target"] == 1
    assert score_response("ANSWER: A extra", False, label)["target"] is None
    assert score_response("ANSWER: A", True, label)["target"] is None
    assert score_response("ANSWER: B", False, label)["invalid_reason"] == "format"
