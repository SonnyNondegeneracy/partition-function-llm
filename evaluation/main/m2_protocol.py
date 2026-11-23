"""Frozen direct TaskSpec parser and reward, shared by training and probes."""
import re


def score_response(text, truncated, label):
    # Both formal direct tasks use the exact uppercase marker. The candidate
    # alphabet remains task-specific and comes from the frozen mapping.
    if truncated:
        return {'target': None, 'invalid_reason': 'truncated', 'reward': 0., 'u': 0.}
    alphabet = 'ABCD' if label['task_id'].startswith('T2') else 'ABCDEFG'
    pattern = rf'\s*ANSWER: ([{alphabet}])\s*'
    match = re.fullmatch(pattern, text)
    key = match.group(1).upper() if match else None
    target = label['mapping'].get(key)
    if target is None or target == label['source']:
        return {'target': None, 'invalid_reason': 'format', 'reward': 0., 'u': 0.}
    u = float(label['state_rewards'][str(target)])
    return {'target': target, 'invalid_reason': None, 'reward': 1.+u, 'u': u}
