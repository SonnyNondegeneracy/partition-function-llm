"""M1-compatible count estimator for sampled checkpoint measurements."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from protocol import score_response

from pathlib import Path as _ArchivePath
import sys as _archive_sys
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[2] / 'analysis'))
from action_precision import balance_action

ALPHA = 0.1


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def graph(records):
    labels = [json.loads(row["label"]) for row in records]
    n = len(labels)
    edges = sorted({(int(label["source"]), int(target))
                    for label in labels for target in label["mapping"].values()
                    if int(target) != int(label["source"])})
    if len(edges) == 0 or len({source for source, _ in edges}) != n:
        raise ValueError("registered T1 graph has missing source support")
    edge_index = {edge: index for index, edge in enumerate(edges)}
    source, target = np.asarray(edges, dtype=int).T
    return labels, edges, edge_index, source, target, n


def _fit(counts, invalid, source, target, n, alpha=ALPHA):
    degree = np.bincount(source, minlength=n).astype(float)
    attempts = invalid + np.bincount(source, weights=counts, minlength=n).astype(np.int64)
    if np.any(attempts <= 0):
        raise ValueError("every registered source needs at least one attempt")
    # Pseudocounts smooth probabilities but never create observed reciprocal support.
    transition = (counts.astype(float) + alpha) / (attempts[source] + alpha * degree[source])
    action_model = {"src": source, "dst": target, "n": n}
    phi, action, diagnostics = balance_action(action_model, transition)
    if not diagnostics.get("converged", False):
        raise RuntimeError(f"action solver failed: {diagnostics}")
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(source, target))}
    pairs = [(k, lookup[(j, i)]) for (i, j), k in lookup.items()
             if i < j and (j, i) in lookup and counts[k] > 0 and counts[lookup[(j, i)]] > 0]
    left = np.asarray([item[0] for item in pairs], dtype=int)
    right = np.asarray([item[1] for item in pairs], dtype=int)
    r = corr(phi[source[left]] - phi[target[left]],
             np.log(transition[left]) - np.log(transition[right])) if len(left) >= 3 else None
    raw_r = corr(phi[source[left]] - phi[target[left]],
                 np.log(counts[left] / attempts[source[left]])
                 - np.log(counts[right] / attempts[source[right]])) if len(left) >= 3 else None
    return {
        "r": r,
        "raw_count_logratio_r": raw_r,
        "phi_action": np.asarray(phi, dtype=float).tolist(),
        "action": float(action),
        "action_solver": diagnostics,
        "t_hat": transition.tolist(),
        "attempts": attempts.tolist(),
        "reciprocal_pairs_observed": int(len(pairs)),
        "reciprocal_pairs_total": int(sum(i < j and (j, i) in lookup for i, j in lookup)),
    }


def summarize(records, rows, *, bootstrap=500, seed=0, alpha=ALPHA):
    labels, edges, edge_index, source, target, n = graph(records)
    counts = np.zeros(len(edges), dtype=np.int64)
    invalid = np.zeros(n, dtype=np.int64)
    reasons = [Counter() for _ in range(n)]
    lengths, truncated = [], 0
    for row in rows:
        source_id = int(row["source"])
        if source_id < 0 or source_id >= n:
            raise ValueError("sample source is not registered")
        parsed = score_response(row["text"], bool(row["truncated"]), labels[source_id])
        if parsed["target"] is None:
            invalid[source_id] += 1
            reasons[source_id][parsed["invalid_reason"] or "invalid"] += 1
        else:
            counts[edge_index[(source_id, parsed["target"])]] += 1
        lengths.append(len(row.get("token_ids", [])))
        truncated += int(bool(row["truncated"]))
    fitted = _fit(counts, invalid, source, target, n, alpha)
    u = np.asarray([float(labels[0]["state_rewards"][str(i)]) for i in range(n)])
    attempts = np.asarray(fitted["attempts"], dtype=np.int64)
    reward_sums = np.bincount(source, weights=counts * (1.0 + u[target]), minlength=n)
    mean_reward = float(np.mean(reward_sums / attempts))
    entropy_by_source = []
    for i in range(n):
        values = counts[source == i]
        if values.sum() == 0:
            entropy_by_source.append(None)
        else:
            p = values[values > 0] / values.sum()
            entropy_by_source.append(float(-np.sum(p * np.log(p))))
    support = csr_matrix((counts > 0, (source, target)), shape=(n, n))
    scc = int(connected_components(support, directed=True, connection="strong")[0])
    alignment = corr(-np.asarray(fitted["phi_action"]), u)
    boot_r, boot_alignment, boot_reward = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(int(bootstrap)):
        sampled_counts = np.zeros_like(counts)
        sampled_invalid = np.zeros_like(invalid)
        for i in range(n):
            ix = np.flatnonzero(source == i)
            probabilities = np.r_[counts[ix], invalid[i]].astype(float)
            probabilities /= probabilities.sum()
            draw = rng.multinomial(int(attempts[i]), probabilities)
            sampled_counts[ix], sampled_invalid[i] = draw[:-1], draw[-1]
        fitted_boot = _fit(sampled_counts, sampled_invalid, source, target, n, alpha)
        if fitted_boot["r"] is not None:
            boot_r.append(fitted_boot["r"])
        value = corr(-np.asarray(fitted_boot["phi_action"]), u)
        if value is not None:
            boot_alignment.append(value)
        rewards = np.bincount(source, weights=sampled_counts * (1.0 + u[target]), minlength=n)
        boot_reward.append(float(np.mean(rewards / attempts)))

    def interval(values):
        if len(values) < max(10, int(bootstrap) // 4):
            return None
        return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]

    return {
        **fitted, "edges": [list(edge) for edge in edges], "counts": counts.tolist(),
        "invalid_counts": invalid.tolist(), "alpha": alpha, "n_states": n,
        "u": u.tolist(), "total_attempts": int(attempts.sum()),
        "invalid_rate": float(invalid.sum() / attempts.sum()),
        "valid_rate": float(counts.sum() / attempts.sum()),
        "reason_counts": [dict(item) for item in reasons],
        "observed_directed_edges": int(np.count_nonzero(counts)),
        "observed_scc_count": scc,
        "reward_alignment_corr_minus_phi_u": alignment,
        "mean_reward": mean_reward,
        "mean_semantic_reward": float(np.mean(np.bincount(source, weights=counts * u[target], minlength=n) / attempts)),
        "raw_conditional_semantic_entropy": float(np.mean([x for x in entropy_by_source if x is not None])) if any(x is not None for x in entropy_by_source) else None,
        "entropy_by_source": entropy_by_source,
        "truncated_rate": float(truncated / max(len(rows), 1)),
        "generation_tokens": {"mean": float(np.mean(lengths)) if lengths else 0.0,
                              "quantiles": np.quantile(lengths, [.5, .9, .95, .99, 1]).tolist() if lengths else []},
        "bootstrap": {"replicates": int(bootstrap), "finite_r_replicates": len(boot_r),
                      "r_ci95": interval(boot_r), "alignment_ci95": interval(boot_alignment),
                      "reward_ci95": interval(boot_reward)},
    }
