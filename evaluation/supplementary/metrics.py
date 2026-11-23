"""Original M2 count estimator with supplementary reward/length reporting."""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathlib import Path as _ArchivePath
import sys as _archive_sys
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[2] / 'analysis'))
from action_precision import balance_action
from protocol import score_response


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fit(counts, invalid, src, dst, n):
    attempts = invalid + np.bincount(src, weights=counts, minlength=n).astype(int)
    degree = np.bincount(src, minlength=n)
    t = (counts + .1) / (attempts[src] + .1 * degree[src])
    phi, action, diagnostics = balance_action({"src": src, "dst": dst, "n": n}, t)
    if not diagnostics["converged"]:
        raise RuntimeError(f"Action solver did not converge: {diagnostics}")
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(src, dst))}
    pairs = [(k, lookup[j, i]) for (i, j), k in lookup.items()
             if i < j and (j, i) in lookup and counts[k] > 0 and counts[lookup[j, i]] > 0]
    left = np.array([p[0] for p in pairs], int)
    right = np.array([p[1] for p in pairs], int)
    value = corr(phi[src[left]] - phi[dst[left]], np.log(t[left]) - np.log(t[right]))
    return {"r": value, "phi_action": phi.tolist(), "action": action,
            "action_solver": diagnostics, "t_hat": t.tolist(),
            "attempts": attempts.tolist(), "reciprocal_pairs_observed": len(pairs),
            "reciprocal_pairs_total": sum(i < j and (j, i) in lookup for i, j in lookup)}


def summarize(records, rows, replicates, seed):
    labels = [__import__("json").loads(row["label"]) for row in records]
    n = len(labels)
    edges = sorted((i, int(target)) for i, label in enumerate(labels) for target in label["mapping"].values())
    lookup = {edge: k for k, edge in enumerate(edges)}
    src, dst = np.array(edges, int).T
    counts = np.zeros(len(edges), np.int64)
    invalid = np.zeros(n, np.int64)
    reasons = [Counter() for _ in range(n)]
    lengths, truncated, empty_thinking = [], 0, 0
    for row in rows:
        i = row["source"]
        scored = score_response(row["text"], row["truncated"], labels[i])
        if scored["target"] is None:
            invalid[i] += 1
            reasons[i][scored["invalid_reason"]] += 1
        else:
            counts[lookup[i, scored["target"]]] += 1
        truncated += int(row["truncated"])
        lengths.append(len(row["token_ids"]))
        if labels[i].get("thinking") and "</think>" in row["text"]:
            empty_thinking += not row["text"].split("</think>")[0].strip()
    u = np.array([labels[0]["state_rewards"][str(i)] for i in range(n)], float)
    fitted = fit(counts, invalid, src, dst, n)
    attempts = np.array(fitted["attempts"])
    if np.any(attempts == 0):
        raise ValueError("A registered source has no measurement attempts")
    reward_sums = np.bincount(src, weights=counts * (1 + u[dst]), minlength=n)
    mean_reward = float(np.mean(reward_sums / attempts))
    mean_u = float(np.mean(np.bincount(src, weights=counts * u[dst], minlength=n) / attempts))
    mean_var = float(np.sum(np.maximum(0, np.bincount(src, weights=counts * (1 + u[dst]) ** 2,
        minlength=n) / attempts - (reward_sums / attempts) ** 2) / (attempts - 1)) / n ** 2)
    entropy = []
    for i in range(n):
        c = counts[src == i]
        if c.sum() == 0:
            entropy.append(None)
        else:
            p = c[c > 0] / c.sum()
            entropy.append(float(-np.sum(p * np.log(p))))
    support = csr_matrix((counts > 0, (src, dst)), shape=(n, n))
    support.eliminate_zeros()
    scc = connected_components(support, directed=True, connection="strong")[0]
    alignment = corr(-np.array(fitted["phi_action"]), u)
    boot_r, boot_align, boot_reward, boot_draws = [], [], [], []
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        c, bad = np.zeros_like(counts), np.zeros_like(invalid)
        for i in range(n):
            ix = np.flatnonzero(src == i)
            draw = rng.multinomial(int(attempts[i]), np.r_[counts[ix], invalid[i]] / attempts[i])
            c[ix], bad[i] = draw[:-1], draw[-1]
        result = fit(c, bad, src, dst, n)
        if result["r"] is not None:
            boot_r.append(result["r"])
        a = corr(-np.array(result["phi_action"]), u)
        if a is not None:
            boot_align.append(a)
        reward = float(np.mean(np.bincount(src, weights=c * (1 + u[dst]), minlength=n) / attempts))
        boot_reward.append(reward)
        boot_draws.append({"replicate": replicate, "r": result["r"], "alignment": a, "reward": reward})
    def ci(values):
        return np.quantile(values, [.025, .975]).tolist() if len(values) >= max(10, replicates // 4) else None
    return {**fitted, "edges": edges, "counts": counts.tolist(), "invalid_counts": invalid.tolist(),
            "alpha": .1, "n_states": n, "u": u.tolist(), "total_attempts": int(attempts.sum()),
            "invalid_rate": float(invalid.sum() / attempts.sum()), "valid_rate": float(counts.sum() / attempts.sum()),
            "reason_counts": [dict(x) for x in reasons], "observed_directed_edges": int((counts > 0).sum()),
            "observed_scc_count": int(scc), "reward_alignment_corr_minus_phi_u": alignment,
            "mean_reward": mean_reward, "mean_reward_variance": mean_var, "mean_semantic_reward": mean_u,
            "raw_conditional_semantic_entropy": float(np.mean([h for h in entropy if h is not None]))
                if any(h is not None for h in entropy) else None,
            "entropy_by_source": entropy, "truncated_rate": truncated / len(rows),
            "empty_reasoning_rate": empty_thinking / len(rows),
            "generation_tokens": {"mean": float(np.mean(lengths)),
                                  "quantiles": np.quantile(lengths, [.5, .9, .95, .99, 1]).tolist()},
            "bootstrap": {"replicates": replicates, "finite_r_replicates": len(boot_r),
                          "r_ci95": ci(boot_r), "alignment_ci95": ci(boot_align),
                          "reward_ci95": ci(boot_reward), "draws": boot_draws}}
