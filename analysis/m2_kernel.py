#!/usr/bin/env python3
"""CPU-side contracts and measurements for the M2 short-action RLHF study.

This module deliberately has no model or Magnus dependency.  A training or
inference runner only has to emit a :class:`TaskSpec`, candidate logits and
standardized transition records.  The semantic kernel is built from complete
candidate sequence probabilities, with invalid mass retained explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from action_precision import balance_action, SOLVER_VERSION


SCHEMA_VERSION = "m2.taskspec.v1"
DEFAULT_PSEUDOCOUNT = 0.1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemperatureContract:
    tau_rollout: float = 1.0
    tau_policy: float = 1.0
    tau_reference: float = 1.0
    tau_eval: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    thinking: bool = False
    stop: tuple[str, ...] = ()
    template_revision: str = "unspecified"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemperatureContract":
        fields = dict(value)
        fields["stop"] = tuple(fields.get("stop", ()))
        return cls(**fields)

    def validate(self, require_same: bool = True) -> None:
        taus = (self.tau_rollout, self.tau_policy, self.tau_reference,
                self.tau_eval)
        if not all(np.isfinite(x) and x > 0 for x in taus):
            raise ValueError("all temperatures must be finite and positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or null")
        if not np.isclose(self.repetition_penalty, 1.0):
            raise ValueError("M2 contract requires repetition_penalty=1")
        if require_same and not np.allclose(taus, taus[0], rtol=0, atol=1e-12):
            raise ValueError(
                "main M2 measurement requires tau_rollout=tau_policy="
                "tau_reference=tau_eval")
        if self.top_p != 1.0 or self.top_k is not None:
            raise ValueError("main M2 contract requires top_p=1 and top_k=null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_rollout": self.tau_rollout,
            "tau_policy": self.tau_policy,
            "tau_reference": self.tau_reference,
            "tau_eval": self.tau_eval,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "thinking": self.thinking,
            "stop": list(self.stop),
            "template_revision": self.template_revision,
        }

    @property
    def is_same_temperature(self) -> bool:
        taus = (self.tau_rollout, self.tau_policy, self.tau_reference,
                self.tau_eval)
        return bool(np.allclose(taus, taus[0], rtol=0, atol=1e-12))


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    target_state: str
    token_ids: tuple[int, ...]
    source_state: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        tokens = tuple(int(x) for x in value["token_ids"])
        if not tokens:
            raise ValueError("candidate token_ids cannot be empty")
        return cls(candidate_id=str(value["candidate_id"]),
                   target_state=str(value["target_state"]),
                   token_ids=tokens,
                   source_state=(None if value.get("source_state") is None
                                 else str(value["source_state"])))

    def as_dict(self) -> dict[str, Any]:
        result = {"candidate_id": self.candidate_id,
                  "target_state": self.target_state,
                  "token_ids": list(self.token_ids)}
        if self.source_state is not None:
            result["source_state"] = self.source_state
        return result


@dataclass
class TaskSpec:
    task_id: str
    version: str
    state_ids: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    worlds: tuple[dict[str, Any], ...]
    reward_table: dict[str, float]
    horizon: int
    max_new_tokens: int
    seed: int
    parser: dict[str, Any] = field(default_factory=lambda: {
        "kind": "exact_candidate_id", "strip_whitespace": True,
        "case_sensitive": True})
    temperature_contract: TemperatureContract = field(default_factory=TemperatureContract)
    prompt_template: str = "{prompt}"
    candidate_template: str = "{candidate_id}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        if value.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("unsupported TaskSpec schema_version")
        result = cls(
            task_id=str(value["task_id"]), version=str(value["version"]),
            state_ids=tuple(str(x) for x in value["state_ids"]),
            candidates=tuple(Candidate.from_dict(x) for x in value["candidates"]),
            worlds=tuple(dict(x) for x in value.get("worlds", [])),
            reward_table={str(k): float(v) for k, v in value["reward_table"].items()},
            horizon=int(value["horizon"]), max_new_tokens=int(value["max_new_tokens"]),
            seed=int(value["seed"]), parser=dict(value.get("parser", {})),
            temperature_contract=TemperatureContract.from_dict(
                value.get("temperature_contract", {})),
            prompt_template=str(value.get("prompt_template", "{prompt}")),
            candidate_template=str(value.get("candidate_template", "{candidate_id}")),
        )
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "version": self.version,
            "state_ids": list(self.state_ids),
            "candidates": [c.as_dict() for c in self.candidates],
            "worlds": list(self.worlds),
            "reward_table": dict(self.reward_table),
            "horizon": self.horizon,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "parser": dict(self.parser),
            "temperature_contract": self.temperature_contract.as_dict(),
            "prompt_template": self.prompt_template,
            "candidate_template": self.candidate_template,
        }

    @property
    def task_hash(self) -> str:
        return sha256_json(self.as_dict())

    @property
    def reward_hash(self) -> str:
        """Hash only the frozen HF proxy table for run-level provenance."""
        return sha256_json({"task_id": self.task_id,
                            "version": self.version,
                            "reward_table": self.reward_table})

    @property
    def state_index(self) -> dict[str, int]:
        return {state: i for i, state in enumerate(self.state_ids)}

    def validate(self) -> None:
        if not self.task_id or not self.version:
            raise ValueError("task_id and version are required")
        if not self.state_ids or len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("state_ids must be nonempty and unique")
        if self.horizon <= 0 or self.max_new_tokens <= 0:
            raise ValueError("horizon and max_new_tokens must be positive")
        if set(self.reward_table) != set(self.state_ids):
            raise ValueError("reward_table must contain every canonical state")
        ids = [c.candidate_id for c in self.candidates]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be nonempty and unique")
        index = self.state_index
        for c in self.candidates:
            if c.target_state not in index:
                raise ValueError(f"unknown candidate target state: {c.target_state}")
            if c.source_state is not None and c.source_state not in index:
                raise ValueError(f"unknown candidate source state: {c.source_state}")
        # Exact sequence parsing is ambiguous when one candidate is a prefix of
        # another.  Requiring a prefix-free code makes sequence probabilities a
        # disjoint parser partition and keeps invalid mass well-defined.
        sequences = [c.token_ids for c in self.candidates]
        if len(set(sequences)) != len(sequences):
            raise ValueError("candidate token sequences must be unique")
        for a in sequences:
            if any(len(a) < len(b) and a == b[:len(a)] for b in sequences):
                raise ValueError("candidate token sequences must be prefix-free")
        self.temperature_contract.validate(require_same=True)
        splits = {str(w.get("split")) for w in self.worlds}
        if self.worlds and not splits.issubset({"train", "eval", "test"}):
            raise ValueError("world split must be train, eval, or test")

    def parse_text(self, text: str) -> dict[str, Any]:
        raw = str(text)
        value = raw.strip() if self.parser.get("strip_whitespace", True) else raw
        if not self.parser.get("case_sensitive", True):
            lookup = {c.candidate_id.lower(): c for c in self.candidates}
            candidate = lookup.get(value.lower())
        else:
            lookup = {c.candidate_id: c for c in self.candidates}
            candidate = lookup.get(value)
        if candidate is None:
            return {"valid": False, "raw": raw, "normalized": value,
                    "candidate_id": None, "target_state": None}
        return {"valid": True, "raw": raw, "normalized": value,
                "candidate_id": candidate.candidate_id,
                "target_state": candidate.target_state}

    def parse_token_ids(self, token_ids: Sequence[int]) -> dict[str, Any]:
        ids = tuple(int(x) for x in token_ids)
        for c in self.candidates:
            if ids == c.token_ids:
                return {"valid": True, "token_ids": list(ids),
                        "candidate_id": c.candidate_id,
                        "target_state": c.target_state}
        return {"valid": False, "token_ids": list(ids),
                "candidate_id": None, "target_state": None}

    def worlds_by_split(self, split: str) -> list[dict[str, Any]]:
        split = str(split)
        if split not in {"train", "eval", "test"}:
            raise ValueError("split must be train, eval, or test")
        return [dict(world) for world in self.worlds if world.get("split") == split]

    def candidate_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        targets = np.array([self.state_index[c.target_state] for c in self.candidates],
                           dtype=int)
        tokens = np.array([c.token_ids for c in self.candidates], dtype=object)
        sources = None
        if any(c.source_state is not None for c in self.candidates):
            if not all(c.source_state is not None for c in self.candidates):
                raise ValueError("source_state must be specified for every candidate or none")
            sources = np.array([self.state_index[c.source_state] for c in self.candidates],
                               dtype=int)
        return targets, tokens, sources


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


@dataclass
class SemanticKernel:
    probabilities: np.ndarray
    invalid_mass: np.ndarray
    candidate_probabilities: np.ndarray
    candidate_log_probabilities: np.ndarray
    full_vocab_log_normalizer: np.ndarray
    temperature: float
    valid_row_mass: np.ndarray

    def validate(self, atol: float = 1e-10) -> None:
        if self.probabilities.ndim != 2:
            raise ValueError("semantic probabilities must be a square matrix")
        n = self.probabilities.shape[0]
        if self.probabilities.shape[1] != n:
            raise ValueError("semantic probabilities must be square")
        if np.any(self.probabilities < -atol) or np.any(self.invalid_mass < -atol):
            raise ValueError("negative semantic mass")
        if not np.allclose(self.valid_row_mass + self.invalid_mass, 1.0,
                           atol=atol, rtol=0):
            raise ValueError("valid and invalid mass must sum to one")


def semantic_kernel_from_logits(
    logits: np.ndarray,
    candidate_targets: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    *,
    temperature: float = 1.0,
    candidate_sources: Sequence[int] | None = None,
    n_states: int | None = None,
) -> SemanticKernel:
    """Aggregate candidate sequence probabilities into a semantic kernel.

    ``logits`` is either ``[n_source, vocab]`` for one-token candidates or
    ``[n_source, n_candidate, max_length, vocab]``.  In the latter form each
    candidate row contains the logits at its own prefixes, which is what an
    inference runner must save for exact sequence probabilities.  The parser
    candidates are disjoint; probability not assigned to a valid candidate is
    retained in ``invalid_mass`` rather than dropped.
    """
    logits = np.asarray(logits, dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    targets = np.asarray(candidate_targets, dtype=int)
    sequences = [tuple(int(t) for t in seq) for seq in candidate_token_ids]
    m = len(sequences)
    if targets.shape != (m,) or not m:
        raise ValueError("candidate targets and sequences must have equal nonzero length")
    n = logits.shape[0]
    if np.any(targets < 0):
        raise ValueError("candidate targets must be nonnegative")
    if n_states is None:
        n_states = int(targets.max()) + 1
    if n_states <= int(targets.max()):
        raise ValueError("n_states must exceed the largest candidate target")
    if candidate_sources is None:
        sources = np.full(m, -1, dtype=int)
    else:
        sources = np.asarray(candidate_sources, dtype=int)
        if sources.shape != (m,) or np.any((sources >= n) | (sources < 0)):
            raise ValueError("candidate_sources must have one valid source per candidate")
    if logits.ndim == 2:
        if any(len(seq) != 1 for seq in sequences):
            raise ValueError("2-D logits only support one-token candidates")
        vocab = logits.shape[1]
        token_indices = np.array([seq[0] for seq in sequences], dtype=int)
        if np.any((token_indices < 0) | (token_indices >= vocab)):
            raise ValueError("candidate token id outside vocabulary")
        scaled = logits / temperature
        full_lse = _logsumexp(scaled, axis=1)
        logp = scaled[:, token_indices] - full_lse[:, None]
        # A source-specific candidate list can be used for randomized worlds.
        if candidate_sources is not None:
            logp = logp.copy()
            logp[np.arange(n)[:, None] != sources[None, :]] = -np.inf
    elif logits.ndim == 4:
        if logits.shape[1] != m:
            raise ValueError("4-D logits second dimension must equal candidates")
        max_len, vocab = logits.shape[2], logits.shape[3]
        if any(len(seq) > max_len for seq in sequences):
            raise ValueError("candidate sequence longer than logits prefix axis")
        logp = np.full((n, m), -np.inf, dtype=float)
        # Different candidates have different prefixes, so each candidate has
        # its own conditional-logit stack.  Padding positions are ignored.
        for c, seq in enumerate(sequences):
            token_indices = np.asarray(seq, dtype=int)
            if np.any((token_indices < 0) | (token_indices >= vocab)):
                raise ValueError("candidate token id outside vocabulary")
            stack = logits[:, c, :len(seq), :] / temperature
            lse = _logsumexp(stack, axis=2)
            logp[:, c] = np.sum(stack[np.arange(n)[:, None],
                                       np.arange(len(seq))[None, :],
                                       token_indices[None, :]] - lse, axis=1)
        if candidate_sources is not None:
            logp[np.arange(n)[:, None] != sources[None, :]] = -np.inf
        full_lse = _logsumexp(logits / temperature, axis=3)
    else:
        raise ValueError("logits must have shape [n,vocab] or [n,m,L,vocab]")
    probabilities = np.exp(logp)
    probabilities[~np.isfinite(probabilities)] = 0.0
    kernel = np.zeros((n, int(n_states)), dtype=float)
    for c, target in enumerate(targets):
        kernel[:, target] += probabilities[:, c]
    valid_mass = kernel.sum(axis=1)
    if np.any(valid_mass > 1 + 1e-9):
        raise ValueError("candidate parser mass exceeds one; candidates overlap")
    invalid = np.clip(1.0 - valid_mass, 0.0, 1.0)
    result = SemanticKernel(kernel, invalid, probabilities, logp, full_lse,
                            float(temperature), valid_mass)
    result.validate()
    return result


def reconstruct_temperature_from_logits(
    logits: np.ndarray,
    candidate_targets: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    temperatures: Iterable[float],
    *,
    candidate_sources: Sequence[int] | None = None,
    n_states: int | None = None,
) -> dict[float, SemanticKernel]:
    """Rebuild semantic kernels offline from saved raw logits."""
    return {float(t): semantic_kernel_from_logits(
        logits, candidate_targets, candidate_token_ids, temperature=float(t),
        candidate_sources=candidate_sources, n_states=n_states) for t in temperatures}


def reciprocal_pairs(T: np.ndarray, atol: float = 0.0) -> list[tuple[int, int]]:
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    n = T.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if T[i, j] > atol and T[j, i] > atol]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_potential(
    T: np.ndarray,
    pairs: Sequence[tuple[int, int]] | None = None,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
    reward: Sequence[float] | None = None,
    cycles: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """Blindly fit ``log(T_ij/T_ji) ~= phi_i - phi_j`` on reciprocal edges.

    Reward is accepted only for the post-fit alignment field; it never enters
    the least-squares objective.  The fit uses an explicit zero-mean gauge and
    reports a held-out pair score and cycle affinities.
    """
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    n = T.shape[0]
    if pairs is None:
        pairs = reciprocal_pairs(T)
    pairs = [(int(i), int(j)) for i, j in pairs
             if 0 <= i < n and 0 <= j < n and i != j and
             T[i, j] > 0 and T[j, i] > 0]
    if len(pairs) < 2:
        raise ValueError("at least two positive reciprocal pairs are required")
    y = np.array([np.log(T[i, j] / T[j, i]) for i, j in pairs], float)
    A = np.zeros((len(pairs), n), float)
    for row, (i, j) in enumerate(pairs):
        A[row, i] = 1.0
        A[row, j] = -1.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_train = int(round(train_fraction * len(pairs)))
    n_train = min(max(n_train, 1), len(pairs) - 1)
    train = np.sort(order[:n_train]); heldout = np.sort(order[n_train:])
    # The augmented gauge row keeps disconnected or weakly connected toy
    # graphs deterministic; a connectedness warning is returned below.
    aug_A = np.vstack([A[train], np.ones((1, n))])
    aug_y = np.r_[y[train], 0.0]
    phi = np.linalg.lstsq(aug_A, aug_y, rcond=None)[0]
    phi -= phi.mean()
    predicted = A @ phi
    residual = y - predicted
    def rmse(indices: np.ndarray) -> float:
        return float(np.sqrt(np.mean(residual[indices] ** 2))) if len(indices) else float("nan")
    cycles_out = []
    for cycle in cycles:
        cycle = [int(x) for x in cycle]
        if len(cycle) < 2:
            continue
        affinity = 0.0
        valid = True
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            if not (0 <= a < n and 0 <= b < n and T[a, b] > 0 and T[b, a] > 0):
                valid = False
                break
            affinity += np.log(T[a, b] / T[b, a])
        if valid:
            cycles_out.append({"cycle": cycle, "affinity": float(affinity)})
    result: dict[str, Any] = {
        "phi": phi.tolist(),
        "pairs": [list(x) for x in pairs],
        "train_pair_indices": train.tolist(),
        "heldout_pair_indices": heldout.tolist(),
        "log_ratios": y.tolist(),
        "predicted_log_ratios": predicted.tolist(),
        "train_rmse": rmse(train),
        "heldout_rmse": rmse(heldout),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r_pair": _pearson(y, predicted),
        "cycle_affinities": cycles_out,
        "gauge": "mean(phi)=0",
        "fit_blind_to_reward": True,
    }
    if reward is not None:
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (n,):
            raise ValueError("reward must have one value per state")
        result["reward_alignment_corr_minus_phi"] = _pearson(-phi, reward)
        result["reward_alignment_corr_phi"] = _pearson(phi, reward)
    return result


def fit_action_potential(
    T: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    exclude_self: bool = True,
    reward: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit the paper's least-action potential on the semantic state graph.

    The fitted quantity is ``phi = beta*V`` in the dimensionless convention
    used by the paper.  Every positive valid state transition contributes to
    the action; invalid/parser mass is not silently renormalized into another
    semantic state.  Reciprocal pairs are used only after the action fit to
    compute the detailed-balance correlation ``r_action``.

    ``support_mask`` controls which positive edges enter the action.  The
    optional ``pair_mask`` controls which edges are eligible for the reported
    reciprocal statistic; this is how finite-count observed masks are kept
    separate from pseudocount-supported action edges.

    ``fit_potential`` remains available as a pairwise log-ratio least-squares
    diagnostic.  It must not be used as the primary potential estimate.
    """
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a finite square kernel")
    if not np.isfinite(T).all() or np.any(T < 0):
        raise ValueError("T must contain finite nonnegative masses")
    mask = T > 0
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != T.shape:
            raise ValueError("support_mask must have the same shape as T")
        mask &= support_mask
    if exclude_self:
        mask &= ~np.eye(T.shape[0], dtype=bool)
    src, dst = np.where(mask)
    weights = T[src, dst]
    if len(weights) == 0:
        raise ValueError("action fit requires at least one positive non-self edge")
    model = {"src": src.astype(int), "dst": dst.astype(int), "n": int(T.shape[0])}
    phi, action, diagnostics = balance_action(model, weights)
    if pair_mask is None:
        pair_mask = mask
    else:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != T.shape:
            raise ValueError("pair_mask must have the same shape as T")
        pair_mask &= ~np.eye(T.shape[0], dtype=bool) if exclude_self else True
    pairs = reciprocal_pairs(T * pair_mask)
    if pairs:
        x = np.asarray([phi[i] - phi[j] for i, j in pairs], float)
        y = np.asarray([np.log(T[i, j] / T[j, i]) for i, j in pairs], float)
        r = _pearson(x, y)
        residual_rms = float(np.sqrt(np.mean((y - x) ** 2)))
    else:
        x = y = np.array([], dtype=float)
        r = float("nan")
        residual_rms = float("nan")
    result: dict[str, Any] = {
        "phi": phi.tolist(),
        "phi_action": phi.tolist(),
        "action": float(action),
        "action_solver": SOLVER_VERSION,
        "action_diagnostics": diagnostics,
        "action_edges": int(len(weights)),
        "reciprocal_pairs": int(len(pairs)),
        "r_action": float(r),
        "edge_potential_residual_rms": residual_rms,
        "gauge": "mean(phi_action)=0",
        "potential_method": "paper_exponential_action",
        "self_loops_excluded": bool(exclude_self),
    }
    if reward is not None:
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (T.shape[0],):
            raise ValueError("reward must have one value per state")
        result["reward_alignment_corr_minus_phi"] = _pearson(-phi, reward)
        result["reward_alignment_corr_phi"] = _pearson(phi, reward)
    return result


def bootstrap_potential(
    T: np.ndarray, pairs: Sequence[tuple[int, int]], *, n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Pair-bootstrap the blind potential score, retaining failed resamples."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    failures = 0
    pairs = list(pairs)
    for _ in range(n_boot):
        sample = [pairs[int(i)] for i in rng.integers(0, len(pairs), len(pairs))]
        try:
            fit = fit_potential(T, sample, train_fraction=1.0, seed=seed)
            if np.isfinite(fit["r_pair"]):
                values.append(float(fit["r_pair"]))
            else:
                failures += 1
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    if not values:
        raise RuntimeError("all potential bootstrap samples failed")
    arr = np.asarray(values)
    return {"n_boot": int(n_boot), "n_success": len(values),
            "n_failed": failures, "r_pair_mean": float(arr.mean()),
            "r_pair_ci95": np.quantile(arr, [0.025, 0.975]).tolist()}


def bootstrap_action_counts(
    counts: np.ndarray,
    *,
    attempts: Sequence[int] | None = None,
    candidate_mask: np.ndarray | None = None,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    min_edge_count: int = 1,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the paper action fit from finite semantic transition counts.

    ``counts`` contains valid parser-state edge counts.  ``attempts`` may be
    larger than their row sums when failed/invalid outputs were observed; the
    missing mass is sampled as an unreported failure category and is never
    redistributed to semantic states.  Pseudocounts affect reconstruction
    only, while ``min_edge_count`` defines the observed reciprocal mask.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or np.any(counts < 0):
        raise ValueError("counts must be a square nonnegative matrix")
    n = counts.shape[0]
    if attempts is None:
        attempts = counts.sum(axis=1)
    attempts = np.asarray(attempts, dtype=np.int64)
    if attempts.shape != (n,) or np.any(attempts < counts.sum(axis=1)):
        raise ValueError("attempts must cover each row's valid counts")
    if candidate_mask is None:
        raise ValueError("candidate_mask is required: pass the fixed semantic support; "
                         "do not infer support by dropping zero-count edges")
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != counts.shape or np.any(np.diag(candidate_mask)):
        raise ValueError("candidate_mask must be square and exclude self loops")
    if pseudocount <= 0 or min_edge_count < 0 or n_boot < 2:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    actions: list[float] = []
    pairs: list[int] = []
    failures = 0
    degree = candidate_mask.sum(axis=1).astype(float)
    if np.any(degree == 0):
        raise ValueError("every source needs at least one candidate edge")
    for _ in range(int(n_boot)):
        draw = np.zeros_like(counts)
        for i in range(n):
            valid = np.flatnonzero(candidate_mask[i])
            if not len(valid) or attempts[i] == 0:
                continue
            probabilities = np.r_[counts[i, valid] / attempts[i],
                                  max(0, attempts[i] - counts[i].sum()) / attempts[i]]
            sampled = rng.multinomial(int(attempts[i]), probabilities)
            draw[i, valid] = sampled[:-1]
        reconstructed = np.zeros_like(counts, dtype=float)
        for i in range(n):
            valid = candidate_mask[i]
            reconstructed[i, valid] = (draw[i, valid] + pseudocount) / (
                attempts[i] + pseudocount * degree[i])
        try:
            fit = fit_action_potential(
                reconstructed,
                support_mask=candidate_mask,
                pair_mask=draw >= int(min_edge_count),
            )
            if np.isfinite(fit["r_action"]):
                values.append(float(fit["r_action"]))
                actions.append(float(fit["action"]))
                pairs.append(int(fit["reciprocal_pairs"]))
            else:
                failures += 1
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            failures += 1
    if not values:
        raise RuntimeError("all action bootstrap samples failed")
    arr = np.asarray(values)
    return {
        "n_boot": int(n_boot), "n_success": len(values), "n_failed": failures,
        "r_action_mean": float(arr.mean()),
        "r_action_ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "action_mean": float(np.mean(actions)),
        "observed_pair_count_mean": float(np.mean(pairs)),
        "information_boundary": "Counts and candidate_mask only; unseen states/edges are not inferred.",
    }


def enumerate_path_metrics(T: np.ndarray, start: int, horizon: int) -> dict[str, Any]:
    """Enumerate finite-horizon valid paths without turning top-1 into a proof."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    if not 0 <= start < T.shape[0] or horizon <= 0:
        raise ValueError("invalid start or horizon")
    paths: list[tuple[tuple[int, ...], float]] = []
    def visit(node: int, prefix: tuple[int, ...], mass: float, depth: int) -> None:
        if depth == horizon:
            paths.append((prefix, mass))
            return
        for target, probability in enumerate(T[node]):
            if probability > 0:
                visit(target, prefix + (target,), mass * float(probability), depth + 1)
    visit(start, (start,), 1.0, 0)
    masses = np.array([m for _, m in paths], float)
    total = float(masses.sum())
    if total > 0:
        normalized = masses / total
        entropy = float(-np.sum(normalized * np.log(np.maximum(normalized, 1e-300))))
        order = np.argsort(-masses)
        top = masses[order]
    else:
        entropy, top = float("nan"), np.array([])
    possible = len(paths)
    return {"start": int(start), "horizon": int(horizon),
            "path_mass": total, "path_entropy": entropy,
            "effective_paths": float(np.exp(entropy)) if np.isfinite(entropy) else 0.0,
            "top_path_mass": float(top[0]) if len(top) else 0.0,
            "second_path_mass": float(top[1]) if len(top) > 1 else 0.0,
            "positive_path_count": int(np.sum(masses > 0)),
            "enumerated_path_count": int(possible),
            "coverage": float(np.sum(masses > 0) / possible) if possible else 0.0,
            "paths": [{"states": list(p), "mass": float(m)} for p, m in paths]}


def path_metrics(T: np.ndarray, starts: Sequence[int], horizon: int) -> dict[str, Any]:
    rows = [enumerate_path_metrics(T, int(s), horizon) for s in starts]
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")
    return {"horizon": int(horizon), "starts": [int(s) for s in starts],
            "per_start": rows,
            "mean_path_mass": mean("path_mass"),
            "mean_top_path_mass": mean("top_path_mass"),
            "mean_second_path_mass": mean("second_path_mass"),
            "mean_path_entropy": mean("path_entropy"),
            "mean_effective_paths": mean("effective_paths"),
            "mean_coverage": mean("coverage")}


def reverse_stats(T: np.ndarray, *, threshold: float = 0.0) -> dict[str, Any]:
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    directed = int(np.sum(T > threshold) - np.sum(np.diag(T) > threshold))
    reciprocal = sum(T[i, j] > threshold and T[j, i] > threshold
                     for i in range(n) for j in range(i + 1, n))
    possible_pairs = n * (n - 1) // 2
    reverse_mass = [T[i, j] for i in range(n) for j in range(i + 1, n)
                    if T[i, j] > threshold and T[j, i] > threshold]
    return {"directed_nonself_edges": directed,
            "reciprocal_pairs": int(reciprocal),
            "reciprocal_coverage": float(reciprocal / possible_pairs)
            if possible_pairs else 0.0,
            "mean_reciprocal_forward_mass": float(np.mean(reverse_mass))
            if reverse_mass else 0.0,
            "threshold": float(threshold)}


def make_manifest(
    task: TaskSpec, *, model: Mapping[str, Any], seed: int,
    temperature: TemperatureContract | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = temperature or task.temperature_contract
    contract.validate(require_same=True)
    manifest = {
        "schema_version": "m2.manifest.v1",
        "task_id": task.task_id,
        "task_hash": task.task_hash,
        "reward_hash": task.reward_hash,
        "model": _jsonable(dict(model)),
        "seed": int(seed),
        "temperature_contract": contract.as_dict(),
        "temperature_hash": sha256_json(contract.as_dict()),
        "environment": _jsonable(dict(environment or {})),
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def write_transitions(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            required = {"source_state", "output", "parsed_state", "reward", "tau"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"transition missing fields: {sorted(missing)}")
            handle.write(canonical_json(dict(row)) + "\n")


def read_transitions(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"source_state", "output", "parsed_state", "reward", "tau"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        rows.append(row)
    return rows


def write_logits(path: str | Path, *, logits: np.ndarray,
                 candidate_targets: Sequence[int],
                 candidate_token_ids: Sequence[Sequence[int]],
                 metadata: Mapping[str, Any] | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=np.asarray(logits, dtype=float),
                        candidate_targets=np.asarray(candidate_targets, dtype=int),
                        candidate_token_ids=np.array(
                            [list(x) for x in candidate_token_ids], dtype=object),
                        metadata=np.array(canonical_json(dict(metadata or {}))))


def read_logits(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {"logits": data["logits"],
                "candidate_targets": data["candidate_targets"],
                "candidate_token_ids": [tuple(int(x) for x in seq)
                                         for seq in data["candidate_token_ids"]],
                "metadata": json.loads(str(data["metadata"].item()))}
