"""Tau=1 checkpoint samples; formal admission requires a frozen contract.

Keep all attempts and invalid mass. Pseudocounts never create observed support.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

M2_ROOT = Path(__file__).resolve().parent
M1_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(M2_ROOT))
sys.path.insert(0, str(M1_CODE))
from pathlib import Path as _ArchivePath
import sys as _archive_sys
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[2] / 'analysis'))
from action_precision import balance_action  # noqa: E402
from m2_protocol import score_response  # noqa: E402

MODEL = os.environ.get('ENERGY_MODEL', 'Qwen/Qwen3.5-9B')
ALPHA = 0.1


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def task_graph(records):
    labels = [json.loads(row["label"]) for row in records]
    n = len(records)
    edges = sorted({(int(label["source"]), int(target))
                    for label in labels for target in label["mapping"].values()
                    if int(target) != int(label["source"])})
    edge_index = {edge: i for i, edge in enumerate(edges)}
    src = np.asarray([edge[0] for edge in edges], dtype=int)
    dst = np.asarray([edge[1] for edge in edges], dtype=int)
    if len(edges) == 0 or len(set(src)) != n:
        raise ValueError("frozen candidate graph has missing source support")
    return labels, edges, edge_index, src, dst, n


def counts_from_rows(labels, edges, edge_index, samples):
    counts = np.zeros(len(edges), dtype=np.int64)
    invalid = np.zeros(len(labels), dtype=np.int64)
    reason_counts = [Counter() for _ in labels]
    for source, rows in enumerate(samples):
        for row in rows:
            result = score_response(row["text"], row["truncated"], labels[source])
            target = result["target"]
            if target is None:
                invalid[source] += 1
                reason_counts[source][result["invalid_reason"] or "invalid"] += 1
            else:
                counts[edge_index[(source, int(target))]] += 1
    return counts, invalid, reason_counts


def compute_metric(counts, invalid, src, dst, n, *, alpha=ALPHA):
    degree = np.bincount(src, minlength=n).astype(float)
    attempts = invalid + np.bincount(src, weights=counts, minlength=n).astype(np.int64)
    t = (counts.astype(float) + alpha) / (attempts[src] + alpha * degree[src])
    model = {"src": src, "dst": dst, "n": n}
    phi, action, diagnostics = balance_action(model, t)
    if not diagnostics['converged']:
        raise RuntimeError(f'Action solver did not converge: {diagnostics}')
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(src, dst))}
    left, right = [], []
    observed = counts >= 1
    for (i, j), k in lookup.items():
        if i < j and (j, i) in lookup:
            other = lookup[(j, i)]
            if observed[k] and observed[other]:
                left.append(k)
                right.append(other)
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    r = corr(phi[src[left]] - phi[dst[left]], np.log(t[left]) - np.log(t[right])) \
        if len(left) >= 3 else None
    raw_r = corr(phi[src[left]] - phi[dst[left]],
                 np.log(counts[left] / attempts[src[left]])
                 - np.log(counts[right] / attempts[src[right]])) if len(left) >= 3 else None
    return {
        "r": r,
        "raw_count_logratio_r": raw_r,
        "reciprocal_pairs_observed": int(len(left)),
        "reciprocal_pairs_total": int(sum(
            1 for (i, j) in lookup if i < j and (j, i) in lookup)),
        "phi_action": phi.tolist(),
        "action": float(action),
        "action_solver": diagnostics,
        "counts": counts.tolist(),
        "invalid_counts": invalid.tolist(),
        "attempts": attempts.tolist(),
        "invalid_rate": float(invalid.sum() / max(attempts.sum(), 1)),
        "valid_rate": float(counts.sum() / max(attempts.sum(), 1)),
        "t_hat": t.tolist(),
        "observed_edge_mask": observed.tolist(),
        "alpha": alpha,
    }


def bootstrap_metric(counts, invalid, src, dst, n, *, replicates, seed, rewards=None):
    rng = np.random.default_rng(seed)
    attempts = invalid + np.bincount(src, weights=counts, minlength=n).astype(np.int64)
    samples = []
    alignments = []
    for _ in range(replicates):
        c = np.zeros_like(counts)
        bad = np.zeros_like(invalid)
        for i in range(n):
            ix = np.flatnonzero(src == i)
            probs = np.r_[counts[ix], invalid[i]].astype(float)
            probs /= probs.sum()
            draw = rng.multinomial(int(attempts[i]), probs)
            c[ix] = draw[:-1]
            bad[i] = draw[-1]
        fitted = compute_metric(c, bad, src, dst, n)
        value = fitted["r"]
        if value is not None:
            samples.append(value)
        if rewards is not None:
            alignment = corr(-np.asarray(fitted['phi_action']), rewards)
            if alignment is not None:
                alignments.append(alignment)
    if len(samples) < max(10, replicates // 4):
        return {"finite_replicates": len(samples), "replicates": replicates,
                "sigma": None, "ci95": None}
    arr = np.asarray(samples)
    return {"finite_replicates": int(len(arr)), "replicates": replicates,
            "sigma": float(arr.std(ddof=1)),
            "ci95": [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))],
            "reward_alignment_ci95": np.quantile(alignments, [.025, .975]).tolist()
                if alignments else None}


def generate_source(model, tokenizer, prompt, label, *, samples, batch_size, seed):
    rows = []
    torch.manual_seed(seed)
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        batch = tokenizer([prompt] * size, add_special_tokens=False,
                          return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                # Keep the HF evaluator's sampling contract identical to the
                # OpenRLHF/vLLM rollout contract.  Transformers' implicit
                # GenerationConfig has top_k=50 and min_new_tokens=0; leaving
                # either implicit would silently measure a different kernel.
                **batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                min_new_tokens=1, repetition_penalty=1.0,
                max_new_tokens=int(label["max_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prefix_len = batch.input_ids.shape[1]
        for ids in generated[:, prefix_len:].tolist():
            eos = tokenizer.eos_token_id
            hit = next((i for i, token in enumerate(ids) if token == eos), None)
            if hit is None:
                text_ids, truncated = ids, len(ids) >= int(label["max_tokens"])
            else:
                text_ids, truncated = ids[:hit], False
            rows.append({"text": tokenizer.decode(text_ids, skip_special_tokens=False),
                         "token_ids": ids, "truncated": bool(truncated)})
        del batch, generated
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--tasks", nargs="+", choices=["T1", "T2"], default=["T1", "T2"])
    ap.add_argument("--adapter")
    ap.add_argument("--samples-per-source", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--bootstrap-replicates", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--formal-contract")
    ap.add_argument("--policy-id")
    args = ap.parse_args()
    admission = {'formal_admissible': False}
    if args.formal_contract:
        from formal_measurement_contract import validate_request
        admission = validate_request(args.formal_contract, args.policy_id,
                                     args.prepared, args.adapter, MODEL)
        assert args.samples_per_source in admission['policy']['sample_budgets']
        assert args.seed == admission['policy']['sample_seeds'][str(args.samples_per_source)]
        contract = json.loads(Path(args.formal_contract).read_text())
        assert args.batch_size == contract['sampling']['batch_size']
        assert args.bootstrap_replicates == contract['sampling']['bootstrap_replicates']
    if args.samples_per_source < 4 or args.batch_size < 1:
        raise ValueError("too few samples or invalid batch size")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager",
        local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False) \
        if args.adapter else base
    model.eval()
    env = {
        "versions": {k: importlib.metadata.version(k)
                     for k in ("torch", "transformers", "peft", "tokenizers")},
        "model": MODEL, "adapter": args.adapter,
        "adapter_sha256": sha256(Path(args.adapter, "adapter_model.safetensors"))
            if args.adapter else None,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": 1, "repetition_penalty": 1.0, "thinking": False,
        "samples_per_source": args.samples_per_source,
        "batch_size": args.batch_size,
        "attention_implementation": "eager",
        "bootstrap_replicates": args.bootstrap_replicates,
        "gpu": torch.cuda.get_device_name(0), **admission,
        "seed": args.seed, "task_seed_offsets": {"T1": 0, "T2": 20},
    }
    atomic_json(out / "environment.json", env)
    all_metrics = {}
    for task in args.tasks:
        task_seed = args.seed + (20 if task == 'T2' else 0)
        prepared_path = Path(args.prepared, f"{task}_prepared.json")
        records = json.loads(prepared_path.read_text())
        labels, edges, edge_index, src, dst, n = task_graph(records)
        samples = []
        raw_path = out / f"{task}_samples.jsonl"
        started = time.monotonic()
        with raw_path.open("w") as raw:
            for source, record in enumerate(records):
                label = labels[source]
                rows = generate_source(model, tok, record["input"], label,
                                       samples=args.samples_per_source,
                                       batch_size=args.batch_size,
                                       seed=task_seed + source)
                samples.append(rows)
                for row in rows:
                    raw.write(json.dumps({"task": task, "source": source, **row}, ensure_ascii=False) + "\n")
                raw.flush()
                os.fsync(raw.fileno())
                print(json.dumps({'task': task, 'source': source, 'samples': len(rows),
                                  'seconds_so_far': time.monotonic() - started}), flush=True)
                if torch.cuda.max_memory_allocated() > .90 * torch.cuda.get_device_properties(0).total_memory:
                    raise RuntimeError('GPU allocation exceeded the resource gate')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 > .85 * float(os.getenv('M2_MEMORY_GIB', '96')):
                    raise RuntimeError('Host memory exceeded the resource gate')
        counts, invalid, reasons = counts_from_rows(labels, edges, edge_index, samples)
        metric = compute_metric(counts, invalid, src, dst, n)
        metric["bootstrap"] = bootstrap_metric(
            counts, invalid, src, dst, n, replicates=args.bootstrap_replicates,
            seed=task_seed + 100000,
            rewards=[float(labels[0]['state_rewards'][str(i)]) for i in range(n)])
        metric.update({
            "task": task, "task_hash": hashlib.sha256(
                Path(M2_ROOT, "tasks", f"{task}_direct_formal_v1.json").read_bytes()).hexdigest(),
            "prepared_sha256": sha256(prepared_path), "n_states": n,
            "edges": edges, "reason_counts": [dict(x) for x in reasons],
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "reward_alignment_corr_minus_phi_u": corr(
                -np.asarray(metric["phi_action"]),
                [float(labels[0]["state_rewards"][str(i)]) for i in range(n)]),
            **admission,
            "sampling_seed": task_seed, "bootstrap_seed": task_seed + 100000,
        })
        all_metrics[task] = metric
        atomic_json(out / f"{task}_metrics.json", metric)
        print(json.dumps({"task": task, "r": metric["r"],
                          "r_bootstrap": metric["bootstrap"],
                          "invalid_rate": metric["invalid_rate"],
                          "seconds": metric["wall_seconds"]}), flush=True)
    atomic_json(out / "measurement_summary.json", {
        "status": "CHECKPOINT_SAMPLE_MEASUREMENT_COMPLETE",
        **admission, "metrics": all_metrics,
    })


if __name__ == "__main__":
    main()
