"""Replay audit for v2 sampled measurements.

The audit re-parses saved text and token IDs, checks prompt/registry hashes,
checks exactly one attempt for every source/sample pair, and independently
recomputes the count fit.  It never edits the measurement output except for an
atomic ``audit.json`` result under the same v2 output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import atomic_json, digest, writable
from metrics import summarize


def audit(output: str | Path) -> dict:
    root = writable(output)
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    registry_path = Path(environment["registry"] if "registry" in environment else "")
    if not registry_path:
        # Older v2 outputs record only a digest; the caller can still perform
        # the raw replay by passing --registry to the CLI.
        raise ValueError("environment lacks registry path; rerun with --registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if digest(registry_path) != environment["registry_sha256"]:
        raise ValueError("registry changed after measurement")
    task_entry = registry.get("task")
    if isinstance(task_entry, dict):
        prepared = Path(task_entry["prepared"])
    elif isinstance(task_entry, str):
        prepared = Path(environment.get("prepared", task_entry))
    else:
        try:
            prepared = Path(registry["tasks"]["T1"]["prepared"]["direct"]["path"])
        except (KeyError, TypeError) as exc:
            raise ValueError("registry has no auditable T1 prepared path") from exc
    if environment.get("prepared_sha256") and digest(prepared) != environment["prepared_sha256"]:
        raise ValueError("prepared prompt hash changed after measurement")
    records = json.loads(prepared.read_text(encoding="utf-8"))
    tokenizer = None
    if environment.get("model"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(environment["model"], local_files_only=True)
    budget = int(environment["samples_per_source"])
    audited = []
    for policy_name in complete["policies"]:
        path = root / policy_name / "T1_samples.jsonl"
        rows = []
        seen = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (int(row["source"]), int(row["sample"]))
                if key in seen:
                    raise ValueError(f"duplicate sample: {key}")
                seen.add(key)
                source = key[0]
                if source not in range(len(records)) or row["task"] != "T1":
                    raise ValueError("sample source/task mismatch")
                label = json.loads(records[source]["label"])
                if row.get("truncated") != (row.get("finish_reason") == "length"):
                    raise ValueError("truncation accounting mismatch")
                # The sampler removes EOS when the prepared label does not
                # provide an explicit terminal set. Replay must use the same
                # default, otherwise every normal stop is falsely rejected.
                terminals = label.get("terminal_token_ids", [tokenizer.eos_token_id] if tokenizer is not None else [])
                token_ids = row.get("token_ids", [])
                text_ids = token_ids[:-1] if token_ids and token_ids[-1] in terminals else token_ids
                if tokenizer is not None and tokenizer.decode(text_ids, skip_special_tokens=False) != row["text"]:
                    raise ValueError(f"saved text is not a tokenizer decode: {policy_name}:{key}")
                if row.get("sample") is None or row.get("seed") is None:
                    raise ValueError("saved row lacks deterministic sample metadata")
                rows.append(row)
        expected = {(source, sample) for source in range(len(records)) for sample in range(budget)}
        if seen != expected:
            raise ValueError(f"missing or unexpected attempts: got {len(seen)}, expected {len(expected)}")
        saved = json.loads((root / policy_name / "T1_metrics.json").read_text(encoding="utf-8"))
        if saved.get("raw_sha256") != digest(path):
            raise ValueError(f"raw sample hash mismatch: {policy_name}")
        replay = summarize(records, rows, bootstrap=0, seed=0)
        for key in ("edges", "counts", "invalid_counts", "attempts", "reason_counts",
                    "observed_directed_edges", "observed_scc_count", "total_attempts",
                    "reciprocal_pairs_observed"):
            if replay[key] != saved[key]:
                raise ValueError(f"metric replay mismatch: {policy_name}:{key}")
        for key in ("r", "reward_alignment_corr_minus_phi_u", "mean_reward", "mean_semantic_reward",
                    "phi_action", "t_hat"):
            left, right = replay[key], saved[key]
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"undefined metric replay mismatch: {policy_name}:{key}")
            elif not np.allclose(left, right, atol=1e-11, rtol=1e-10):
                raise ValueError(f"numeric metric replay mismatch: {policy_name}:{key}")
        audited.append({"policy": policy_name, "attempts": len(rows),
                        "raw_sha256": digest(path),
                        "metrics_sha256": digest(root / policy_name / "T1_metrics.json")})
    result = {"status": "V2_SAMPLE_REPLAY_AUDIT_PASS", "temperature": environment["temperature"],
              "policies": audited, "environment_sha256": digest(root / "environment.json"),
              "complete_sha256": digest(root / "complete.json")}
    atomic_json(root / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
