"""Sample the frozen T1 policy registry and fit the M1 transition metric.

This executable is deliberately independent of the formal M2 outputs.  It
accepts one or more policies from ``t1_registry.json`` and writes one output
tree per policy.  ``--smoke`` is bounded to eight attempts/source; a normal
run defaults to the registered 4096 attempts/source contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from common import MODEL_DEFAULT, atomic_json, digest, runtime_contract, writable
from metrics import summarize
from protocol import check_parser


def _versions(packages):
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _sample(args, registry, policy_names):
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("sample measurement requires the Magnus vLLM runtime") from exc

    model_path = registry["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    task = registry["task"]
    records = json.loads(Path(task["prepared"]).read_text(encoding="utf-8"))
    if digest(task["prepared"]) != task["prepared_sha256"]:
        raise ValueError("T1 prepared prompts changed after registry freeze")
    if digest(Path(model_path) / "config.json") != registry["model_config_sha256"]:
        raise ValueError("model config changed after registry freeze")

    # The CLI default is the full budget, but --smoke intentionally maps that
    # default to the bounded eight-attempt gate. An explicit 16 is also useful
    # for a slightly wider engineering smoke.
    if args.smoke:
        if args.samples_per_source not in (8, 16, 4096):
            raise ValueError("smoke budget must be 8 or 16 attempts/source")
        sample_count = 8 if args.samples_per_source == 4096 else args.samples_per_source
    else:
        sample_count = args.samples_per_source
    if not args.smoke and sample_count != 4096:
        raise ValueError("full v2 sample contract is exactly 4096 attempts/source")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        raise ValueError("temperature must be finite and positive")

    max_model_len = max(len(tokenizer.encode(row["input"], add_special_tokens=False))
                        + json.loads(row["label"])["max_tokens"] for row in records)
    llm = LLM(model=model_path, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel,
              max_model_len=max(2048, max_model_len), max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.prefill_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=16,
              max_loras=1, max_cpu_loras=max(1, len(policy_names)), seed=args.seed,
              enforce_eager=not args.cuda_graphs)
    output = writable(args.output)
    output.mkdir(parents=True, exist_ok=False)
    package_versions = _versions(("torch", "transformers", "vllm", "tokenizers"))
    environment = {
        "schema_version": "m2.extension_v2.sample_environment.v1",
        "data_role": "extension_v2_smoke" if args.smoke else "extension_v2_supplementary",
        "model": model_path, "model_config_sha256": registry["model_config_sha256"],
        "registry": str(Path(args.registry).resolve()), "registry_sha256": digest(args.registry), "task": "T1",
        "prepared": str(Path(task["prepared"]).resolve()),
        "prepared_sha256": task["prepared_sha256"],
        "policies": policy_names, "temperature": float(args.temperature),
        "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
        "min_tokens": 1, "thinking": False, "samples_per_source": sample_count,
        "chunk_size": args.chunk_size, "bootstrap_replicates": args.bootstrap,
        "seed": args.seed, "versions": package_versions,
        "runtime_contract": runtime_contract(package_versions),
        "engine": {"tensor_parallel_size": args.tensor_parallel, "max_num_seqs": args.max_num_seqs,
                    "max_num_batched_tokens": args.prefill_tokens,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "cuda_graphs": bool(args.cuda_graphs), "prefix_caching": True},
    }
    atomic_json(output / "environment.json", environment)
    policy_outputs = []
    for policy_number, policy_name in enumerate(policy_names):
        policy = registry["policies"][policy_name]
        adapter = policy.get("adapter")
        if adapter:
            adapter_dir = Path(adapter)
            if digest(adapter_dir / "adapter_model.safetensors") != policy["adapter_sha256"]:
                raise ValueError(f"adapter hash mismatch: {policy_name}")
            config_hash = digest(adapter_dir / "adapter_config.json")
            expected_config_hash = policy.get("adapter_config_sha256")
            if expected_config_hash is not None and config_hash != expected_config_hash:
                raise ValueError(f"adapter config hash mismatch: {policy_name}")
            # Full trajectories created by the first v2 driver predate the
            # config-hash field.  Keep those immutable policies measurable,
            # while recording the computed hash so the audit remains explicit.
            policy_config_hash = config_hash
        else:
            policy_config_hash = None
        lora = LoRARequest(policy_name, policy_number + 1, adapter) if adapter else None
        policy_out = output / policy_name
        policy_out.mkdir()
        rows = []
        started = time.monotonic()
        with (policy_out / "T1_samples.jsonl").open("w", encoding="utf-8") as raw:
            for source, record in enumerate(records):
                label = json.loads(record["label"])
                if hashlib.sha256(record["input"].encode()).hexdigest() != label["rendered_prompt_sha256"]:
                    raise ValueError(f"prompt hash mismatch at source {source}")
                for start in range(0, sample_count, args.chunk_size):
                    size = min(args.chunk_size, sample_count - start)
                    sampling = [SamplingParams(temperature=float(args.temperature), top_p=1.0, top_k=-1,
                                               repetition_penalty=1.0, min_tokens=1,
                                               max_tokens=int(label["max_tokens"]), skip_special_tokens=False,
                                               seed=args.seed + policy_number * 1_000_000
                                               + source * 10_000 + start + offset)
                                for offset in range(size)]
                    prompts = [{"prompt_token_ids": tokenizer.encode(record["input"], add_special_tokens=False)}] * size
                    results = llm.generate(prompts, sampling, lora_request=lora, use_tqdm=False)
                    if len(results) != size:
                        raise RuntimeError("vLLM returned a different number of completions")
                    for offset, result in enumerate(results):
                        if len(result.outputs) != 1:
                            raise RuntimeError("expected one completion per vLLM request")
                        completion = result.outputs[0]
                        token_ids = list(completion.token_ids)
                        terminals = label.get("terminal_token_ids", [tokenizer.eos_token_id])
                        text_ids = token_ids[:-1] if token_ids and token_ids[-1] in terminals else token_ids
                        row = {"task": "T1", "source": source, "sample": start + offset,
                               "text": tokenizer.decode(text_ids, skip_special_tokens=False),
                               "token_ids": token_ids,
                               "truncated": completion.finish_reason == "length",
                               "finish_reason": completion.finish_reason,
                               "stop_reason": completion.stop_reason,
                               "seed": args.seed + policy_number * 1_000_000 + source * 10_000 + start + offset}
                        rows.append(row)
                        raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw.flush()
                    os.fsync(raw.fileno())
        summary = summarize(records, rows, bootstrap=args.bootstrap,
                            seed=args.seed + policy_number * 1_000_000 + 999_999)
        summary.update({"policy": policy_name, "round": policy["round"],
                        "temperature": float(args.temperature),
                        "samples_per_source": sample_count,
                        "adapter_config_sha256": policy_config_hash,
                        "adapter_config_hash_verified": policy.get("adapter_config_sha256") is not None
                            if adapter else True,
                        "raw_sha256": digest(policy_out / "T1_samples.jsonl"),
                        "wall_seconds": time.monotonic() - started,
                        "registry_sha256": digest(args.registry),
                        "data_role": environment["data_role"]})
        atomic_json(policy_out / "T1_metrics.json", summary)
        policy_outputs.append({"policy": policy_name, "round": policy["round"],
                              "r": summary["r"], "corr_minus_phi_u": summary["reward_alignment_corr_minus_phi_u"],
                              "invalid_rate": summary["invalid_rate"],
                              "reciprocal_pairs": summary["reciprocal_pairs_observed"]})
    atomic_json(output / "complete.json", {"status": "COMPLETE", "task": "T1",
                                            "policies": policy_names,
                                            "temperature": float(args.temperature),
                                            "samples_per_source": sample_count,
                                            "results": policy_outputs})
    return policy_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parent / "t1_registry.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", "--policies", nargs="+", dest="policies", default=["base"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--samples-per-source", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=82020000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    check_parser()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    unknown = [name for name in args.policies if name not in registry["policies"]]
    if unknown:
        raise ValueError(f"unknown T1 policies: {unknown}")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("duplicate policy selection")
    print(json.dumps({"status": "STARTING_V2_SAMPLE", "policies": args.policies,
                      "temperature": args.temperature, "smoke": args.smoke}, ensure_ascii=False), flush=True)
    result = _sample(args, registry, args.policies)
    print(json.dumps({"status": "COMPLETE", "results": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
