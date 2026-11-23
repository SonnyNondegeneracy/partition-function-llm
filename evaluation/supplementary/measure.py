"""Independent complete-generation counts, with bounded smoke and replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from common import ROOT, HISTORY, atomic_json, digest, writable
from protocol import check_parser


def run(args):
    out = writable(args.output)
    out.mkdir(parents=True, exist_ok=False)
    registry = json.loads(Path(args.registry).read_text())
    policy_registry = json.loads(Path(args.policies).read_text())
    model_path = policy_registry["model"]
    if registry["model"] != model_path:
        raise ValueError("Tokenizer/model task registry mismatch")
    atomic_json(out / "input_contract.json", {
        "registry": str(Path(args.registry).resolve()), "policies": str(Path(args.policies).resolve()),
        "selected_policies": args.policy, "selected_tasks": args.tasks,
        "mode": args.mode, "samples_per_source": args.samples, "sources": args.sources,
        "registry_sha256": digest(args.registry), "policies_sha256": digest(args.policies)})
    check_parser()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    adapters = [policy_registry["policies"][p]["adapter"] for p in args.policy]
    context_limit = max(2048, max(
        len(tokenizer.encode(row["input"], add_special_tokens=False)) + json.loads(row["label"])["max_tokens"]
        for task in args.tasks for row in json.loads(Path(
            registry["tasks"][task]["prepared"][args.mode]["path"]).read_text())))
    max_seqs = 32 if args.mode == "thinking" else 64
    before = time.monotonic()
    llm = LLM(model=model_path, dtype="bfloat16", enforce_eager=not args.cuda_graphs,
              language_model_only=True, tensor_parallel_size=1,
              max_model_len=context_limit, max_num_seqs=max_seqs, max_num_batched_tokens=args.prefill_tokens,
              gpu_memory_utilization=.65, enable_prefix_caching=True,
              enable_lora=True, max_lora_rank=16, max_loras=1,
              max_cpu_loras=max(1, len(adapters)), seed=args.seed,
              **({"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}} if args.cuda_graphs else {}))
    env = {"backend": "vllm_complete_generation_counts", "model": model_path,
           "model_config_sha256": digest(Path(model_path) / "config.json"),
           "registry_sha256": digest(args.registry), "policies_sha256": digest(args.policies),
           "versions": {k: importlib.metadata.version(k) for k in
                        ("torch", "transformers", "vllm", "peft", "tokenizers")},
           "sampling": {"temperature": 1., "top_p": 1., "top_k": -1,
                        "min_tokens": 1, "repetition_penalty": 1., "seed": args.seed,
                        "samples_per_source": args.samples, "chunk_size": args.chunk_size},
           "engine": {"eager": not args.cuda_graphs, "cuda_graphs": args.cuda_graphs,
                      "max_num_seqs": max_seqs, "max_num_batched_tokens": args.prefill_tokens,
                      "max_model_len": context_limit, "prefix_caching": True,
                      "gpu_memory_utilization": .65, "adapter_mode": "native_vllm_lora",
                      "enable_lora": True},
           "load_seconds": time.monotonic() - before, "mode": args.mode,
           "data_role": "extension_smoke_only" if args.smoke else "m2_supplementary_measurement",
           "formal_m2_admissible": False}
    atomic_json(out / "environment.json", env)
    from metrics import summarize
    for policy_number, policy_name in enumerate(args.policy):
        policy = policy_registry["policies"][policy_name]
        adapter = policy["adapter"]
        if adapter and digest(Path(adapter) / "adapter_model.safetensors") != policy["adapter_sha256"]:
            raise ValueError(f"Adapter hash mismatch: {policy_name}")
        lora = LoRARequest(policy_name, policy_number + 1, adapter) if adapter else None
        dest = out / policy_name
        dest.mkdir()
        summaries = {}
        for task_number, task in enumerate(args.tasks):
            prepared = registry["tasks"][task]["prepared"][args.mode]
            if digest(prepared["path"]) != prepared["sha256"]:
                raise ValueError(f"Frozen task changed: {task}")
            records = json.loads(Path(prepared["path"]).read_text())
            rows = []
            started = time.monotonic()
            raw_path = dest / f"{task}_samples.jsonl"
            task_seed = args.seed + policy_number * 1000000 + task_number * 100000
            with raw_path.open("w") as raw:
                for source, record in enumerate(records):
                    if args.sources is not None and source not in args.sources:
                        continue
                    label = json.loads(record["label"])
                    if hashlib.sha256(record["input"].encode()).hexdigest() != label["rendered_prompt_sha256"]:
                        raise ValueError("Prompt hash mismatch")
                    for start in range(0, args.samples, args.chunk_size):
                        size = min(args.chunk_size, args.samples - start)
                        params = [SamplingParams(temperature=1., top_p=1., top_k=-1,
                                  repetition_penalty=1., min_tokens=1, max_tokens=label["max_tokens"],
                                  seed=task_seed + source * 10000 + start + k,
                                  skip_special_tokens=False) for k in range(size)]
                        prompts = [{"prompt_token_ids": tokenizer.encode(record["input"], add_special_tokens=False)}] * size
                        outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
                        if len(outputs) != size:
                            raise RuntimeError("Generation lost a request")
                        for k, output in enumerate(outputs):
                            if len(output.outputs) != 1:
                                raise RuntimeError("Expected one independent completion per request")
                            completion = output.outputs[0]
                            ids = list(completion.token_ids)
                            terminals = label.get("terminal_token_ids", [tokenizer.eos_token_id])
                            text_ids = ids[:-1] if ids and ids[-1] in terminals else ids
                            row = {"task": task, "source": source, "sample": start + k,
                                   "text": tokenizer.decode(text_ids, skip_special_tokens=False),
                                   "token_ids": ids, "truncated": completion.finish_reason == "length",
                                   "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                                   "seed": task_seed + source * 10000 + start + k}
                            rows.append(row)
                            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                        raw.flush()
                        os.fsync(raw.fileno())
                    print(json.dumps({"policy": policy_name, "task": task, "source": source,
                                      "samples": args.samples, "seconds": time.monotonic() - started}), flush=True)
            if args.sources is None:
                result = summarize(records, rows, args.bootstrap, task_seed + 99999)
            else:
                from protocol import score_response
                import numpy as np
                scores = [score_response(r["text"], r["truncated"], json.loads(records[r["source"]]["label"])) for r in rows]
                lengths = [len(r["token_ids"]) for r in rows]
                result = {"r": None, "reward_alignment_corr_minus_phi_u": None,
                          "status": "PARTIAL_SOURCE_ENGINEERING_SMOKE_NO_KERNEL_FIT",
                          "invalid_rate": float(np.mean([s["target"] is None for s in scores])),
                          "mean_reward": float(np.mean([s["reward"] for s in scores])),
                          "truncated_rate": float(np.mean([r["truncated"] for r in rows])),
                          "generation_tokens": {"mean": float(np.mean(lengths)),
                              "quantiles": np.quantile(lengths, [.5, .9, .95, .99, 1]).tolist()}}
            result.update({"policy": policy_name, "task": task, "mode": args.mode,
                           "beta_est": .375 / policy["round"] if policy["round"] else None,
                           "round": policy["round"], "adapter_sha256": policy.get("adapter_sha256"),
                           "prepared_sha256": prepared["sha256"], "raw_sha256": digest(raw_path),
                           "wall_seconds": time.monotonic() - started, "sampling_seed": task_seed,
                           "bootstrap_seed": task_seed + 99999, "data_role": env["data_role"]})
            summaries[task] = result
            atomic_json(dest / f"{task}_metrics.json", result)
            print(json.dumps({"policy": policy_name, "task": task, "r": result["r"],
                              "corr": result["reward_alignment_corr_minus_phi_u"],
                              "reward": result["mean_reward"], "invalid": result["invalid_rate"],
                              "tokens": result["generation_tokens"]}), flush=True)
        atomic_json(dest / "summary.json", {"status": "SUPPLEMENTARY_COUNTS_COMPLETE", "metrics": summaries})
    atomic_json(out / "complete.json", {"status": "COMPLETE", "policies": args.policy, "tasks": args.tasks})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--policy", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2"])
    ap.add_argument("--mode", choices=["direct", "thinking"], required=True)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--prefill-tokens", type=int, default=1024)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sources", nargs="+", type=int)
    args = ap.parse_args()
    if args.samples < (1 if args.smoke else 4) or args.chunk_size < 1:
        raise ValueError("Invalid sample budget")
    if args.sources is not None and not args.smoke:
        raise ValueError("Source subsampling is engineering smoke only")
    run(args)


if __name__ == "__main__":
    main()
