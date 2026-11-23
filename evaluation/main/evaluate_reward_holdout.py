"""Independent held-out reward audit for calibration checkpoints.

This script is deliberately separate from the training rollout.  Its output
is calibration-only and is never accepted as formal Q/T/r data.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from m2_protocol import score_response  # noqa: E402

MODEL = os.environ.get('ENERGY_MODEL', 'Qwen/Qwen3.5-9B')


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generate(model, tokenizer, prompt: str, label: dict, samples: int,
             batch_size: int, seed: int):
    rows = []
    torch.manual_seed(seed)
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        batch = tokenizer([prompt] * size, add_special_tokens=False,
                          return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                min_new_tokens=1, repetition_penalty=1.0,
                max_new_tokens=int(label["max_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = batch.input_ids.shape[1]
        for ids in output[:, prompt_len:].tolist():
            eos = tokenizer.eos_token_id
            hit = next((i for i, token in enumerate(ids) if token == eos), None)
            if hit is None:
                text_ids, truncated = ids, len(ids) >= int(label["max_tokens"])
            else:
                text_ids, truncated = ids[:hit], False
            rows.append({
                "text": tokenizer.decode(text_ids, skip_special_tokens=False),
                "token_ids": ids,
                "truncated": bool(truncated),
            })
        del batch, output
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--task", choices=["T1", "T2"], required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--samples-per-source", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--formal", action="store_true",
                    help="mark this independent held-out audit as formal trajectory data")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    prepared = Path(args.prepared, f"{args.task}_prepared.json")
    records = json.loads(prepared.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager",
        local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    env = {
        "status": "FORMAL_HELDOUT" if args.formal else "CALIBRATION_HELDOUT",
        "data_role": "formal_heldout" if args.formal else "calibration_heldout_only",
        "formal_admissible": bool(args.formal),
        "task": args.task,
        "model": MODEL,
        "adapter": args.adapter,
        "adapter_sha256": file_sha256(Path(args.adapter, "adapter_model.safetensors")),
        "samples_per_source": args.samples_per_source,
        "seed": args.seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "thinking": False,
        "versions": {k: importlib.metadata.version(k)
                     for k in ("torch", "transformers", "peft", "tokenizers")},
        "gpu": torch.cuda.get_device_name(0),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    counts = Counter()
    invalid = Counter()
    rewards = []
    semantic = []
    rows_path = out / "samples.jsonl"
    start_time = time.monotonic()
    with rows_path.open("w") as raw:
        for source, record in enumerate(records):
            label = json.loads(record["label"])
            rows = generate(model, tokenizer, record["input"], label,
                            args.samples_per_source, args.batch_size,
                            args.seed + source)
            for row in rows:
                result = score_response(row["text"], row["truncated"], label)
                row.update({"task": args.task, "source": source,
                            "target": result["target"], "reward": result["reward"],
                            "u": result["u"], "invalid_reason": result["invalid_reason"]})
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                rewards.append(float(result["reward"]))
                semantic.append(float(result["u"]))
                if result["target"] is None:
                    invalid[source] += 1
                else:
                    counts[(source, int(result["target"]))] += 1
    total = max(len(rewards), 1)
    metrics = {
        **env,
        "status": "FORMAL_HELDOUT_COMPLETE" if args.formal else "CALIBRATION_HELDOUT_COMPLETE",
        "sources": len(records),
        "total_samples": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "mean_semantic_reward": float(np.mean(semantic)) if semantic else 0.0,
        "invalid_rate": float(sum(invalid.values()) / total),
        "invalid_counts": dict(sorted(invalid.items())),
        "observed_directed_edges": len(counts),
        "possible_directed_edges": sum(len(json.loads(r["label"])["mapping"]) for r in records),
        "wall_seconds": time.monotonic() - start_time,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "formal_admissible": bool(args.formal),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
