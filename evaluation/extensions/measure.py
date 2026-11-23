"""Compatibility entry point for the v2 request generator.

Older request records call ``measure.py`` and pass a separate policy JSON via
``--policies``.  The implementation lives in :mod:`measure_sample`; this
wrapper only normalizes that request schema and refuses non-direct/non-T1 jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import digest
from measure_sample import _sample
from protocol import check_parser


def normalized(registry_path: str, policies_path: str | None, task: str) -> dict:
    source = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if task != "T1":
        raise ValueError("v2 compatibility measurement supports T1 only")
    if isinstance(source.get("task"), str):
        prepared = Path(source["task"])
        prepared_sha256 = source.get("task_sha256", digest(prepared))
    elif isinstance(source.get("task"), dict):
        prepared = Path(source["task"]["prepared"])
        prepared_sha256 = source["task"]["prepared_sha256"]
    else:
        try:
            item = source["tasks"][task]["prepared"]["direct"]
            prepared = Path(item["path"])
            prepared_sha256 = item["sha256"]
        except (KeyError, TypeError) as exc:
            raise ValueError("registry has no direct T1 prepared input") from exc
    policy_source = source if policies_path is None else json.loads(Path(policies_path).read_text(encoding="utf-8"))
    model = policy_source.get("model", source.get("model"))
    if not model:
        raise ValueError("registry lacks model path")
    policies = policy_source.get("policies", {})
    return {
        "schema_version": "m2.extension_v2.t1_registry.compat.v1",
        "model": model,
        "model_config_sha256": source.get("model_config_sha256", digest(Path(model) / "config.json")),
        "origin_manifest": str(Path(registry_path).resolve()),
        "origin_manifest_sha256": digest(registry_path),
        "task": {"name": "T1", "prepared": str(prepared.resolve()),
                 "prepared_sha256": prepared_sha256},
        "policies": policies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--policies", help="separate policy registry JSON")
    parser.add_argument("--policy", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", default=["T1"])
    parser.add_argument("--mode", choices=["direct"], default="direct")
    parser.add_argument("--samples", "--samples-per-source", type=int, default=4096,
                        dest="samples_per_source")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=82020000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.tasks != ["T1"]:
        raise ValueError("compatibility entry point only supports --tasks T1")
    check_parser()
    registry = normalized(args.registry, args.policies, "T1")
    unknown = [name for name in args.policy if name not in registry["policies"]]
    if unknown:
        raise ValueError(f"unknown T1 policies: {unknown}")
    return _sample(args, registry, args.policy)


if __name__ == "__main__":
    main()
