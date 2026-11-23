"""Own a local Ray cluster and stop only that cluster on exit."""
from __future__ import annotations

import os
import runpy
import sys


def main():
    import ray
    temporary = os.environ["TMPDIR"] + "/ray"
    os.makedirs(temporary, exist_ok=True)
    os.environ.pop("RAY_ADDRESS", None)
    num_cpus = int(os.environ.get("M2_RAY_CPUS", "16"))
    num_gpus = int(os.environ.get("M2_RAY_GPUS", "1"))
    if num_gpus < 1:
        raise ValueError("M2_RAY_GPUS must be positive")
    ray.init(address="local", _temp_dir=temporary, num_cpus=num_cpus, num_gpus=num_gpus,
             object_store_memory=2147483648, include_dashboard=False,
             runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false", "NCCL_DEBUG": "WARN",
                          "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                          # Ray workers are separate processes and do not
                          # inherit the driver's module search path reliably.
                          # Propagate the frozen OpenRLHF and FLA overlays so
                          # Qwen3.5's training-side imports match vLLM/formal M2.
                          "PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    try:
        runpy.run_module("openrlhf.cli.train_ppo_ray", run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
