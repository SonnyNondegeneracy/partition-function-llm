"""Load a cumulative PEFT adapter into an already constructed OpenRLHF model.

OpenRLHF keeps the reference model fixed for one train invocation. The M2
round driver starts a fresh invocation at each round; this helper makes the
previous cumulative adapter the actor and reference state at that boundary.
It uses PEFT's public state-dict loader and never merges the 9B base weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from peft import set_peft_model_state_dict
from safetensors.torch import load_file


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cumulative_adapter(model, adapter_dir: str | Path) -> dict:
    """Load and validate an HF PEFT adapter into ``model``."""
    root = Path(adapter_dir)
    cfg_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    if not cfg_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete cumulative adapter: {root}")
    cfg = json.loads(cfg_path.read_text())
    peft_cfg = getattr(model, "peft_config", {}).get("default")
    if peft_cfg is None:
        raise TypeError("target model has no default PEFT adapter")
    for key, expected in (("r", peft_cfg.r), ("lora_alpha", peft_cfg.lora_alpha)):
        if int(cfg[key]) != int(expected):
            raise ValueError(f"adapter {key}={cfg[key]} does not match model {expected}")
    expected_targets = sorted(str(x) for x in (peft_cfg.target_modules or []))
    actual_targets = sorted(str(x) for x in (cfg.get("target_modules") or []))
    if expected_targets != actual_targets:
        raise ValueError(f"adapter target_modules mismatch: {actual_targets} != {expected_targets}")

    state = load_file(str(weights_path), device="cpu")
    if not state or not any(v.abs().sum().item() > 0 for v in state.values()):
        raise ValueError("cumulative adapter contains no nonzero tensors")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    # PEFT reports every frozen base-model tensor as missing because an HF
    # adapter intentionally stores only LoRA tensors.  Only adapter keys are
    # required here; any unexpected key still indicates a wrong payload.
    missing_adapter = [k for k in missing if "lora_" in k or "lora" in k]
    if missing_adapter or unexpected:
        raise ValueError(f"cumulative adapter state mismatch: missing={missing_adapter[:4]} unexpected={unexpected[:4]}")
    report = {
        "adapter_dir": str(root),
        "adapter_sha256": _sha256(weights_path),
        "tensor_count": len(state),
        "nonzero_tensor_count": sum(v.abs().sum().item() > 0 for v in state.values()),
        "r": int(cfg["r"]),
        "lora_alpha": int(cfg["lora_alpha"]),
    }
    print("M2_CUMULATIVE_ADAPTER_LOADED " + json.dumps(report, sort_keys=True), flush=True)
    return report
