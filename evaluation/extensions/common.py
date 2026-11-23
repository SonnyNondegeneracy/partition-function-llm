"""Shared paths and immutable-input helpers for the M2 v2 supplement."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('ENERGY_RUN_ROOT', './evaluation_output')).resolve()
M2_ROOT = ROOT.parent
FORMAL = M2_ROOT / "formal_v1"
MODEL_DEFAULT = os.environ.get('ENERGY_MODEL', 'Qwen/Qwen3.5-9B')


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def writable(path: str | Path) -> Path:
    """Allow writes only below extensions_v2; formal data remain read-only."""
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(ROOT.resolve()):
        raise ValueError(f"v2 output is outside extensions_v2: {candidate}")
    return candidate


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_contract(actual: dict[str, str | None]):
    """Optionally enforce the delivered Magnus package versions."""
    source = os.environ.get("M2_RUNTIME_INSPECTION")
    if not source:
        return None
    expected = read_json(source).get("versions")
    if isinstance(expected, dict):
        # The frozen inspection may cover a superset of a tool's imports (for
        # example training records peft/deepspeed/ray, while measurement only
        # imports torch/transformers/vllm/tokenizers). Enforce the declared
        # intersection and leave undeclared package versions as observations.
        overlap = {name: actual[name] for name in actual if name in expected}
        expected_overlap = {name: expected[name] for name in overlap}
        if overlap != expected_overlap:
            raise RuntimeError(f"runtime version contract mismatch: actual={actual}, expected={expected}")
    return {"source": str(Path(source).resolve()), "versions": expected}
