"""Path boundary and provenance for isolated M2 supplements."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('ENERGY_RUN_ROOT', './evaluation_output')).resolve()
ALLOWED = ROOT.resolve()
HISTORY = ROOT.parent / "formal_v1"


def writable(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ALLOWED) or path.is_relative_to(HISTORY.resolve()):
        raise ValueError(f"Output outside supplementary write boundary: {path}")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def policies():
    d = json.loads((HISTORY / "formal_measurement_manifest.json").read_text())
    return d["model"], {
        key: value for key, value in d["policies"].items()
        if key == "base" or key.startswith("T1_")
    }
