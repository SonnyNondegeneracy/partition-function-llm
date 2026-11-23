"""Copy proven libraries into isolated writable overlays; preserve originals."""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import shutil

from runtime_utils import atomic_json, digest, writable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--family", choices=["qwen35", "gemma4"], required=True)
    args = ap.parse_args()
    out = writable(args.output)
    out.mkdir(parents=True, exist_ok=False)
    original = Path(__file__).resolve().parents[1] / "third_party/openrlhf"
    target = out / "openrlhf"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    edits = {}
    deepspeed = Path(importlib.util.find_spec("deepspeed").origin).parent
    shutil.copytree(deepspeed, out / "deepspeed", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    nvtx = out / "deepspeed/utils/nvtx.py"
    text = nvtx.read_text()
    if "enable_nvtx = True" in text:
        text = text.replace("enable_nvtx = True", "enable_nvtx = False")
        nvtx.write_text(text)
    elif "enable_nvtx = False" not in text:
        raise RuntimeError("Unexpected DeepSpeed NVTX switch")
    peft = Path(importlib.util.find_spec("peft").origin).parent
    shutil.copytree(peft, out / "peft", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    availability = out / "peft/import_utils.py"
    lines = availability.read_text().splitlines(keepends=True)
    function = next(node for node in ast.parse("".join(lines)).body
                    if isinstance(node, ast.FunctionDef) and node.name == "is_torchao_available")
    lines[function.lineno-1:function.end_lineno] = [
        "def is_torchao_available():\n",
        "    # Match delivered M2: BF16 standard LoRA never uses TorchAO.\n",
        "    return False\n"]
    availability.write_text("".join(lines))
    edits["peft/import_utils.py"] = "disable unused TorchAO dispatcher; equivalent to original M2 uninstall, confined to this overlay"
    hashes = {str(p.relative_to(out)): digest(p) for p in out.rglob("*.py")}
    atomic_json(out / "manifest.json", {"source_openrlhf": str(original), "family": args.family,
                "edits": edits, "nvtx": "disabled, same as delivered M2", "sha256": hashes})
    print("ISOLATED_RUNTIME_READY", str(out), flush=True)


if __name__ == "__main__":
    main()
