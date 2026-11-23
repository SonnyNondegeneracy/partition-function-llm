"""Build causal-conv1d for the A100-only sm80 target in a Magnus overlay."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


URL = (
    "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/"
    "causal_conv1d-1.7.0.tar.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args()
    overlay = Path(args.overlay)
    overlay.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="causal_conv1d_sm80_") as temp:
        temp_path = Path(temp)
        archive = temp_path / "causal_conv1d-1.7.0.tar.gz"
        urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(temp_path)
        source = temp_path / "causal_conv1d-1.7.0"
        setup = source / "setup.py"
        text = setup.read_text()
        pattern = r"    cc_flag = \[\].*?    # HACK:"
        replacement = (
            '    cc_flag = ["-gencode", "arch=compute_80,code=sm_80"]\n\n'
            "    # HACK:"
        )
        patched, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("causal-conv1d setup.py architecture block was not found")
        setup.write_text(patched)
        subprocess.run(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(overlay),
                str(source),
            ],
            check=True,
        )
    print(f"built causal-conv1d 1.7.0 for sm80 into {overlay}", flush=True)


if __name__ == "__main__":
    main()
