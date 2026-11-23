# Third-party notices

The top-level MIT license covers this project's own code and documentation.

- third_party/openrlhf/: frozen OpenRLHF 0.10.4 with project changes for Qwen3.5 text-only loading, cumulative LoRA loading, effective-weight synchronization, resource settings and audit hooks. Apache-2.0; see OpenRLHF_LICENSE and APACHE-2.0.txt. Original copyright notices remain in source files. The exact patched source is included; openrlhf_patch_manifest.json records its historical provenance. Upstream: https://github.com/OpenRLHF/OpenRLHF/tree/v0.10.4 .
- analysis/action_precision.py and m2_kernel.py: this project's potential-reconstruction solver, used by the reported analysis.
- Qwen3.5-9B and Qwen3.5-4B: base weights are not redistributed. Their fixed revisions and Apache-2.0 notices are listed in the companion data archive. Adapter weights are project artifacts; base-model obligations still apply.
- Matplotlib, SciPy, NumPy, ReportLab, PyMuPDF, OpenPyXL, PyTorch, Transformers, PEFT, vLLM, Ray, DeepSpeed, FLA and causal-conv1d are external dependencies under their respective upstream licenses. Installing them does not change their licenses.
- Arial is not redistributed. If unavailable, the schematic builder uses installed DejaVu Sans; font substitution can alter layout. Set ENERGY_FONT_DIR to a licensed Arial font folder for the original schematic typography.
- The approved theory PDF is in the companion data archive under its project license. External papers and Nature guideline PDFs are not redistributed.

The human-readable original filesystem paths in frozen historical provenance are source identifiers, not executable dependencies. No credentials are needed for CPU replay.
