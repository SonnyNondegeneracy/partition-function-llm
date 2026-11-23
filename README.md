# Potential emergence in LLM agents: a statistical physics perspective

Code for the paper by Zhuo-Yang Song and Hua Xing Zhu ([arXiv:2611.27618](https://arxiv.org/abs/2611.27618)): post-training of Qwen3.5 and Gemma-4 agents on two finite-state
tasks (T1, T2), measurement of state-transition counts, and the analysis and figures of the paper.

Data (transition counts, raw responses, adapters and source data):
[Nondegeneracy/partition-function-llm](https://huggingface.co/datasets/Nondegeneracy/partition-function-llm) on Hugging Face.

## Layout

| Folder | Content |
|---|---|
| `plotting/` | Figures 1-5 (`figureN.py`), the calibration `beta_eff.py` and `recalibrate.py`, source data written by the scripts |
| `analysis/` | Potential reconstruction (`m2_kernel.py`, `action_precision.py`), errors-in-variables calibration (`calibration_wtls.py`), raw-response replay |
| `training/` | Round-by-round KL-regularised post-training with cumulative LoRA adapters (main T1/T2, Qwen3.5-4B, KL scan) |
| `evaluation/` | Measurement of transition counts: main sampler, thinking mode and prompt variants, sampling temperature |
| `revision/` | Training seeds, training amount, reward interpolation and Gemma-4-E4B runs (`runs_p02.json`), with their sampler |
| `configs/`, `environments/` | Recorded training commands, base-model revisions, Python requirements |
| `third_party/` | Patched OpenRLHF 0.10.4 used for training, with its licence |

## Reproduce the figures (CPU, about 10 minutes)

Tested with Python 3.12 on Linux. The figures use Arial; put the Arial TTF files in `~/.fonts`, otherwise matplotlib
falls back to its default font.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r environments/requirements-cpu.txt
huggingface-cli download --repo-type dataset Nondegeneracy/partition-function-llm \
    --include "measurements/*" --local-dir ../hfdata

export PFL_DATA=../hfdata PFL_FIGURES=figures
python plotting/recalibrate.py
for n in 1 2 3 4 5; do python plotting/figure$n.py; done
```

`recalibrate.py` fits one calibration factor per task jointly on the three training seeds (c = 3.036 for T1 and 3.451
for T2) and recomputes the Fig. 2 and Fig. 3 source data. Each figure script reads the counts, writes its source data to
`plotting/source_data/figureN/` (see the README there) and saves the figure to `$PFL_FIGURES`.

| Paper | Script | Output file |
|---|---|---|
| Fig. 1 | `plotting/figure1.py` | `theory_experiment_overview.pdf` |
| Fig. 2 | `plotting/figure2.py` | `figure2_potential_emergence.pdf` |
| Fig. 3 | `plotting/figure3.py` | `figure1_reward_entropy.pdf` |
| Fig. 4 | `plotting/figure4.py` | `figure3_ablations.pdf` |
| Fig. 5 | `plotting/figure5.py` | `figure4_training_ablations.pdf` |

## Training and measurement (GPU)

The original runs used NVIDIA A100 80 GB GPUs. Base models: Qwen/Qwen3.5-9B, Qwen/Qwen3.5-4B and Gemma-4-E4B; the
Qwen revisions are fixed in `configs/base_model_fingerprints.json`.

```sh
pip install -r environments/requirements-gpu.txt -r environments/requirements-cpu.txt
python evaluation/download_base.py --size 9B --output /path/to/qwen9b
python training/prepare_runtime.py --family qwen35 --output /path/to/overlay
pip install --no-deps --target /path/to/fastpath fla-core==0.5.2 flash-linear-attention==0.5.2 'einops>=0.8,<0.9'
python training/build_causal_conv1d_sm80.py --overlay /path/to/fastpath
```

Main experiment, Qwen3.5-4B and KL scan (runs in `configs/training_runs.json`). Without `--execute` the command only
prints the eight round commands:

```sh
python training/run.py --data-root ../hfdata --run main_T1 --model /path/to/qwen9b \
    --output /path/to/run --overlay /path/to/overlay --fastpath /path/to/fastpath --execute
```

Each round loads the previous cumulative adapter as actor and frozen reference, runs four optimiser steps and saves a new
cumulative adapter. Measurements of the main experiment use `evaluation/main/measure_checkpoint_sampling.py` with 4,096
samples per source; thinking mode and prompt variants use `evaluation/supplementary/measure.py`, and the sampling
temperature uses `evaluation/extensions/measure_sample.py`. The sampling seed of every measurement is recorded in the
data archive (`metadata/`).

Revision runs (training seeds, training amount, reward interpolation, Gemma-4-E4B), trained and measured with fixed
seeds; models are expected in `$W/models/{qwen9b,qwen4b,gemma4}`:

```sh
python revision/p02_driver.py --work $W ref kl_scan/lambda_0p2_reinforce_baseline_full_qwen35_v2   # re-measure archived adapters
python revision/p02_driver.py --work $W train E4_T1_seed2                                          # train and measure a run
python revision/p02_driver.py --work $W status
```

Gemma-4-E4B rounds were measured from adapters merged into the base weights, because vLLM 0.22.1 does not apply
Gemma-4 LoRA weights.

## Citation

```bibtex
@misc{song2026potential,
  title         = {Potential emergence in {LLM} agents: a statistical physics perspective},
  author        = {Song, Zhuo-Yang and Zhu, Hua Xing},
  year          = {2026},
  eprint        = {2611.27618},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.27618},
  url           = {https://arxiv.org/abs/2611.27618}
}
```

## Licence

Code: MIT (`LICENSE`). Third-party code keeps its licence (`THIRD_PARTY_NOTICES.md`, `third_party/`).
