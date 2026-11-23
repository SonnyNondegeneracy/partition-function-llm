"""Figure 4: sensitivity of post-training organisation and reward ablation, 183 x 153 mm.

Redrawn from the frozen source data of the paper figure (hfdata/.../frozen/figure3/observations.csv), with seed
variability added to the error bars of the main-protocol series: half-width 1.96*sqrt(s_boot^2 + s_seed^2), s_boot from the
95% bootstrap interval, s_seed the standard deviation over the three training seeds.
Rows: order parameter r, mean reward R, state Shannon entropy H (source means); x axis -log beta_nominal, rounds 1-8.

    python plotting/figure4.py          # writes figures/figure3_ablations.{pdf,png}
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, BLUE, ORANGE, INK, MM  # noqa: E402
import json  # noqa: E402
from beta_eff import FROZEN, REV, P02A, FIGURES, CODE, PREP, load, rewards, order_parameter  # noqa: E402
from m2_kernel import fit_action_potential  # noqa: E402

GREEN = "#009E73"
OUT = FIGURES; OUT.mkdir(parents=True, exist_ok=True)
SD = HERE / "source_data" / "figure4"; SD.mkdir(parents=True, exist_ok=True)
setup()
obs = list(csv.DictReader(open(FROZEN / "figure3" / "observations.csv")))
X = -np.log(0.375 / np.arange(1, 9))
COLS = [("Inference settings", [("direct_9B", r"Direct, $\tau=1$", BLUE, "o"), ("thinking_9B", r"Thinking, $\tau=1$", ORANGE, "s"),
                                ("direct_temperature_07", r"Direct, $\tau=0.7$", GREEN, "^")], "direct_9B"),
        ("Model size and family", [("direct_9B", "Qwen3.5-9B", BLUE, "o"), ("direct_4B", "Qwen3.5-4B", ORANGE, "s")], "direct_9B"),
        ("Prompt variants", [("direct_9B", "Original", BLUE, "o"), ("six_variant_mean", "Six-variant mean", ORANGE, "s")], "direct_9B"),
        ("Evaluation task", [("m2_t1_t1", "T1 eval, T1 train", BLUE, "o"), ("m2_t1_t2", "T2 eval, T1 train", ORANGE, "s")], "m2_t1_t1"),
        ("Reward structure", [("direct_9B", r"$\alpha=0$ (main)", BLUE, "o")] +
         [(f"E5_{a:g}", rf"$\alpha={a:g}$", c, "D") for a, c in zip((0.25, 0.5, 1, 2, 4), ("#B4A7D6", "#9A86C8", "#7B5EA7", "#5E3C8F", "#3F1F6E"))], "direct_9B")]
COLS[1][1].append(("gemma4", "Gemma-4-E4B", GREEN, "^"))
ROWS = [("r", "r_low", "r_high", "$r$"), ("reward", "reward_low", "reward_high", "$R$"), ("entropy", "entropy_low", "entropy_high", "$H$")]

# seeds of the main T1 protocol: r, R, H (source means) per round
u = rewards("T1_main.json", 8); seeds = {}; rows = []
for sd in ("seed2", "seed3"):
    vals = {"r": [], "reward": [], "entropy": []}
    for m in range(1, 9):
        f = P02A / f"E4_T1_{sd}" / f"r{m:02d}" / "counts.json"; N, inv, _ = load(f); att = N.sum(1) + inv
        P = N / N.sum(1, keepdims=True)
        vals["r"].append(order_parameter(f)); vals["reward"].append((((N * u).sum(1) - inv) / att).mean())
        vals["entropy"].append((-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(1)).mean())
        rows.append(dict(seed=sd, round=m, minus_log_beta_nominal=X[m - 1], r=vals["r"][-1], reward=vals["reward"][-1], entropy=vals["entropy"][-1]))
    seeds[sd] = vals
with open(SD / "seeds_T1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---- additional series (this revision): Gemma-4 (merged-weight measurement) and reward interpolation E5 -----------
RES = REV


def reward_matrix(prepared, n=8):
    """R_ij - 1 for valid destinations: u_j + alpha * eps_ij (eps from each source's prepared record)."""
    recs = [json.loads(r["label"]) for r in json.load(open(PREP / prepared))]
    u_ = np.array([float(recs[0]["state_rewards"][str(k)]) for k in range(n)]); M = np.tile(u_, (n, 1))
    for lab in recs:
        if "edge_epsilon" in lab:
            eps = json.loads(lab["edge_epsilon"].replace("'", '"')) if isinstance(lab["edge_epsilon"], str) else lab["edge_epsilon"]
            i = int(lab["source"])
            for j, e in eps.items():
                M[i, int(j)] += float(lab["alpha"]) * float(e)
    return M


def observe(N, inv, S, M):
    att = N.sum(1) + inv; P = N / np.maximum(N.sum(1, keepdims=True), 1)
    T = np.where(S, (N + 0.1) / (att + 0.1 * S.sum(1))[:, None], 0.0)
    r = fit_action_potential(T, support_mask=S, pair_mask=N > 0)["r_action"]
    R = (((N * M).sum(1) - inv) / att).mean(); H = (-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(1)).mean()
    return r, R, H


def series_with_bootstrap(files, M, nboot=500, seed=0):
    rng = np.random.default_rng(seed); out = []
    for f in files:
        N, inv, S = load(f); val = observe(N, inv, S, M); draws = []
        full = np.c_[N, inv]; n = full.sum(1)
        for _ in range(nboot):
            B = np.array([rng.multinomial(nn, row / nn) for row, nn in zip(full, n)])
            draws.append(observe(B[:, :-1], B[:, -1], S, M))
        lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
        out.append(dict(zip(("r", "reward", "entropy"), val)) | {f"{k}_low": l for k, l in zip(("r", "reward", "entropy"), lo)}
                   | {f"{k}_high": h for k, h in zip(("r", "reward", "entropy"), hi)})
    return out


CACHE = SD / "additional_series.json"
extra = json.load(open(CACHE)) if CACHE.exists() else {}
todo = {"gemma4": ([RES / "P03" / "E7_gemma4_T1_merged" / f"r{m:02d}" / "counts.json" for m in range(1, 9)], reward_matrix("gemma4_T1_direct.json"))}
for a in (0.25, 0.5, 1, 2, 4):
    todo[f"E5_{a:g}"] = ([RES / "P02" / f"E5_alpha{a:g}" / f"r{m:02d}" / "counts.json" for m in range(1, 9)], reward_matrix(f"T1_mix_a{a:g}.json"))
for k, (files, M) in todo.items():
    if k not in extra:
        extra[k] = series_with_bootstrap(files, M); json.dump(extra, open(CACHE, "w"), indent=1); print("bootstrapped", k)
for k, rows_ in extra.items():
    for m, row in enumerate(rows_, 1):
        obs.append({"series": k, "round": str(m), **{kk: str(v) for kk, v in row.items()}})

fig, axs = plt.subplots(3, 5, figsize=(183 * MM, 153 * MM))
fig.subplots_adjust(left=.07, right=.975, bottom=.07, top=.76, wspace=.42, hspace=.38)
for ci, (title, series, main) in enumerate(COLS):
    for ri, (key, lo, hi, ylab) in enumerate(ROWS):
        ax = axs[ri, ci]
        for name, label, col, mk in series:
            o = sorted([r for r in obs if r["series"] == name and r["round"] not in ("0", "")], key=lambda r: int(r["round"]))
            y = np.array([float(r[key]) for r in o])
            ax.plot(X, y, color=col, lw=.9, zorder=2)
            if o[0][lo]:
                lo_, hi_ = np.array([float(r[lo]) for r in o]), np.array([float(r[hi]) for r in o])
                if name in ("direct_9B", "m2_t1_t1"):          # main protocol: add retraining variability
                    sb = (hi_ - lo_) / 3.92; ss = np.std(np.vstack([y, seeds["seed2"][key], seeds["seed3"][key]]), axis=0, ddof=1)
                    hw = 1.96 * np.sqrt(sb ** 2 + ss ** 2); lo_, hi_ = y - hw, y + hw
                ax.vlines(X, lo_, hi_, color=col, lw=.6, zorder=2)
            ax.plot(X, y, ls="none", marker=mk, ms=3.2, mfc="white", mec=col, mew=.85, zorder=3)
        ax.set_xticks([1, 2, 3])
        if ci == 0:
            ax.set_ylabel(ylab)
        if ri == 2:
            ax.set_xlabel(r"$-\log\beta_{\mathrm{nominal}}$")
        ax.text(-.25, 1.06, "abcdefghijklmno"[ri * 5 + ci], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    h = [Line2D([], [], ls="none", marker=mk, ms=3.2, mfc="white", mec=col, mew=.85, label=label) for _, label, col, mk in series]
    axs[0, ci].legend(handles=h, loc="upper center", bbox_to_anchor=(.5, 2.05), frameon=False, fontsize=5.8, handlelength=1.6,
                      title=title, title_fontproperties={"weight": "bold", "size": 6.2})
fig.savefig(OUT / "figure3_ablations.pdf", facecolor="white")
fig.savefig(OUT / "figure3_ablations.png", dpi=300, facecolor="white")
print({sd: [round(v["r"][-1], 3), round(v["reward"][-1], 3), round(v["entropy"][-1], 3)] for sd, v in seeds.items()})
