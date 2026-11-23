"""Figure 3: reward and state-entropy consistency tests, 183 x 112 mm.

Redrawn from the frozen source data of the paper figure (hfdata/.../frozen/figure1: calibration points, fits, bands,
observations), with seed variability added to the error bars of
b-g: half-width 1.96*sqrt(s_boot^2 + s_seed^2), s_boot from the 95% bootstrap interval, s_seed the standard deviation
over the three training seeds (main and two further seeds) at the same round.
a  Calibration (nominal units): reward difference x inverse-beta difference vs Jeffreys divergence, one point per source
   state and round pair (main experiment), error bars combining bootstrap and seed spread; fits through the origin use the
   round pairs of all three training seeds.
b,e  T1 source t5; c,f  T1 source mean; d,g  T2 source mean. Top: mean reward (implementation reward minus 1, invalid -1);
bottom: state Shannon entropy over valid responses.

    python plotting/figure3.py          # writes figures/figure1_reward_entropy.{pdf,png}
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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, BLUE, ORANGE, INK, MM  # noqa: E402
from beta_eff import FROZEN, RECAL, P02A, FIGURES, load, rewards, coords, pooled_c  # noqa: E402

PURPLE = "#7B5EA7"
OUT = FIGURES; OUT.mkdir(parents=True, exist_ok=True)
SD = HERE / "source_data" / "figure3"; SD.mkdir(parents=True, exist_ok=True)
F1 = FROZEN / "figure1"
B0 = 0.375
setup()
MK = dict(ls="none", marker="o", ms=3.4, mfc="white", mew=.85)




obs = list(csv.DictReader(open(RECAL / "figure1" / "observations.csv")))   # fits redone at the three-seed c
cur = list(csv.DictReader(open(RECAL / "figure1" / "curves.csv")))
SERIES = [("T1_t5", "T1", 8, "T1_main.json", PURPLE, 5), ("T1_mean", "T1", 8, "T1_main.json", BLUE, None),
          ("T2_mean", "T2", 12, "T2_main.json", ORANGE, None)]

# seeds: per-round reward and entropy at their own calibrated beta_eff
seed_rows = []; seeds = {}
for task, n, prep in (("T1", 8, "T1_main.json"), ("T2", 12, "T2_main.json")):
    u = rewards(prep, n)
    for sd in ("seed2", "seed3"):
        d = P02A / f"E4_{task}_{sd}"; files = [d / f"r{m:02d}" / "counts.json" for m in range(1, 9)]
        c = pooled_c(task)
        R, H = [], []
        for f in files:
            N, inv, _ = load(f); att = N.sum(1) + inv
            R.append(((N * u).sum(1) - inv) / att)
            P = N / N.sum(1, keepdims=True); H.append(-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(1))
        seeds[(task, sd)] = dict(c=c, x=-np.log(c * B0 / np.arange(1, 9)), R=np.array(R), H=np.array(H))
        for m in range(8):
            seed_rows.append(dict(task=task, seed=sd, c=c, round=m + 1, minus_log_beta_eff=seeds[(task, sd)]["x"][m],
                                  reward_mean=R[m].mean(), entropy_mean=H[m].mean(), reward_t5=R[m][5] if task == "T1" else "",
                                  entropy_t5=H[m][5] if task == "T1" else ""))
with open(SD / "seeds_reward_entropy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seed_rows[0])); w.writeheader(); w.writerows(seed_rows)

fig = plt.figure(figsize=(183 * MM, 112 * MM))
G = fig.add_gridspec(2, 4, left=.075, right=.985, bottom=.1, top=.86, wspace=.42, hspace=.32, width_ratios=[1.25, 1, 1, 1])


def lab(a, l, title=None):
    a.text(-.2, 1.04, l, transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")
    if title:
        a.set_title(title, fontsize=7, pad=6)


# a: calibration
a = fig.add_subplot(G[:, 0])
for task, col, mk in (("T1", BLUE, "o"), ("T2", ORANGE, "^")):
    # one point per source state and round pair (28 per source): main-experiment value; error bars combine the
    # bootstrap interval with the spread over the three training seeds, as in panels b-g
    P = list(csv.DictReader(open(F1 / f"calibration_points_{task}.csv")))
    x = np.array([float(r["symmetric_KL"]) for r in P]); y = np.array([float(r["scaled_reward_difference"]) for r in P])
    n_ = 8 if task == "T1" else 12; u = rewards(f"{task}_main.json", n_); sxs, sys_ = [x], [y]
    for sd in ("seed2", "seed3"):
        d = P02A / f"E4_{task}_{sd}"; N = []
        for m in range(1, 9):
            Nm, _, S = load(d / f"r{m:02d}" / "counts.json"); N.append(Nm)
        sx, sy = coords(np.asarray(N), S, u, np.arange(1, 9) / B0); sxs.append(sx.ravel()); sys_.append(sy.ravel())
    sbx = (np.array([float(r["x_high"]) for r in P]) - np.array([float(r["x_low"]) for r in P])) / 3.92
    sby = (np.array([float(r["y_high"]) for r in P]) - np.array([float(r["y_low"]) for r in P])) / 3.92
    hx = 1.96 * np.hypot(sbx, np.std(sxs, axis=0, ddof=1)); hy = 1.96 * np.hypot(sby, np.std(sys_, axis=0, ddof=1))
    a.errorbar(x, y, xerr=hx, yerr=hy, ls="none", marker=mk, ms=2.2, mfc="white", mec=col, mew=.5, ecolor=col,
               elinewidth=.35, alpha=.7)
    c = pooled_c(task); xx = np.array([0, x.max() * 1.02]); a.plot(xx, c * xx, color=col, lw=1, zorder=5)
    a.plot([], [], color=col, lw=1, label=f"{task}: $c={c:.2f}$")
a.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.2)
a.set_xlabel(r"Jeffreys divergence, $J_i^{mn}$")
a.set_ylabel(r"$(\beta_{\mathrm{nominal},n}^{-1}-\beta_{\mathrm{nominal},m}^{-1})(R^L_{in}-R^L_{im})$"); lab(a, "a", "Calibration")

# b-g
ROWAX = [[], []]
for k, (series, task, n, prep, col, src) in enumerate(SERIES):
    o = [r for r in obs if r["series"] == series]; cu = [r for r in cur if r["series"] == series]
    cx = -np.log([float(r["beta_effective"]) for r in cu]); ox = -np.log([float(r["beta_effective"]) for r in o])
    for row, (key, fit, lo, hi, ylab) in enumerate((("reward_shifted", "reward_fit", "reward_band_low", "reward_band_high", "Mean reward"),
                                                    ("H_nats", "H_fit", "H_band_low", "H_band_high", "State Shannon entropy"))):
        ax = fig.add_subplot(G[row, k + 1], sharey=ROWAX[row][0] if ROWAX[row] else None); ROWAX[row].append(ax)
        if k > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.fill_between(cx, [float(r[lo]) for r in cu], [float(r[hi]) for r in cu], color=col, alpha=.13, lw=0)
        ax.plot(cx, [float(r[fit]) for r in cu], color=col, lw=1.1, zorder=2)
        lk = "reward" if row == 0 else "H"; y = np.array([float(r[key]) for r in o])
        sv = [seeds[(task, sd)]["R" if row == 0 else "H"] for sd in ("seed2", "seed3")]
        sv = [v[:, src] if src is not None else v.mean(1) for v in sv]
        sb = (np.array([float(r[f"{lk}_high"]) for r in o]) - np.array([float(r[f"{lk}_low"]) for r in o])) / 3.92
        hw = 1.96 * np.sqrt(sb ** 2 + np.std(np.vstack([y] + sv), axis=0, ddof=1) ** 2)
        ax.vlines(ox, y - hw, y + hw, color=col, lw=.6, zorder=3)
        ax.plot(ox, [float(r[key]) for r in o], mec=col, zorder=4, **MK)
        if k == 0:
            ax.set_ylabel(ylab)
        if row == 1:
            ax.set_xlabel(r"$-\log\beta_{\mathrm{eff}}$")
        title = {"T1_t5": "T1 (source t5)", "T1_mean": "T1 (source mean)", "T2_mean": "T2 (source mean)"}[series] if row == 0 else None
        lab(ax, "bcdefg"[k + 3 * row], title)

fig.legend(handles=[Line2D([], [], mec=INK, label="Measured", **MK),
                    Line2D([], [], color=INK, lw=1.1, label=r"Theory: $\beta_{\mathrm{eff}}=c\,\beta_{\mathrm{nominal}}$"),
                    Patch(facecolor=INK, alpha=.13, lw=0, label="Theoretical remainder band")],
           loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4)
fig.savefig(OUT / "figure1_reward_entropy.pdf", facecolor="white")
fig.savefig(OUT / "figure1_reward_entropy.png", dpi=300, facecolor="white")
print({k: round(v["c"], 3) for k, v in seeds.items()})
