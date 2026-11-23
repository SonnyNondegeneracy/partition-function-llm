"""Figure 2: potential structure strengthens during post-training, 183 x 72 mm.

a  Per-transition test of the partition-function description: predicted change T_ij - Q_ij of the ideal optimum
   T = Q exp(u/beta_eff,m)/Z (from the initial kernel Q and the calibrated beta_eff,m only) versus the measured change.
   Rounds 1, 4 and 8 are drawn; statistics use all 8 rounds.
b  Order parameter r versus -log beta_eff, T1 and T2 main experiments with fits. c of each task is fitted jointly on its
   three training seeds (recalibrate.py source data).
c  Centred measured potential in round 8 versus the theoretical limit Phi_i(0).
Error bars in b and c: 1.96*sqrt(s_boot^2 + s_seed^2), s_seed the standard deviation over the three training seeds.

    python plotting/figure2.py          # writes figures/figure2_potential_emergence.{pdf,png}
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
from beta_eff import (RECAL, MAIN, P02A, FIGURES, load, rewards, kern, tilt, order_parameter,  # noqa: E402
                      pooled_c)
from m2_kernel import fit_action_potential  # noqa: E402  (analysis/ on path via beta_eff)

GREY = "#888888"
OUT = FIGURES; OUT.mkdir(parents=True, exist_ok=True)
SD = HERE / "source_data" / "figure2"; SD.mkdir(parents=True, exist_ok=True)
TASKS = [("T1", 8, "T1_main.json", BLUE, "o"), ("T2", 12, "T2_main.json", ORANGE, "^")]
SHOW = (1, 4, 8)
B0 = 0.375
setup()
MK = dict(ls="none", ms=3.4, mfc="white", mew=.85)
STAT = dict(ha="right", va="bottom", fontsize=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 72 * MM), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
fig.subplots_adjust(left=.06, right=.985, bottom=.17, top=.83, wspace=.38)


def lab(a, l):
    a.text(-.2 if a.get_position().width < .3 else -.14, 1.06, l, transform=a.transAxes, fontsize=8,
           fontweight="bold", va="bottom")


# one calibration factor per task, fitted jointly on its three training seeds
SEEDC = {(task, sd): pooled_c(task) for task, *_ in TASKS for sd in ("seed2", "seed3")}


def transition_stats(Q, S, u, c, files, task=None):
    xs, ys = [], []
    for m, f in enumerate(files, 1):
        N, _, _ = load(f, task); P = tilt(Q, S, u, c * B0 / m); T = kern(N, S)
        xs.append((P - Q)[S]); ys.append((T - Q)[S])
    xs, ys = np.concatenate(xs), np.concatenate(ys); big = np.abs(ys) > .01
    return np.corrcoef(xs, ys)[0, 1], np.mean(np.sign(xs[big]) == np.sign(ys[big]))


# a: per-transition prediction
a = ax[0]; stats = {}; rows = []
for task, n, prep, col, mk in TASKS:
    u = rewards(prep, n); c = pooled_c(task)
    N0, _, S = load(MAIN / "base" / "4096.json", task); Q = kern(N0, S)
    xs, ys = [], []
    for m in range(1, 9):
        N, _, _ = load(MAIN / f"{task}_r{m:02d}" / "4096.json", task)
        P = tilt(Q, S, u, c * B0 / m); T = kern(N, S)
        x, y = (P - Q)[S], (T - Q)[S]; xs.append(x); ys.append(y)
        for (i, j), xv, yv in zip(np.argwhere(S), x, y):
            rows.append(dict(task=task, round=m, source=i, target=j, predicted=xv, measured=yv))
        if m in SHOW:
            a.plot(x, y, marker=mk, mec=col, zorder=3 if task == "T1" else 2, **MK)
    xs, ys = np.concatenate(xs), np.concatenate(ys); big = np.abs(ys) > .01
    stats[task] = dict(n=len(xs), pearson=np.corrcoef(xs, ys)[0, 1], same_sign=np.mean(np.sign(xs[big]) == np.sign(ys[big])))
    for sd in ("seed2", "seed3"):
        d = P02A / f"E4_{task}_{sd}"; Ns, _, Ss = load(d / "base" / "counts.json")
        stats[task][sd] = transition_stats(kern(Ns, Ss), Ss, u, SEEDC[(task, sd)], [d / f"r{m:02d}" / "counts.json" for m in range(1, 9)])
L = .8
a.fill_between([0, L], 0, L, color=BLUE, alpha=.06, lw=0); a.fill_between([-L, 0], -L, 0, color=BLUE, alpha=.06, lw=0)
a.axhline(0, color=GREY, lw=.5, zorder=1); a.axvline(0, color=GREY, lw=.5, zorder=1)
a.plot([-L, L], [-L, L], color=GREY, ls=(0, (3, 2)), lw=.8, zorder=1)
a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal")
def three(t, k):
    v = [stats[t]["pearson" if k == 0 else "same_sign"]] + [stats[t][sd][k] for sd in ("seed2", "seed3")]
    return "/".join(f"{x:.2f}" if k == 0 else f"{x:.0%}" for x in v)
a.text(.97, .03, f"Pearson: T1 {stats['T1']['pearson']:.2f}\n             T2 {stats['T2']['pearson']:.2f}", transform=a.transAxes, **STAT)
a.set_xlabel(r"Predicted $T_{ij}-Q_{ij}$ at $\beta_{\mathrm{eff},m}$"); a.set_ylabel(r"Measured $T_{ij}-Q_{ij}$"); lab(a, "a")
with open(SD / "panel_a_transitions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# b: r vs -log beta_eff
obs = list(csv.DictReader(open(RECAL / "figure2" / "r_observations.csv")))
cur = list(csv.DictReader(open(RECAL / "figure2" / "r_fitted_curves.csv")))
b = ax[1]; seed_rows = []
for task, n, prep, col, mk in TASKS:
    o = [r for r in obs if r["task"] == task]
    xx = -np.log([float(r["beta_effective"]) for r in o]); yy = np.array([float(r["r"]) for r in o])
    cu = [r for r in cur if r["task"] == task]
    b.plot(-np.log([float(r["beta_effective"]) for r in cu]), [float(r["fitted_r"]) for r in cu], color=col, lw=1)
    u = rewards(prep, n); seed_r = []
    for sd in ("seed2", "seed3"):
        d = P02A / f"E4_{task}_{sd}"; files = [d / f"r{m:02d}" / "counts.json" for m in range(1, 9)]
        c = SEEDC[(task, sd)]
        r = [order_parameter(f) for f in files]; x = -np.log(c * B0 / np.arange(1, 9))
        seed_r.append(r)
        seed_rows += [dict(task=task, seed=sd, c=c, round=m, minus_log_beta_eff=xv, r=rv) for m, xv, rv in zip(range(1, 9), x, r)]
    sb = (np.array([float(r["r_high"]) for r in o]) - np.array([float(r["r_low"]) for r in o])) / 3.92
    hw = 1.96 * np.hypot(sb, np.std(np.vstack([yy] + seed_r), axis=0, ddof=1))     # bootstrap + three training seeds
    b.vlines(xx, yy - hw, yy + hw, color=col, lw=.6)
    b.plot(xx, yy, marker=mk, mec=col, zorder=3, **MK)
b.set_xlabel(r"$-\log\beta_{\mathrm{eff}}$"); b.set_ylabel("Order parameter, $r$"); lab(b, "b")
with open(SD / "panel_b_seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seed_rows[0])); w.writeheader(); w.writerows(seed_rows)

# c: potential
pot = list(csv.DictReader(open(RECAL / "figure2" / "potential_comparison.csv")))
cc = ax[2]; allv = []; pear = {}
for task, n, prep, col, mk in TASKS:
    p = [r for r in pot if r["task"] == task]
    xx = np.array([float(r["predicted_limit"]) for r in p]); yy = np.array([float(r["scaled_phi"]) for r in p])
    pear[task] = [np.corrcoef(xx, yy)[0, 1]]; seed_phi = []
    for sd in ("seed2", "seed3"):
        f = P02A / f"E4_{task}_{sd}" / "r08" / "counts.json"; N, inv, S = load(f); att = N.sum(1) + inv
        T = np.where(S, (N + 0.1) / (att + 0.1 * S.sum(1))[:, None], 0.0)
        phi = np.array(fit_action_potential(T, support_mask=S, pair_mask=N > 0)["phi"]) * SEEDC[(task, sd)] * B0 / 8
        phi -= phi.mean(); seed_phi.append(phi)
        pear[task].append(np.corrcoef(xx, phi)[0, 1])
    sb = (np.array([float(r["scaled_phi_high"]) for r in p]) - np.array([float(r["scaled_phi_low"]) for r in p])) / 3.92
    hw = 1.96 * np.hypot(sb, np.std(np.vstack([yy] + seed_phi), axis=0, ddof=1))  # bootstrap + three training seeds
    cc.vlines(xx, yy - hw, yy + hw, color=col, lw=.6)
    cc.plot(xx, yy, marker=mk, mec=col, zorder=3, **MK); allv += list(xx) + list(yy + hw) + list(yy - hw)
cc.text(.97, .03, f"Pearson: T1 {pear['T1'][0]:.2f}\n             T2 {pear['T2'][0]:.2f}", transform=cc.transAxes, **STAT)
lim = [min(allv) - .05, max(allv) + .05]; cc.plot(lim, lim, color=GREY, ls=(0, (3, 2)), lw=.8, zorder=0)
cc.set_xlabel(r"Predicted limit, $\Phi_i(0)$ (centred)"); cc.set_ylabel(r"Measured $\beta_{\mathrm{eff}}\hat{\phi}_i$ (centred)"); lab(cc, "c")

fig.legend(handles=[Line2D([], [], marker="o", mec=BLUE, label="T1", **MK), Line2D([], [], marker="^", mec=ORANGE, label="T2", **MK),
                    Line2D([], [], color=INK, lw=1, label="Fit"),
                    Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=.8, label="Equality")],
           loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, handlelength=1.5, columnspacing=1.2)
fig.savefig(OUT / "figure2_potential_emergence.pdf", facecolor="white")
fig.savefig(OUT / "figure2_potential_emergence.png", dpi=300, facecolor="white")
for t, s in stats.items():
    print(f"Fig2a {t}: n={s['n']} pearson={s['pearson']:.3f} same_sign(|dT|>0.01)={s['same_sign']:.3f}")
print("Fig2c pearson", {k: [round(x, 3) for x in v] for k, v in pear.items()})
print("Fig2a seeds", {t: {sd: [round(x, 3) for x in stats[t][sd]] for sd in ("seed2", "seed3")} for t in stats})
print("seed c:", sorted({(r['task'], r['seed'], round(r['c'], 3)) for r in seed_rows}))
