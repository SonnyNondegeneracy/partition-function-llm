"""Figure 5: training ablations -- training settings enter only through beta_eff, 183 x 60 mm.

Branches: main experiment (lambda = 0.375, lr 1e-5, 4 steps per round; paper counts and calibration), seven further
KL coefficients (lambda = 0.2 ... 5, all 8 rounds re-measured) and eight training-amount branches at lambda = 0.375 and 3
(lr x steps per round = 2e-5 x 4, 1e-5 x 8, 5e-6 x 8 for 4 rounds; 1e-5 x 32 for 2 rounds). Every branch is calibrated
on all of its measured rounds, beta_eff,m = c * lambda / m.

a  Calibration relation for every branch with at least 3 rounds, in beta_eff units, averaged over source states.
b  Order parameter of every branch at its own beta_eff on the main-experiment curve r(beta_eff) of Fig. 2 (no refit).
c  Calibration factor c versus the KL coefficient; c = 1 is the ideal optimum in every round.
The training-amount branches (E3) are calibrated and written to branches.csv but not drawn.
Two further training seeds of the main protocol enter only the error bars of the main experiment in b and c.

    python plotting/figure5.py          # writes figures/figure4_training_ablations.{pdf,png}
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, BLUE, INK, MM  # noqa: E402
from beta_eff import (RECAL, MAIN, P02A, FIGURES, load, rewards, coords, calibrate, calibrate_pooled, order_parameter,  # noqa: E402
                      frozen_c, pooled_c, rounds, trajectories)

GREY = "#888888"
OUT = FIGURES; OUT.mkdir(parents=True, exist_ok=True)
SD = HERE / "source_data" / "figure5"; SD.mkdir(parents=True, exist_ok=True)
setup()
MK = dict(ls="none", marker="o", ms=3.4, mfc="white", mec=BLUE, mew=.85)
REF = dict(color=GREY, lw=.7, ls=(0, (3, 2)))
u = rewards("T1_main.json", 8)

# ---- branches -------------------------------------------------------------------------------------------------
KL = {0.2: "lambda_0p2_reinforce_baseline_full_qwen35_v2", 0.3: "lambda_0p3_reinforce_baseline_full_qwen35_v2",
      0.5: "lambda_0p5_reinforce_baseline_full_qwen35_v2", 0.7: "lambda_0p7_reinforce_baseline_full_qwen35_v2",
      1.5: "lambda_1p5_reinforce_baseline_full_qwen35_v2_1gpu", 3.0: "lambda_3_reinforce_baseline_full_qwen35_v2",
      5.0: "lambda_5_reinforce_baseline_full_qwen35_v2"}
E3 = {"lr2x4": (2e-5, 4), "lr1x8": (1e-5, 8), "lr05x8": (5e-6, 8), "lr1x32": (1e-5, 32)}
branches = [dict(name="main", lam=0.375, lr=1e-5, steps=4, ms=list(range(1, 9)), task="T1",
                 files=[MAIN / f"T1_r{m:02d}" / "4096.json" for m in range(1, 9)], c=pooled_c("T1"))]
for sd in ("seed2", "seed3"):                           # two further training seeds of the main protocol
    dd = P02A / f"E4_T1_{sd}"
    branches.append(dict(name=f"main_{sd}", seed=True, lam=0.375, lr=1e-5, steps=4, ms=rounds(dd), task=None,
                         files=[dd / f"r{m:02d}" / "counts.json" for m in rounds(dd)]))
for lam, d in KL.items():
    dd = P02A / "kl_scan" / d
    branches.append(dict(name=f"kl_{lam:g}", lam=lam, lr=1e-5, steps=4, ms=rounds(dd), task=None,
                         files=[dd / f"r{m:02d}" / "counts.json" for m in rounds(dd)]))
for tag, lam in (("l0375", 0.375), ("l30", 3.0)):
    for k, (lr, st) in E3.items():
        dd = P02A / f"E3_{tag}_{k}"
        branches.append(dict(name=f"E3_{tag}_{k}", lam=lam, lr=lr, steps=st, ms=rounds(dd), task=None,
                             files=[dd / f"r{m:02d}" / "counts.json" for m in rounds(dd)]))
SHOWN = lambda b: not b.get("seed") and not b["name"].startswith("E3_")   # figure: main experiment and KL scan
for b in branches:
    ms = np.array(b["ms"])
    if "c" not in b:
        b["c"] = calibrate(b["files"], ms / b["lam"], u, b["task"])
    b["speed"] = 1 / (b["c"] * b["lam"])
    b["binv"] = ms * b["speed"]                         # 1/beta_eff,m
    b["r"] = [order_parameter(f, b["task"]) for f in b["files"]]
with open(SD / "branches.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["branch", "lambda", "lr", "steps_per_round", "rounds", "c", "per_round_change_inv_beta_eff"])
    for b in branches:
        w.writerow([b["name"], b["lam"], b["lr"], b["steps"], len(b["ms"]), f"{b['c']:.4f}", f"{b['speed']:.4f}"])

fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 60 * MM)); fig.subplots_adjust(left=.07, right=.985, bottom=.2, top=.9, wspace=.42)


def lab(a, l):
    a.text(-.25, 1.04, l, transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")


# a: calibration relation, every branch, beta_eff units, mean over source states
a = ax[0]; rows = []; nb = 0
for b in branches:
    if len(b["ms"]) < 3 or not SHOWN(b):
        continue
    N = np.array([load(f, b["task"])[0] for f in b["files"]]); S = load(b["files"][0], b["task"])[2]
    x, y = coords(N, S, u, b["binv"]); x, y = x.mean(1), y.mean(1); nb += 1
    a.plot(x, y, **{**MK, "ms": 2.2, "mew": .55})
    rows += [dict(branch=b["name"], jeffreys=xv, inv_beta_eff_diff_times_reward_diff=yv) for xv, yv in zip(x, y)]
a.plot([3e-3, 2], [3e-3, 2], **REF, zorder=0); a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(3e-3, 2); a.set_ylim(3e-3, 2); a.set_aspect("equal")
a.text(.05, .97, f"{nb} branches\n{len(rows)} round pairs", transform=a.transAxes, va="top", fontsize=5.5)
a.set_xlabel(r"Jeffreys divergence, $J_i^{mn}$"); a.set_ylabel(r"$\Delta(1/\beta_{\mathrm{eff}})\,\Delta R$"); lab(a, "a")
with open(SD / "panel_a_pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# b: every branch on the main-experiment r(beta_eff) curve
cur = [r for r in csv.DictReader(open(RECAL / "figure2" / "r_fitted_curves.csv")) if r["task"] == "T1"]
obs = [r for r in csv.DictReader(open(RECAL / "figure2" / "r_observations.csv")) if r["task"] == "T1"]
CX = np.array([-np.log(float(r["beta_effective"])) for r in cur]); CY = np.array([float(r["fitted_r"]) for r in cur])
o = np.argsort(CX); CX, CY = CX[o], CY[o]
c = ax[1]; c.plot(CX, CY, color=INK, lw=.9, zorder=1)
ox = -np.log([float(r["beta_effective"]) for r in obs]); oy = np.array([float(r["r"]) for r in obs])
sb = (np.array([float(r["r_high"]) for r in obs]) - np.array([float(r["r_low"]) for r in obs])) / 3.92
ss = np.std(np.vstack([oy] + [b["r"] for b in branches if b.get("seed")]), axis=0, ddof=1)
c.vlines(ox, oy - 1.96 * np.hypot(sb, ss), oy + 1.96 * np.hypot(sb, ss), color=INK, lw=.6, zorder=4)
c.plot(ox, oy, ls="none", marker="s", ms=3.4, mfc="white", mec=INK, mew=.85, zorder=5)
rows = []; res = []; nr = 0; others = [b for b in branches if b["name"] != "main" and SHOWN(b)]
for b in others:
    x = np.log(b["binv"]); c.plot(x, b["r"], **{**MK, "ms": 3}, zorder=3); nr += len(x)
    for m, xv, rv in zip(b["ms"], x, b["r"]):
        dv = rv - np.interp(xv, CX, CY) if CX[0] <= xv <= CX[-1] else np.nan
        rows.append(dict(branch=b["name"], round=m, minus_log_beta_eff=xv, r=rv, deviation_from_main_curve=dv)); res.append(dv)
c.text(.04, .97, f"Main experiment (squares, line)\n{len(others)} other branches, {nr} rounds", transform=c.transAxes, va="top", fontsize=5.5)
c.set_xlabel(r"$-\log\beta_{\mathrm{eff}}$"); c.set_ylabel("Order parameter, $r$"); lab(c, "b")
with open(SD / "panel_b_r.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# c: calibration factor versus KL coefficient (default lr x steps per round)
cx = ax[2]
tr = trajectories("T1"); seed_c = [frozen_c("T1")] + [calibrate_pooled([t], u) for t in tr[1:]]   # three seeds, each alone
for b in branches:
    if not SHOWN(b):
        continue
    main = b["name"] == "main"
    if main:
        cx.vlines(b["lam"], b["c"] - 1.96 * np.std(seed_c, ddof=1), b["c"] + 1.96 * np.std(seed_c, ddof=1), color=INK, lw=.6, zorder=4)
    cx.plot(b["lam"], b["c"], ls="none", marker="s" if main else "o", ms=3.4, mfc="white", mec=INK if main else BLUE,
            mew=.85, zorder=5 if main else 3)
with open(SD / "panel_c_calibration.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["branch", "kl_coefficient", "c", "c_seed_halfwidth"])
    for b in branches:
        if SHOWN(b):
            w.writerow([b["name"], b["lam"], f"{b['c']:.4f}", f"{1.96 * np.std(seed_c, ddof=1):.4f}" if b["name"] == "main" else ""])
cx.axhline(1, color=INK, lw=.9, zorder=0)
cx.set_xscale("log"); cx.set_yscale("log"); cx.set_xlim(.15, 6)
cx.set_xticks([.2, .5, 1, 2, 5]); cx.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
cx.set_yticks([.5, 1, 2, 5]); cx.set_yticklabels(["0.5", "1", "2", "5"]); cx.minorticks_off()
cx.set_xlabel(r"KL coefficient $\beta_0$"); cx.set_ylabel("Calibration factor, $c$"); lab(cx, "c")

fig.savefig(OUT / "figure4_training_ablations.pdf", facecolor="white")
fig.savefig(OUT / "figure4_training_ablations.png", dpi=300, facecolor="white")
res = np.array(res); res = res[~np.isnan(res)]
print(f"panel b: {len(res)} rounds inside curve range, median |dev| {np.median(np.abs(res)):.3f}, max {np.abs(res).max():.3f}")
for b in branches:
    print(f"{b['name']:16s} lam={b['lam']:<5g} lr={b['lr']:.0e} steps={b['steps']:<2d} rounds={len(b['ms'])} c={b['c']:.3f} speed={b['speed']:.3f}")
