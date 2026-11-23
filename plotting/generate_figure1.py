"""Figure 1: a representative source and the T1/T2 source averages.

Run from any directory: python llm_rlhf_paper/manuscript/generate_figure1.py
The data reproduce the website. One beta scale is calibrated per task from
all source/round pairs using symmetric KL, before fitting source amplitudes.
Task means retain four fitted coefficients conditional on this shared scale.
No checkpoints, training runs, or website files are changed.
"""
from pathlib import Path
from itertools import combinations
import csv
import hashlib
import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, MaxNLocator
import numpy as np
from calibration_wtls import (SharedSlopeIntrinsicScatter, covariance_blocks,
                              fit_with_bootstrap, subset_covariance)

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "llm_rlhf_paper"
EVIDENCE = ROOT / "llm_rlhf_web/assets/evidence.js"
APP = ROOT / "llm_rlhf_web/app.js"
M2 = ROOT / "llm_rlhf_m2"
OUT = PAPER / "figures"
SOURCE = OUT / "source_data/figure1"
BETA0 = 0.375
ROUNDS = np.arange(1, 9)
BETAS = BETA0 / ROUNDS
BOOTSTRAPS = 500
TAU = 0.30


def load_data():
    text = EVIDENCE.read_text()
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1)


def task_data(data, task):
    records = [next(r for r in data["records"]
                    if r["policy"] == f"{task}_r{m:02d}"
                    and r["eval_task"] == task and r["method"] == "sample"
                    and r.get("budget") == 4096) for m in ROUNDS]
    kernels = np.asarray([r["T"] for r in records], dtype=float)
    utility = np.asarray(records[0]["u"])
    mass = kernels.sum(axis=-1)
    H = entropy(kernels / mass[..., None])
    # Exactly the website convention: full-normalization E[training reward]-1.
    # The invalid sink has raw reward 0, hence shifted reward -1.
    reward = (kernels * (1 + utility)).sum(axis=-1) - 1
    np.testing.assert_allclose(H.mean(axis=1), [r["H_sem"] for r in records], atol=1e-12)
    np.testing.assert_allclose(reward.mean(axis=1), [r["expected_reward"] - 1 for r in records], atol=1e-12)
    np.testing.assert_allclose(BETAS, [r["beta_est_nominal"] for r in records])
    bootstrap_R, bootstrap_H, bootstrap_counts, observed_counts, audits = [], [], [], [], []
    for record, T in zip(records, kernels):
        n = int(record["budget"])
        raw_path = M2 / f"data/measurements/samples/{record['policy']}/{n}.json"
        metric = json.loads(raw_path.read_text())["metrics"][task]
        counts = np.zeros_like(T, dtype=int)
        for (i, j), count in zip(metric["edges"], metric["counts"]):
            counts[i, j] = count
        attempts = np.asarray(metric["attempts"], dtype=int)
        invalid = np.asarray(metric["invalid_counts"], dtype=int)
        assert np.all(attempts == n)
        np.testing.assert_array_equal(counts.sum(axis=1) + invalid, attempts)
        np.testing.assert_allclose(counts / n, T, atol=1e-12)
        assert not counts[~np.asarray(data["tasks"][task]["support"], dtype=bool)].any()
        rng = np.random.default_rng(int(record["bootstrap_seed"]))
        rr, hh, cc = [], [], []
        for row, bad_count in zip(counts, invalid):
            full_counts = np.r_[row, bad_count]
            assert full_counts.min() >= 0 and full_counts.sum() == n
            full_sample = rng.multinomial(n, full_counts / n, size=BOOTSTRAPS)
            assert np.all(full_sample.sum(axis=1) == n)
            sample = full_sample[:, :-1]
            cc.append(sample)
            legal_total = sample.sum(axis=1)
            assert np.all(legal_total > 0)
            rr.append((sample / n * (1 + utility)).sum(axis=1) - 1)
            hh.append(entropy(sample / legal_total[:, None]))
        bootstrap_R.append(np.asarray(rr).T)
        bootstrap_H.append(np.asarray(hh).T)
        bootstrap_counts.append(np.stack(cc, axis=1))
        observed_counts.append(counts)
        # Independent check: the mean-reward sampling variance is analytic.
        full_prob = np.c_[T, invalid / n]
        full_reward = np.r_[utility, -1.]
        row_mean = full_prob @ full_reward
        row_variance = (full_prob @ full_reward**2 - row_mean**2) / n
        analytic_se = float(np.sqrt(row_variance.sum()) / len(T))
        boot_se = float(np.std(np.mean(rr, axis=0), ddof=1))
        assert .85 < boot_se / analytic_se < 1.15
        audits.append(dict(policy=record["policy"], raw_file=str(raw_path.relative_to(ROOT)),
                           raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                           counts_match_website=True, all_bootstrap_totals_preserved=True,
                           mean_reward_analytic_se=analytic_se, mean_reward_bootstrap_se=boot_se))
    bootstrap_R, bootstrap_H = np.asarray(bootstrap_R), np.asarray(bootstrap_H)
    ci_H = np.quantile(bootstrap_H.mean(axis=2), [.025, .975], axis=1)
    frozen_ci = np.asarray([[r["H_sem_low"], r["H_sem_high"]] for r in records]).T
    np.testing.assert_allclose(ci_H, frozen_ci, atol=1e-12)
    return dict(records=records, R=reward, H=H, mass=mass,
                bootstrap_R=bootstrap_R, bootstrap_H=bootstrap_H, audits=audits,
                counts=np.asarray(observed_counts), bootstrap_counts=np.asarray(bootstrap_counts),
                support=np.asarray(data["tasks"][task]["support"], dtype=bool), utility=utility)


def fixed_source_parameters(data, task, index):
    task_path = M2 / f"config/tasks/{task}_direct_formal_v1.json"
    task_doc = json.loads(task_path.read_text())
    utility = np.asarray([state["reward"] for state in task_doc["states"]])
    np.testing.assert_array_equal(utility, [state["reward"] for state in data["tasks"][task]["states"]])
    base_path = M2 / f"data/measurements/kernels/base/{task}.npz"
    with np.load(base_path, allow_pickle=False) as base:
        support = base["registered_support"].astype(bool)
        np.testing.assert_array_equal(support, data["tasks"][task]["support"])
        np.testing.assert_array_equal(base["state_reward"], utility)
        Q = base["Q_base"].copy()
    reward_max = float(utility[support[index]].max())
    optimal = support[index] & np.isclose(utility, reward_max, rtol=0, atol=1e-12)
    suboptimal = support[index] & ~optimal
    gap = float(reward_max - utility[suboptimal].max())
    if optimal.sum() == 1:
        Hstar = 0.0  # Exact: the optimal set is a singleton, independent of Q.
    else:
        qstar = Q[index, optimal]
        if np.any(qstar <= 0):
            raise ValueError("Fixed reference does not cover the optimal set.")
        Hstar = float(entropy(qstar / qstar.sum()))
    return dict(Rinf=reward_max, raw_Rmax=reward_max + 1, delta=gap, Hstar=Hstar,
                optimal_targets=[task_doc["states"][j]["id"] for j in np.flatnonzero(optimal)],
                fixed_parameters=["Rmax", "delta", "Hstar"], free_parameters=["A"],
                task_definition=str(task_path.relative_to(ROOT)),
                task_definition_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
                reference_kernel=str(base_path.relative_to(ROOT)),
                reference_kernel_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                method="single_source_fixed_limits_fit_A_only")


def fit_amplitude(reward, H, fixed):
    beta = fixed.get("beta_scale", 1.) * BETAS
    tail = np.exp(-fixed["delta"] / beta)
    x = np.r_[-tail, tail / beta]
    y = np.r_[reward - fixed["Rinf"], H - fixed["Hstar"]]
    w = np.r_[weights("R", fixed["delta"], beta), weights("H", fixed["delta"], beta)]
    A = max(0., float(np.dot(w * x, y) / np.dot(w * x, x)))
    return dict(fixed, A=A, loss=float(np.sum(w * (A*x - y)**2)))


def calibration_coordinates(counts, support, utility, pseudocount=.5, betas=None):
    """Exact two-round identity for a fixed-reference exponential family.

    Conditioning on legal support preserves the exponential family. The
    half-count regularizes finite-sample logarithms only; reward and Shannon
    entropy panels continue to use the original unsmoothed observations.
    """
    betas = BETAS if betas is None else np.asarray(betas, float)
    if len(counts) != len(betas):
        raise ValueError("One nominal beta is required per checkpoint")
    probability = (counts + pseudocount * support)
    probability = probability / probability.sum(axis=-1, keepdims=True)
    log_probability = np.log(np.where(support, probability, 1.))
    reward = probability @ utility if np.ndim(utility) == 1 else (probability*utility).sum(axis=-1)
    pairs = list(combinations(range(len(betas)), 2))
    x, y = [], []
    for m, n in pairs:
        x.append(((probability[n]-probability[m]) *
                  (log_probability[n]-log_probability[m])).sum(axis=-1))
        y.append((1/betas[n]-1/betas[m]) * (reward[n]-reward[m]))
    return np.asarray(x), np.asarray(y), pairs


def calibrate_task(obj, fixed, state_ids):
    """Estimate one task-shared c with correlated errors and intrinsic scatter.

    Joint complete-row bootstrap estimates both-axis covariance and shared-
    checkpoint dependence. Source blocks are independent by sampling design.
    The same estimated measurement covariance is used in bootstrap refits.
    """
    x, y, pairs = calibration_coordinates(obj["counts"], obj["support"], obj["utility"])
    bx, by, _ = calibration_coordinates(obj["bootstrap_counts"], obj["support"], obj["utility"])
    weighted, bc, covariance = fit_with_bootstrap(x, y, bx, by)
    c = weighted["beta_scale"]
    estimator = SharedSlopeIntrinsicScatter(covariance)
    ci = np.quantile(bc, [.025, .975])
    task_name = obj["records"][0]["eval_task"]
    np.savez_compressed(SOURCE / f"calibration_covariance_{task_name}.npz",
                        **covariance, X=x, Y=y, bootstrap_X=bx, bootstrap_Y=by,
                        pairs=np.asarray(pairs), sources=np.asarray(state_ids))
    source_fits = [fit_amplitude(obj["R"][:, i], obj["H"][:, i], dict(p, beta_scale=c))
                   for i, p in enumerate(fixed)]
    bootstrap_rows = []
    for b, scale in enumerate(bc):
        amplitudes = [fit_amplitude(obj["bootstrap_R"][:, b, i], obj["bootstrap_H"][:, b, i],
                                   dict(p, beta_scale=float(scale)))["A"] for i, p in enumerate(fixed)]
        bootstrap_rows.append(dict(replicate=b, beta_scale=float(scale),
                                   loss=float(np.sum((by[:, b]-scale*bx[:, b])**2)),
                                   **{f"A_{s}": float(a) for s, a in zip(state_ids, amplitudes)}))
    for fitted in source_fits:
        fitted.update(method="fit_A_after_correlated_EIV_intrinsic_scatter_calibration",
                      calibrated_parameters=["beta_scale"],
                      source_weight_rule="Website next-order residual weights at delta/beta_effective; fixed normalized floor 0.30.")
    xci, yci = np.quantile(bx, [.025, .975], axis=1), np.quantile(by, [.025, .975], axis=1)
    rows = []
    for k, (m, n) in enumerate(pairs):
        for i, state in enumerate(state_ids):
            rows.append(dict(source=state, round_m=int(m+1), round_n=int(n+1),
                             symmetric_KL=float(x[k, i]), scaled_reward_difference=float(y[k, i]),
                             x_low=float(xci[0,k,i]), x_high=float(xci[1,k,i]),
                             y_low=float(yci[0,k,i]), y_high=float(yci[1,k,i]),
                             fitted_y=float(c*x[k,i]), residual=float(y[k,i]-c*x[k,i])))
    sensitivity = {}
    for alpha in (.25, .5, 1.):
        sx, sy, _ = calibration_coordinates(obj["counts"], obj["support"], obj["utility"], alpha)
        sbx, sby, _ = calibration_coordinates(obj["bootstrap_counts"], obj["support"], obj["utility"], alpha)
        sensitivity[str(alpha)] = SharedSlopeIntrinsicScatter(covariance_blocks(sbx, sby)).fit(
            sx, sy, initial=c)["beta_scale"]
    leave_round_out = {}
    for m in range(8):
        mask = np.asarray([m not in pair for pair in pairs])
        leave_round_out[str(m+1)] = SharedSlopeIntrinsicScatter(subset_covariance(covariance, pairs=mask)).fit(
            x[mask], y[mask], initial=c)["beta_scale"]
    leave_source_out = {}
    for i, state in enumerate(state_ids):
        mask = np.arange(len(state_ids)) != i
        leave_source_out[state] = SharedSlopeIntrinsicScatter(subset_covariance(covariance, sources=mask)).fit(
            x[:, mask], y[:, mask], initial=c)["beta_scale"]
    split_covariance = {}
    for label, slc in [("first_250", slice(0, 250)), ("last_250", slice(250, 500))]:
        split_covariance[label] = SharedSlopeIntrinsicScatter(covariance_blocks(bx[:, slc], by[:, slc])).fit(
            x, y, initial=c)["beta_scale"]
    # Verify the finite-difference identity with a nonuniform reference,
    # unequal reward levels and a known scale, without an asymptotic limit.
    q, reward, true_c = np.array([.12,.21,.67]), np.array([-.3,.2,1.1]), 2.7
    z = np.log(q)[None,:] + reward[None,:]/(true_c*BETAS[:,None])
    p = np.exp(z-z.max(axis=1,keepdims=True)); p /= p.sum(axis=1,keepdims=True)
    tx, ty, _ = calibration_coordinates(p[:,None,:], np.ones((1,3),bool), reward, 0.)
    np.testing.assert_allclose(ty, true_c*tx, rtol=2e-12, atol=1e-12)
    # Keep the proposed entropy-deficit relation as an explicit diagnostic.
    hx = BETAS[:,None]*(obj["H"]-np.array([p["Hstar"] for p in fixed]))
    hy = np.array([p["Rinf"] for p in fixed])-obj["R"]
    hc = float(np.sum(hx*hy)/np.sum(hx*hx))
    loss = float(np.sum((y-c*x)**2))
    calibration = dict(beta_scale=c, beta_scale_count_ci95=ci.tolist(),
                       method="correlated_errors_in_variables_intrinsic_scatter_zero_intercept",
                       formula="(1/beta_n-1/beta_m)*(R_legal_n-R_legal_m) = c*Jeffreys(P_n,P_m)",
                       fitted_parameters=["shared beta_scale", "shared intrinsic_scatter_perpendicular (nuisance)"], sources=state_ids,
                       pairs_per_source=len(pairs), total_points=int(x.size),
                       weights="Gaussian perpendicular-residual likelihood; V(c,tau)=[Cyy+c^2*Cxx-c*(Cxy+Cyx)]/(1+c^2)+tau^2*I; full within-source dependence and fitted intrinsic scatter.",
                       pseudocount_per_legal_target=.5, pseudocount_sensitivity=sensitivity,
                       leave_one_round_out_scales=leave_round_out,
                       leave_one_source_out_scales=leave_source_out,
                       covariance_split_sensitivity=split_covariance,
                       unweighted_baseline_beta_scale=float(np.sum(x*y)/np.sum(x*x)),
                       errors_in_variables=weighted,
                       loss=loss, rmse=float(np.sqrt(loss/x.size)),
                       centered_R2=float(1-loss/np.sum((y-y.mean())**2)),
                       entropy_deficit_diagnostic=dict(beta_scale=hc,
                           centered_R2=float(1-np.sum((hy-hc*hx)**2)/np.sum((hy-hy.mean())**2))),
                       uncertainty="500 paired complete-row count resamples, each reused for all round pairs; conditional on fixed trajectory and chosen estimator. Not model discrepancy or training-seed uncertainty.",
                       synthetic_nonuniform_reference_identity_verified=True)
    return calibration, source_fits, bootstrap_rows, rows


def weights(field, gap, betas=None):
    """Website theory weights; gap and beta must use the same scale.

    The fixed 0.30 floor regularizes normalized next-order residual scales.
    It is not a measurement standard error or a fitted intrinsic scatter.
    H uses exp(-gap/beta), without a 1/beta prefactor; R uses exp(-2*gap/beta).
    Simultaneously rescaling gap and beta leaves these weights unchanged.
    """
    beta = BETAS if betas is None else np.asarray(betas, dtype=float)
    if field not in ("R", "H") or gap < 0 or np.any(beta <= 0):
        raise ValueError("Theory weights require R/H, a nonnegative gap, and positive beta")
    order = 1 if field == "H" else 2
    # Equivalent to exp(-order*gap/beta) divided by its maximum; avoids 0/0.
    shape = np.exp(-order * gap * (1 / beta - 1 / beta.max()))
    w = 1 / (shape ** 2 + TAU ** 2)
    return w * len(beta) / w.sum()


def solve_linear(matrix, rhs):
    # Same pivoting and singularity cutoff as solveLinearSystem in app.js.
    a = np.c_[matrix, rhs].copy()
    for col in range(len(rhs)):
        pivot = col + np.argmax(np.abs(a[col:, col]))
        if abs(a[pivot, col]) < 1e-12:
            return None
        a[[col, pivot]] = a[[pivot, col]]
        a[col, col:] /= a[col, col]
        for row in range(len(rhs)):
            if row != col:
                a[row, col:] -= a[row, col] * a[col, col:]
    return a[:, -1]


def fit_joint(reward, H, beta_scale=1.):
    best = None
    step = -4.0
    while step <= 0.4:
        gap = 10 ** step
        beta_effective = beta_scale * BETAS
        tail = np.exp(-gap / beta_effective)
        X, y, w = [], [], []
        wh, wr = weights("H", gap, beta_effective), weights("R", gap, beta_effective)
        for k, beta in enumerate(beta_effective):
            X.extend([[1, 0, tail[k] / beta], [0, 1, -tail[k]]])
            y.extend([H[k], reward[k]])
            w.extend([wh[k], wr[k]])
        X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
        p = solve_linear(X.T @ (w[:, None] * X), X.T @ (w * y))
        if p is not None and np.all(np.isfinite(p)) and p[0] >= 0 and p[2] >= 0:
            loss = float(np.sum(w * (X @ p - y) ** 2))
            if best is None or loss < best["loss"]:
                best = dict(Hstar=float(p[0]), Rinf=float(p[1]), A=float(p[2]),
                            delta=float(gap), beta_scale=float(beta_scale), loss=loss,
                            fixed_parameters=[], calibrated_parameters=["beta_scale"],
                            free_parameters=["Rinf", "delta", "A", "Hstar"],
                            method="four_parameter_joint_fit_at_calibrated_beta")
        step += 0.02
    if best is None:
        raise RuntimeError("No valid joint fit; do not substitute a fallback curve.")
    return best


def predict(fit, beta):
    beta = fit.get("beta_scale", 1.) * np.asarray(beta)
    tail = np.exp(-fit["delta"] / beta)
    return fit["Rinf"] - fit["A"] * tail, fit["Hstar"] + fit["A"] * tail / beta


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data = load_data()
    tasks = {task: task_data(data, task) for task in ("T1", "T2")}
    t1 = tasks["T1"]
    fixed_tasks = {task: [fixed_source_parameters(data, task, i) for i in range(obj["R"].shape[1])]
                   for task, obj in tasks.items()}
    calibrations, calibrated_fits, candidate_rows, calibration_points = {}, {}, [], {}
    for task, obj in tasks.items():
        task_ids = [state["id"] for state in data["tasks"][task]["states"]]
        calibration, fitted, bootstrap_rows, points = calibrate_task(obj, fixed_tasks[task], task_ids)
        calibrations[task], calibrated_fits[task] = calibration, fitted
        calibration_points[task] = points
        write_csv(SOURCE / f"calibration_bootstrap_{task}.csv", bootstrap_rows)
        write_csv(SOURCE / f"calibration_points_{task}.csv", points)
        final_T = np.asarray(obj["records"][-1]["T"])
        for i, (state, fit) in enumerate(zip(task_ids, fitted)):
            R, H = obj["R"][:, i], obj["H"][:, i]
            pr, ph = predict(fit, BETAS)
            baseline = fit_amplitude(R, H, fixed_tasks[task][i])
            br, bh = predict(baseline, BETAS)
            rrmse = float(np.sqrt(np.mean((pr-R)**2)))
            hrmse = float(np.sqrt(np.mean((ph-H)**2)))
            assert np.ptp(R) > 0 and np.ptp(H) > 0
            score = float(np.sqrt(((rrmse/np.ptp(R))**2 + (hrmse/np.ptp(H))**2)/2))
            optimal = [task_ids.index(j) for j in fit["optimal_targets"]]
            candidate_rows.append(dict(task=task, source=state, index=i,
                                       beta_scale=fit["beta_scale"], A=fit["A"],
                                       reward_range=float(np.ptp(R)), H_range=float(np.ptp(H)),
                                       reward_change=float(R[-1]-R[0]), H_change=float(H[-1]-H[0]),
                                       R_rmse=rrmse, H_rmse=hrmse, normalized_joint_rmse=score,
                                       baseline_R_rmse=float(np.sqrt(np.mean((br-R)**2))),
                                       baseline_H_rmse=float(np.sqrt(np.mean((bh-H)**2))),
                                       calibrated_loss=fit["loss"], baseline_loss=baseline["loss"],
                                       final_optimal_mass=float(final_T[i, optimal].sum()/final_T[i].sum())))
        print(f"{task}: shared beta scale {calibration['beta_scale']:.6f}; count bootstrap complete", flush=True)
    # Normalize by each observed range so nearly flat states do not win
    # merely because their absolute changes are small. This is post-hoc.
    selected = min((row for row in candidate_rows if row["task"] == "T1"),
                   key=lambda row: row["normalized_joint_rmse"])
    representative = selected["index"]
    ids = [state["id"] for state in data["tasks"]["T1"]["states"]]
    state_id = data["tasks"]["T1"]["states"][representative]["id"]
    specifications = [("T1", representative, f"T1 (source {state_id})", "#6A51A3"),
                      ("T1", None, "T1 (source mean)", "#0072B2"),
                      ("T2", None, "T2 (source mean)", "#D55E00")]
    # Fontconfig sees locally installed fonts that Matplotlib's cache can miss.
    for pattern in ("Arial", "Arial:style=Bold", "Arial:style=Italic", "Arial:style=Bold Italic"):
        font_path = subprocess.check_output(["fc-match", "-f", "%{file}", pattern], text=True).strip()
        font_manager.fontManager.addfont(font_path)
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 7, "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.edgecolor": "#303030", "text.color": "#202020",
        "axes.labelcolor": "#202020", "savefig.bbox": None,
    })
    fig = plt.figure(figsize=(183 / 25.4, 112 / 25.4))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.18,1,1,1],
                            left=.082, right=.985, bottom=.12, top=.81, wspace=.55, hspace=.34)
    calax = fig.add_subplot(grid[:, 0])
    axes = np.empty((2,3), dtype=object)
    for row in range(2):
        for col in range(3):
            axes[row,col] = fig.add_subplot(grid[row,col+1], sharey=axes[row,0] if col else None)
            if col:
                axes[row,col].tick_params(labelleft=False)
    for task, color, marker in [("T1", "#0072B2", "o"), ("T2", "#D55E00", "^")]:
        points, calibration = calibration_points[task], calibrations[task]
        x = np.array([p["symmetric_KL"] for p in points])
        y = np.array([p["scaled_reward_difference"] for p in points])
        # Panel a retains its visible intervals; panels b–g omit error bars.
        # Draw percentile endpoints directly, including intervals off the estimate.
        calax.hlines(y, [p["x_low"] for p in points], [p["x_high"] for p in points],
                     color=color, lw=.35, alpha=.22, zorder=1)
        calax.vlines(x, [p["y_low"] for p in points], [p["y_high"] for p in points],
                     color=color, lw=.35, alpha=.22, zorder=1)
        calax.scatter(x,y,s=8,marker=marker,facecolors="white",edgecolors=color,
                      linewidths=.4,alpha=.55,zorder=2)
        line_x = np.array([0.,x.max()*1.02])
        c = calibration["beta_scale"]
        calax.plot(line_x,c*line_x,color=color,lw=1.2,
                   label=rf"{task}: $c={c:.2f}$",zorder=3)
    calibration_position = calax.get_position()
    top_position = axes[0, 0].get_position()
    calax.text(-.19 * top_position.width / calibration_position.width,
               1 + .055 * top_position.height / calibration_position.height, "a",
               transform=calax.transAxes, fontsize=7,
               fontweight="bold", ha="left", va="bottom")
    calax.set_title("Calibration",pad=10)
    calax.set_xlabel(r"Jeffreys divergence, $J_i^{mn}$",labelpad=5)
    calax.set_ylabel(r"$(\beta_{\mathrm{nominal},n}^{-1}-\beta_{\mathrm{nominal},m}^{-1})(R_{in}^{L}-R_{im}^{L})$",labelpad=4)
    all_points = [p for rows in calibration_points.values() for p in rows]
    xmax = max(max(p["symmetric_KL"],p["x_high"]) for p in all_points)
    ymin = min(min(p["scaled_reward_difference"],p["y_low"]) for p in all_points)
    ymax = max(max(p["scaled_reward_difference"],p["y_high"]) for p in all_points)
    calax.set_xlim(0, xmax*1.06)
    calax.set_ylim(min(0.,ymin)-.04*(ymax-ymin), ymax+.06*(ymax-ymin))
    calax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    calax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    calax.legend(loc="upper left",frameon=False,fontsize=6.5,handlelength=1.5,borderpad=.2)
    metadata = dict(evidence_sha256=hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
                    website_app_sha256=hashlib.sha256(APP.read_bytes()).hexdigest(),
                    representative_state=state_id, representative_index=representative,
                    selection_rule="Minimum root-mean-square of reward RMSE / observed reward range and entropy RMSE / observed entropy range across all T1 sources. Post-hoc representative fit; full candidate screen retained.",
                    selection_scores={row["source"]: row["normalized_joint_rmse"] for row in candidate_rows if row["task"] == "T1"},
                    beta_definition="beta_nominal=0.375/round; beta_effective=c_task*beta_nominal, with one c per task shared by all sources and the task-mean panels. Model calibration, not a direct measurement of optimizer KL strength.",
                    displayed_beta_axis="Panels b–g plot -log(beta_effective) on linear axes. Shared calibration uncertainty is retained in source data; no horizontal error bars are drawn in b–g.",
                    task_calibrations=calibrations,
                    reward_definition="sum_j T_full[i,j]*(1+u_j)-1; invalid raw reward=0; source averages uniform",
                    entropy_definition="Dimensionless Shannon entropy of legal-conditional row T_full/sum(T_full), using natural logarithms; no pseudocount",
                    fit_method="Calibrate c_task by a zero-intercept errors-in-variables likelihood with full within-source measurement covariance and fitted intrinsic perpendicular scatter. Use all sources and round pairs. Then fit source-specific A_i with registered Rmax/delta/Hstar fixed. c/f,d/g fit four coefficients conditional on the same task scale. All panels use one shared scale per task.",
                    theory_weights=dict(reward_residual_scale="exp(-2*delta/beta_effective)",
                                        entropy_residual_scale="exp(-delta/beta_effective); no 1/beta prefactor",
                                        normalized_residual="divide each residual scale by its maximum across checkpoints",
                                        rule="1/(normalized_residual**2+0.30**2), then normalize each metric to mean one",
                                        fixed_regularizer=0.30,
                                        scale_equivalence="delta/beta_effective=(delta/c)/beta_nominal",
                                        scope="same rule for source A and task-mean Rinf/Hstar/A/delta; separate from calibration likelihood"),
                    uncertainty=f"Pointwise 95% percentile intervals from {BOOTSTRAPS} row-multinomial replicates, including invalid counts, using recorded bootstrap_seed. Panel a displays horizontal and vertical calibration intervals; panels b–g omit error bars. All intervals remain in source data. No training-seed uncertainty or fit confidence band.",
                    style="Nature final artwork guide: 183 mm, Arial 6.5–7 pt, vector PDF and editable SVG; calibration panel a plus reward/entropy panels b–g.",
                    bootstrap_audit={task: obj["audits"] for task, obj in tasks.items()},
                    entropy_intervals_match_frozen_analysis=True, fits={})

    # Keep the existing calibrated fits and eight-checkpoint coordinates.
    # The residual-scale bands use each series' own fitted A and Delta; for
    # task means these are the A and Delta from the existing four-parameter
    # joint theory fit.
    panel_fits = []
    for task, index, title, color in specifications:
        obj = tasks[task]
        name = f"{task}_{state_id}" if index is not None else f"{task}_mean"
        reward = obj["R"][:, index] if index is not None else obj["R"].mean(axis=1)
        entropy_values = obj["H"][:, index] if index is not None else obj["H"].mean(axis=1)
        scale = calibrations[task]["beta_scale"]
        fit = calibrated_fits[task][index] if index is not None else fit_joint(reward, entropy_values, beta_scale=scale)
        fitted_reward, fitted_entropy = predict(fit, BETAS)
        beta_effective = scale * BETAS
        band_factor = 1.0
        panel_fits.append(dict(task=task, index=index, name=name, title=title, color=color,
                               reward=reward, entropy=entropy_values, fit=fit, scale=scale,
                               fitted_reward=fitted_reward, fitted_entropy=fitted_entropy,
                               beta_effective=beta_effective, band_factor=band_factor))
    metadata["band_definition"] = dict(
        residual_scales={"reward": "A*exp(-2*Delta/beta_effective)",
                         "entropy": "A*exp(-Delta/beta_effective)"},
        half_widths={"T1_t5": {"reward": "A_t5*exp(-2*Delta_t5/beta_effective)",
                                "entropy": "A_t5*exp(-Delta_t5/beta_effective)"},
                     "T1_mean_and_T2_mean": {"reward": "A*exp(-2*Delta/beta_effective)",
                                             "entropy": "A*exp(-Delta/beta_effective)"}},
        multipliers={"T1_t5": 1.0, "T1_mean": 1.0, "T2_mean": 1.0},
        multiplier_rationale={"all_series": "All three series use the full theoretical-remainder scales with multiplier one; each uses its existing fitted amplitude and gap."},
        series_amplitudes="Each series uses A from its existing fit: fixed-limit A_i for T1 source t5 and four-parameter joint-fit A for the T1 and T2 source means.",
        measurement_uncertainty_in_band=False,
        interpretation="Descriptive theoretical-scale envelopes, not confidence intervals.")
    metadata["uncertainty"] += " Panels b–g also show descriptive bands from the first omitted theoretical scales; these contain no measurement uncertainty and are not confidence intervals."

    source_rows, curve_rows = [], []
    row_extents = [[], []]
    dense_beta = np.geomspace(BETAS.max(), BETAS.min(), 401)
    for col, panel in enumerate(panel_fits):
        task, index, title, color = (panel[key] for key in ("task", "index", "title", "color"))
        obj = tasks[task]
        name = panel["name"]
        R, H = panel["reward"], panel["entropy"]
        br = obj["bootstrap_R"][:, :, index] if index is not None else obj["bootstrap_R"].mean(axis=2)
        bh = obj["bootstrap_H"][:, :, index] if index is not None else obj["bootstrap_H"].mean(axis=2)
        ci_r, ci_h = np.quantile(br, [.025, .975], axis=1), np.quantile(bh, [.025, .975], axis=1)
        scale, fit = panel["scale"], panel["fit"]
        band_factor = panel["band_factor"]
        pr, ph = panel["fitted_reward"], panel["fitted_entropy"]
        cr, ch = predict(fit, dense_beta)
        beta_effective = scale * BETAS
        beta_effective_dense = scale * dense_beta
        reward_theory_scale = fit["A"] * np.exp(-2 * fit["delta"] / beta_effective)
        entropy_theory_scale = fit["A"] * np.exp(-fit["delta"] / beta_effective)
        reward_band = band_factor * reward_theory_scale
        entropy_band = band_factor * entropy_theory_scale
        dense_reward_band = band_factor * fit["A"] * np.exp(-2 * fit["delta"] / beta_effective_dense)
        dense_entropy_band = band_factor * fit["A"] * np.exp(-fit["delta"] / beta_effective_dense)
        fit.update(R_rmse=float(np.sqrt(np.mean((pr-R)**2))), H_rmse=float(np.sqrt(np.mean((ph-H)**2))),
                   observed_last_reward=float(R[-1]), asymptote_below_last_reward=bool(fit["Rinf"] < R[-1]))
        metadata["fits"][name] = fit
        for k, beta in enumerate(BETAS):
            source_rows.append(dict(series=name, task=task, source=state_id if index is not None else "uniform mean",
                                    round=int(ROUNDS[k]), beta_nominal=float(beta), beta_effective=float(scale*beta),
                                    beta_effective_low=float(calibrations[task]["beta_scale_count_ci95"][0]*beta),
                                    beta_effective_high=float(calibrations[task]["beta_scale_count_ci95"][1]*beta),
                                    reward_shifted=float(R[k]),
                                    reward_low=float(ci_r[0,k]), reward_high=float(ci_r[1,k]),
                                    H_nats=float(H[k]), H_low=float(ci_h[0,k]), H_high=float(ci_h[1,k]),
                                    reward_fit=float(pr[k]), H_fit=float(ph[k]),
                                    reward_measured_residual=float(R[k]-pr[k]),
                                    H_measured_residual=float(H[k]-ph[k]),
                                    reward_theory_residual_scale=float(reward_theory_scale[k]),
                                    H_theory_residual_scale=float(entropy_theory_scale[k]),
                                    reward_band_half_width=float(reward_band[k]),
                                    H_band_half_width=float(entropy_band[k]),
                                    reward_band_low=float(pr[k]-reward_band[k]),
                                    reward_band_high=float(pr[k]+reward_band[k]),
                                    H_band_low=float(ph[k]-entropy_band[k]),
                                    H_band_high=float(ph[k]+entropy_band[k]),
                                    reward_weight=float(weights("R",fit["delta"],scale*BETAS)[k]),
                                    H_weight=float(weights("H",fit["delta"],scale*BETAS)[k])))
        for beta, beff, reward, ent, rband, hband in zip(dense_beta, beta_effective_dense, cr, ch,
                                                        dense_reward_band, dense_entropy_band):
            curve_rows.append(dict(series=name, beta_nominal=float(beta), beta_effective=float(scale*beta),
                                   reward_fit=float(reward), H_fit=float(ent),
                                   reward_band_half_width=float(rband), H_band_half_width=float(hband),
                                   reward_band_low=float(reward-rband), reward_band_high=float(reward+rband),
                                   H_band_low=float(ent-hband), H_band_high=float(ent+hband)))
        for row, (observed, ci, curve) in enumerate(((R, ci_r, cr), (H, ci_h, ch))):
            ax = axes[row, col]
            band = dense_reward_band if row == 0 else dense_entropy_band
            ax.fill_between(-np.log(scale*dense_beta), curve-band, curve+band,
                            color=color, alpha=.14, linewidth=0, zorder=1)
            ax.plot(-np.log(scale*dense_beta), curve, color=color, lw=1.25, zorder=2)
            assert np.all(ci[0] <= observed) and np.all(ci[1] >= observed)
            ax.plot(-np.log(scale*BETAS), observed, linestyle="none", marker="o", ms=3.8,
                    mfc="white", mec=color, mew=.9, zorder=3)
            lower = min(ci[0].min(), observed.min())
            upper = max(ci[1].max(), observed.max())
            row_extents[row].extend([float(lower), float(upper)])
            row_extents[row].extend([float(np.min(curve-band)), float(np.max(curve+band))])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
            ax.set_xlim(-np.log(scale*BETAS.max()*1.08), -np.log(scale*BETAS.min()/1.08))
            xmin, xmax = ax.get_xlim()
            ticks = np.arange(np.ceil(xmin), np.floor(xmax) + 1)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(int(value)) for value in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(pad=2.5)
            ax.text(-.19, 1.055, "bcdefg"[row*3+col], transform=ax.transAxes,
                    fontsize=7, fontweight="bold", ha="left", va="bottom")
            if row == 0:
                ax.set_title(title, pad=8)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$-\log\beta_{\mathrm{eff}}$", labelpad=5)
        axes[0, col].set_ylabel("Mean reward" if col == 0 else "", labelpad=5)
        axes[1, col].set_ylabel("State Shannon entropy" if col == 0 else "", labelpad=5)
    for row, extent in enumerate(row_extents):
        lower, upper = min(extent), max(extent)
        pad = max(.02, (upper-lower)*.10)
        axes[row, 0].set_ylim(lower-pad, upper+pad)
    handles = [Line2D([], [], marker="o", markersize=3.8, markerfacecolor="white",
                      markeredgecolor="#333333", linestyle="none", label="Measured"),
               Line2D([], [], color="#333333", lw=1.25,
                      label=r"Theory: $\beta_{\mathrm{eff}}=c\,\beta_{\mathrm{nominal}}$"),
               Patch(facecolor="#777777", edgecolor="none", alpha=.20,
                     label="Theoretical remainder band")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55,.984),
               ncol=3, frameon=False, columnspacing=1.4, handlelength=1.7)
    # Render once to verify that all tick and axis labels fit the full canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in [calax, *axes.flat]:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, box
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figure1_reward_entropy.{suffix}", dpi=400, facecolor="white")
    plt.close(fig)
    submission = OUT / "submission"
    submission.mkdir(exist_ok=True)
    shutil.copy2(OUT / "figure1_reward_entropy.pdf", submission / "Figure3.pdf")
    write_csv(SOURCE / "observations.csv", source_rows)
    write_csv(SOURCE / "curves.csv", curve_rows)
    write_csv(SOURCE / "candidate_state_screen.csv", candidate_rows)
    (SOURCE / "fit_parameters.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(representative=state_id, fits=metadata["fits"], figure=str(OUT / "figure1_reward_entropy.pdf")), indent=2))


if __name__ == "__main__":
    main()
