"""Shared helpers for Figs 2 and 5: counts I/O, the paper's beta_eff calibration and the paper's order parameter.

Calibration (as Fig. 3a / generate_figure1.py): (1/b_n - 1/b_m)(R_n - R_m) = c * Jeffreys(P_n, P_m) for every source state
and round pair, 0.5 pseudocounts on valid destinations, errors-in-variables fit through the origin with a full-row
multinomial bootstrap covariance. beta_eff,m = c * beta_nominal,m, fitted on all measured rounds of a trajectory.

Order parameter (Methods): T_ij = (N_ij + 0.1) / (N_i + 0.1 |L(i)|) with N_i the total sample count, least-action
potential, Pearson correlation over pairs observed in both directions. Reproduces the frozen Fig. 2 values exactly.
"""
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "analysis"))
from calibration_wtls import SharedSlopeIntrinsicScatter, covariance_blocks  # noqa: E402
from m2_kernel import fit_action_potential  # noqa: E402

DATA = Path(os.environ.get("PFL_DATA", CODE.parent / "hfdata"))   # local copy of the Hugging Face data archive
HF = DATA / "measurements"                      # released counts and frozen source data
FROZEN = HF / "source_data" / "frozen"
MAIN = HF / "counts" / "main"                    # main experiment counts (paper)
REV = HF / "counts" / "revision"                 # revision runs (KL scan, seeds, reward interpolation, Gemma)
P02A = REV / "P02a"
PREP = CODE / "revision" / "prepared"
FIGURES = Path(os.environ.get("PFL_FIGURES", CODE / "figures"))   # output folder for the figures
RECAL = Path(__file__).resolve().parent / "source_data" / "recalibrated"   # recalibrate.py (three-seed c)


def load(path, task=None):
    """Return valid counts N, invalid counts and support S from a counts.json file."""
    m = json.load(open(path))["metrics"]
    m = m[task] if task and task in m else m[next(iter(m))]
    n = m["n_states"]
    N = np.zeros((n, n), int)
    S = np.zeros((n, n), bool)
    for (i, j), c in zip(m["edges"], m["counts"]):
        N[i, j] = c
        S[i, j] = True
    return N, np.asarray(m["invalid_counts"], int), S


def rewards(prepared, n):
    lab = json.loads(json.load(open(PREP / prepared))[0]["label"])
    return np.array([lab["state_rewards"][str(k)] for k in range(n)])


def kern(N, S, a=0.5):
    """Valid-only kernel with pseudocount a (calibration convention)."""
    T = np.where(S, N + a, 0.0)
    return T / T.sum(1, keepdims=True)


def tilt(Q, S, u, beta):
    """Ideal optimum T = Q exp(u/beta) / Z (eq. optimal-kernel with shared destination rewards)."""
    L = np.where(S, np.log(np.where(S, Q, 1)) + u[None, :] / beta, -np.inf)
    L -= L.max(1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(1, keepdims=True)


def order_parameter(path, task=None):
    N, inv, S = load(path, task)
    att = N.sum(1) + inv
    T = np.where(S, (N + 0.1) / (att + 0.1 * S.sum(1))[:, None], 0.0)
    return fit_action_potential(T, support_mask=S, pair_mask=N > 0)["r_action"]


def coords(counts, support, u, binv, pc=0.5):
    """Calibration coordinates for every round pair: x = Jeffreys divergence, y = d(1/beta) dR (per source)."""
    P = counts + pc * support
    P = P / P.sum(-1, keepdims=True)
    L = np.log(np.where(support, P, 1.0))
    R = P @ u
    x, y = [], []
    for m, n in combinations(range(len(binv)), 2):
        x.append(((P[n] - P[m]) * (L[n] - L[m])).sum(-1))
        y.append((binv[n] - binv[m]) * (R[n] - R[m]))
    return np.asarray(x), np.asarray(y)


def calibrate(files, binv, u, task=None, nboot=300, seed=0):
    """Calibration factor c from counts files of all measured rounds (binv = nominal 1/beta per file)."""
    rng = np.random.default_rng(seed)
    obs, boot, S = [], [], None
    for f in files:
        N, inv, S = load(f, task)
        obs.append(N)
        n = N.sum(1) + inv
        draws = []
        for row, bad, nn in zip(N, inv, n):
            full = np.r_[row, bad]
            draws.append(rng.multinomial(nn, full / nn, size=nboot)[:, :-1])
        boot.append(np.stack(draws, axis=1))
    obs = np.asarray(obs)
    boot = np.asarray(boot)
    x, y = coords(obs, S, u, binv)
    bx, by = coords(boot, S[None, None], u, binv)
    return SharedSlopeIntrinsicScatter(covariance_blocks(bx, by)).fit(x, y)["beta_scale"]


def _blocks(files, binv, u, task, nboot, rng):
    obs, boot, S = [], [], None
    for f in files:
        N, inv, S = load(f, task)
        obs.append(N)
        n = N.sum(1) + inv
        boot.append(np.stack([rng.multinomial(nn, np.r_[row, bad] / nn, size=nboot)[:, :-1]
                              for row, bad, nn in zip(N, inv, n)], axis=1))
    x, y = coords(np.asarray(obs), S, u, binv)
    bx, by = coords(np.asarray(boot), S[None, None], u, binv)
    return x, y, covariance_blocks(bx, by)


def calibrate_pooled(trajectories, u, nboot=500, seed=0, frozen_task=None):
    """One c from several independent training runs: round pairs within each run, block-diagonal covariance.
    With frozen_task, the first trajectory (main experiment) uses the archived paper coordinates and covariance."""
    rng = np.random.default_rng(seed); parts = []
    for k, (files, binv, task) in enumerate(trajectories):
        if k == 0 and frozen_task:
            z = np.load(FROZEN / "figure1" / f"calibration_covariance_{frozen_task}.npz")
            parts.append((z["X"], z["Y"], {c: z[c] for c in ("Cxx", "Cyy", "Cxy")}))
        else:
            parts.append(_blocks(files, binv, u, task, nboot, rng))
    x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    K = [len(p[0]) for p in parts]; ns = x.shape[1]; cov = {}
    for k in ("Cxx", "Cyy", "Cxy"):
        M = np.zeros((ns, sum(K), sum(K))); o = 0
        for p, kk in zip(parts, K):
            M[:, o:o + kk, o:o + kk] = p[2][k]; o += kk
        cov[k] = M
    return SharedSlopeIntrinsicScatter(cov).fit(x, y)["beta_scale"]


def trajectories(task):
    """Main experiment and the two further training seeds of a task (8 rounds each, nominal 1/beta = m/0.375)."""
    binv = np.arange(1, 9) / 0.375
    t = [([MAIN / f"{task}_r{m:02d}" / "4096.json" for m in range(1, 9)], binv, task)]
    for sd in ("seed2", "seed3"):
        t.append(([P02A / f"E4_{task}_{sd}" / f"r{m:02d}" / "counts.json" for m in range(1, 9)], binv, None))
    return t


C_CACHE = Path(__file__).resolve().parent / "source_data" / "pooled_c.json"


def pooled_c(task):
    """Calibration factor of a task from its three training seeds jointly (cached)."""
    d = json.load(open(C_CACHE)) if C_CACHE.exists() else {}
    if task not in d:
        n, prep = {"T1": (8, "T1_main.json"), "T2": (12, "T2_main.json")}[task]
        d[task] = calibrate_pooled(trajectories(task), rewards(prep, n), frozen_task=task)
        C_CACHE.parent.mkdir(parents=True, exist_ok=True); json.dump(d, open(C_CACHE, "w"), indent=1)
    return d[task]


def frozen_c(task):
    return json.load(open(FROZEN / "figure1" / "fit_parameters.json"))["task_calibrations"][task]["beta_scale"]


def rounds(d):
    return sorted(int(p.name[1:]) for p in Path(d).glob("r0*"))
