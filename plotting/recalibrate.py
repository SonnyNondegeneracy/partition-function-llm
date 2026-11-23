"""Recompute the paper's calibrated source data with c fitted jointly on the three training seeds of each task.

c_task comes from beta_eff.pooled_c: round pairs within each of the three training runs (main experiment and two further
seeds), one shared slope, block-diagonal measurement covariance (the main run uses the archived paper covariance).
Everything else is the paper's own procedure, rerun at this c with the paper's functions:
  Fig. 2b  r fit (calibrated_beta.fit_order_parameter, rounds 0-8) and dense curve on rounds 1-8;
  Fig. 2c  scaled potential beta_eff * phi (and its interval) at round 8;
  Fig. 3   fixed-limit amplitude fit for T1 source t5 and four-parameter joint fits for the source means, with bands.
Measured r, R, H and their bootstrap intervals are copied from the frozen source data unchanged.
Output: plotting/source_data/recalibrated/{figure1,figure2}/ with the frozen file schemas (columns that depended on
the single-run calibration interval are dropped).

    python plotting/recalibrate.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from beta_eff import FROZEN, MAIN, pooled_c, order_parameter  # noqa: E402
from calibrated_beta import fit_order_parameter, fit_beta_scale  # noqa: E402
from generate_figure1 import fit_joint, fit_amplitude, predict, BETAS  # noqa: E402

OUT = HERE / "source_data" / "recalibrated"
C = {t: pooled_c(t) for t in ("T1", "T2")}


def rows(path):
    return list(csv.DictReader(open(path)))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def drop(r, keys=("beta_effective_low", "beta_effective_high")):
    return {k: v for k, v in r.items() if k not in keys}


# ---- Fig. 2b: order parameter fit ---------------------------------------------------------------------------------
obs2 = rows(FROZEN / "figure2" / "r_observations.csv"); new_obs, curves, fits = [], [], {}
for task, c in C.items():
    o = [r for r in obs2 if r["task"] == task]
    r = np.r_[order_parameter(MAIN / "base" / "4096.json", task), [float(x["r"]) for x in o]]
    nominal = np.r_[np.inf, [float(x["beta_nominal"]) for x in o]]
    fit = fit_order_parameter(r, nominal, c); fits[task] = fit
    for x in o:
        new_obs.append(drop(x) | dict(beta_effective=c * float(x["beta_nominal"]), beta_scale=c, beta_i_fit=fit["beta_i"]))
    progress = np.linspace(1., 8., 300); grid = .375 / progress; bfit = fit_beta_scale(grid, c, fit["beta_i"])
    for n, b, bf in zip(progress, grid, bfit):
        curves.append(dict(task=task, round=n, beta_nominal=b, beta_effective=c * b, beta_fit=bf, beta_scale=c,
                           fit_model=fit["model"], theta_effective=fit["theta_effective"], beta_i_fit=fit["beta_i"],
                           fitted_r=1 / np.sqrt(1 + fit["theta_effective"] * bf ** 2)))
write(OUT / "figure2" / "r_observations.csv", new_obs)
write(OUT / "figure2" / "r_fitted_curves.csv", curves)

# ---- Fig. 2c: scaled potential --------------------------------------------------------------------------------------
pot = []
for p in rows(FROZEN / "figure2" / "potential_comparison.csv"):
    k = C[p["task"]] / float(p["c"])
    pot.append(p | dict(c=C[p["task"]], beta_effective=C[p["task"]] * float(p["beta_nominal"]),
                        scaled_phi=float(p["scaled_phi"]) * k, scaled_phi_low=float(p["scaled_phi_low"]) * k,
                        scaled_phi_high=float(p["scaled_phi_high"]) * k))
write(OUT / "figure2" / "potential_comparison.csv", pot)

# ---- Fig. 3: reward and entropy fits ----------------------------------------------------------------------------------
obs1 = rows(FROZEN / "figure1" / "observations.csv")
frozen_fits = json.load(open(FROZEN / "figure1" / "fit_parameters.json"))["fits"]
dense = np.geomspace(BETAS.max(), BETAS.min(), 401); new1, cur1, fit1 = [], [], {}
for series in ("T1_t5", "T1_mean", "T2_mean"):
    o = sorted([r for r in obs1 if r["series"] == series], key=lambda r: int(r["round"]))
    task = o[0]["task"]; c = C[task]
    R = np.array([float(r["reward_shifted"]) for r in o]); H = np.array([float(r["H_nats"]) for r in o])
    if series == "T1_t5":
        f0 = frozen_fits[series]
        fit = fit_amplitude(R, H, dict(Rinf=f0["Rinf"], delta=f0["delta"], Hstar=f0["Hstar"], beta_scale=c))
    else:
        fit = fit_joint(R, H, beta_scale=c)
    fit1[series] = {k: v for k, v in fit.items() if isinstance(v, (int, float, str))}
    pr, ph = predict(fit, BETAS); be = c * BETAS
    rb, hb = fit["A"] * np.exp(-2 * fit["delta"] / be), fit["A"] * np.exp(-fit["delta"] / be)
    for k, r in enumerate(o):
        new1.append(drop(r) | dict(beta_effective=be[k], reward_fit=pr[k], H_fit=ph[k], reward_measured_residual=R[k] - pr[k],
                                   H_measured_residual=H[k] - ph[k], reward_theory_residual_scale=rb[k],
                                   H_theory_residual_scale=hb[k], reward_band_half_width=rb[k], H_band_half_width=hb[k],
                                   reward_band_low=pr[k] - rb[k], reward_band_high=pr[k] + rb[k],
                                   H_band_low=ph[k] - hb[k], H_band_high=ph[k] + hb[k]))
    cr, ch = predict(fit, dense); bd = c * dense
    drb, dhb = fit["A"] * np.exp(-2 * fit["delta"] / bd), fit["A"] * np.exp(-fit["delta"] / bd)
    for b, rr, hh, x, y in zip(dense, cr, ch, drb, dhb):
        cur1.append(dict(series=series, beta_nominal=b, beta_effective=c * b, reward_fit=rr, H_fit=hh,
                         reward_band_half_width=x, H_band_half_width=y, reward_band_low=rr - x, reward_band_high=rr + x,
                         H_band_low=hh - y, H_band_high=hh + y))
write(OUT / "figure1" / "observations.csv", new1)
write(OUT / "figure1" / "curves.csv", cur1)
json.dump(dict(c=C, method="one c per task, fitted jointly on the main experiment and two further training seeds",
               figure1_fits=fit1, figure2_fits={t: {k: v for k, v in f.items() if isinstance(v, (int, float, str))}
                                                for t, f in fits.items()}),
          open(OUT / "parameters.json", "w"), indent=1)
print("c", {t: round(v, 4) for t, v in C.items()})
for t, f in fits.items():
    print(t, "r fit theta", round(f["theta_effective"], 2), "beta_i", round(f["beta_i"], 4), "rmse", round(f["rmse"], 4))
for s, f in fit1.items():
    print(s, {k: round(f[k], 4) for k in ("A", "delta", "Rinf", "Hstar") if k in f})
