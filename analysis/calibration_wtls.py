"""Shared-slope errors-in-variables calibration with correlated errors.

X and Y have shape (pairs, sources). Bootstrap arrays have shape
(pairs, replicates, sources). Different sources are independently sampled;
the full within-source covariance retains shared-checkpoint dependence.
The default fits a positive shared slope and Gaussian perpendicular intrinsic
scatter, retaining the likelihood log determinant. Count-only weighted total
least squares is retained as a diagnostic baseline. Measurement covariance is
held fixed when refitting count-bootstrap draws.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize, minimize_scalar, brentq


def covariance_blocks(bx, by):
    bx, by = np.asarray(bx, float), np.asarray(by, float)
    if bx.shape != by.shape or bx.ndim != 3 or bx.shape[1] < 3:
        raise ValueError("Expected matching (pairs, replicates, sources) arrays")
    if not np.all(np.isfinite(bx)) or not np.all(np.isfinite(by)):
        raise ValueError("Nonfinite bootstrap coordinates")
    dx = bx - bx.mean(axis=1, keepdims=True)
    dy = by - by.mean(axis=1, keepdims=True)
    scale = bx.shape[1] - 1
    return dict(Cxx=np.einsum("kbs,lbs->skl", dx, dx) / scale,
                Cyy=np.einsum("kbs,lbs->skl", dy, dy) / scale,
                Cxy=np.einsum("kbs,lbs->skl", dx, dy) / scale)


def subset_covariance(cov, pairs=None, sources=None):
    k = np.arange(cov["Cxx"].shape[1]) if pairs is None else np.flatnonzero(pairs)
    i = np.arange(cov["Cxx"].shape[0]) if sources is None else np.flatnonzero(sources)
    return {name: value[np.ix_(i, k, k)] for name, value in cov.items()}


class SharedSlopeWTLS:
    """Functional errors-in-variables regression, one positive slope, zero intercept."""
    def __init__(self, covariance):
        self.cov = covariance
        self.cross = covariance["Cxy"] + covariance["Cxy"].transpose(0, 2, 1)

    def objective(self, log_c, x, y):
        c = float(np.exp(log_c))
        residual = (y - c * x).T
        variance = self.cov["Cyy"] + c*c*self.cov["Cxx"] - c*self.cross
        value = 0.
        for v, e in zip(variance, residual):
            factor = cho_factor((v+v.T)/2, lower=True, check_finite=False)
            value += float(e @ cho_solve(factor, e, check_finite=False))
        return value

    def fit(self, x, y, initial=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Expected matching finite coordinate arrays")
        if initial is None:
            initial = float(np.sum(x*y) / np.sum(x*x))
        if not np.isfinite(initial) or initial <= 0:
            initial = 1.
        # Scan before bounded refinement so the result does not depend on one
        # initial slope. Expand rather than silently accepting a boundary fit.
        centre = np.log(initial)
        for span in (4., 8., 12.):
            grid = np.linspace(centre-span, centre+span, 49)
            values = np.array([self.objective(z, x, y) for z in grid])
            candidates = [k for k in range(1, len(grid)-1)
                          if values[k] <= values[k-1] and values[k] <= values[k+1]]
            if candidates and 0 < values.argmin() < len(grid)-1:
                break
        else:
            raise RuntimeError("WTLS has no finite interior optimum in the search range")
        fits = [minimize_scalar(self.objective, args=(x, y),
                               bounds=(grid[k-1], grid[k+1]), method="bounded",
                               options={"xatol": 1e-9}) for k in candidates]
        best = min(fits, key=lambda result: result.fun)
        if not best.success:
            raise RuntimeError(best.message)
        c = float(np.exp(best.x))
        return dict(beta_scale=c, weighted_loss=float(best.fun),
                    search_boundary=False, candidate_minima=len(fits))

    def diagnostics(self, c):
        matrices = self.cov["Cyy"] + c*c*self.cov["Cxx"] - c*self.cross
        eigenvalues = np.array([eigh((v+v.T)/2, eigvals_only=True) for v in matrices])
        return dict(residual_covariance_min_eigenvalue=float(eigenvalues.min()),
                    residual_covariance_max_condition=float((eigenvalues[:,-1]/eigenvalues[:,0]).max()),
                    covariance_regularization="none; Cholesky solve",
                    diagnostic_covariance_scope="count-only Y-cX covariance; excludes intrinsic scatter",
                    source_blocks=len(matrices), pairs_per_block=matrices.shape[1])


class SharedSlopeIntrinsicScatter(SharedSlopeWTLS):
    """Correlated Gaussian errors-in-variables likelihood with intrinsic scatter.

    A single task-wide c defines the mean line, with zero intercept. An
    independent Gaussian perpendicular scatter tau is fitted as a nuisance
    parameter. This is the intrinsic-width line likelihood of Hogg, Bovy &
    Lang (2010), with measurement covariance extended to whole source blocks.
    The log determinant is essential: tau is inferred, not set to improve a
    preferred slope. No observations are screened by J or by residual size.
    """
    def likelihood(self, parameters, x, y):
        c = float(np.exp(parameters[0]))
        tau = float(np.exp(parameters[1]))
        normalization = 1+c*c
        residual = (y-c*x).T / np.sqrt(normalization)
        variance = ((self.cov["Cyy"] + c*c*self.cov["Cxx"] - c*self.cross)
                    / normalization + tau*tau*np.eye(x.shape[0]))
        value = 0.
        for v, e in zip(variance, residual):
            factor = cho_factor((v+v.T)/2, lower=True, check_finite=False)
            value += float(e @ cho_solve(factor, e, check_finite=False)
                           + 2*np.log(np.diag(factor[0])).sum())
        return .5*value

    def fit(self, x, y, initial=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Expected matching finite coordinate arrays")
        if initial is None:
            initial = max(float(np.sum(x*y)/np.sum(x*x)), 1e-4)
        tau0 = max(float(np.sqrt(np.mean((y-initial*x)**2)/(1+initial*initial))), 1e-5)
        candidates = [minimize(self.likelihood, np.log([initial, tau0*multiplier]),
                               args=(x, y), method="Nelder-Mead",
                               options={"xatol": 2e-7, "fatol": 1e-8, "maxiter": 700})
                      for multiplier in (.3, 1., 3.)]
        best = min(candidates, key=lambda result: result.fun)
        if not best.success:
            raise RuntimeError(best.message)
        c, tau = np.exp(best.x)
        if not np.isfinite(c+tau) or c <= 0:
            raise RuntimeError("Nonfinite intrinsic-scatter fit")
        return dict(beta_scale=float(c), intrinsic_scatter_perpendicular=float(tau),
                    negative_log_likelihood=float(best.fun),
                    search_boundary=False, optimizer_success=True)

    def profile_interval(self, x, y, point):
        log_c = np.log(point["beta_scale"])
        log_tau = np.log(point["intrinsic_scatter_perpendicular"])
        # 95% one-parameter profile likelihood: Delta(-log L)=chi2_1(.95)/2.
        target = point["negative_log_likelihood"] + 1.920729410347062
        def profile(z):
            r = minimize_scalar(lambda lt: self.likelihood([z, lt], x, y),
                                bounds=(min(-25., log_tau-10), log_tau+6),
                                method="bounded", options={"xatol": 1e-8})
            return r.fun-target
        endpoints = []
        for direction in (-1., 1.):
            step = .02
            while profile(log_c+direction*step) < 0 and step < 10:
                step *= 1.5
            if step >= 10:
                raise RuntimeError("Profile likelihood interval is unbounded")
            limits = sorted([log_c, log_c+direction*step])
            endpoints.append(float(np.exp(brentq(profile, *limits))))
        return endpoints


def fit_with_bootstrap(x, y, bx, by):
    covariance = covariance_blocks(bx, by)
    estimator = SharedSlopeIntrinsicScatter(covariance)
    point = estimator.fit(x, y)
    bootstrap_fits = [estimator.fit(bx[:, b], by[:, b], initial=point["beta_scale"])
                      for b in range(bx.shape[1])]
    draws = np.array([fit["beta_scale"] for fit in bootstrap_fits])
    point.update(estimator.diagnostics(point["beta_scale"]),
                 beta_scale_count_ci95=np.quantile(draws, [.025, .975]).tolist(),
                 beta_scale_profile_ci95=estimator.profile_interval(x, y, point),
                 intrinsic_scatter_count_ci95=np.quantile([fit["intrinsic_scatter_perpendicular"]
                     for fit in bootstrap_fits], [.025, .975]).tolist(),
                 covariance_scope="full within-source X/Y and round-pair covariance; independent source blocks",
                 covariance_bootstrap_replicates=bx.shape[1],
                 covariance_refit_policy="estimated once from joint count bootstrap; fixed in bootstrap refits",
                 intrinsic_scatter_refit_policy="refit the shared slope and intrinsic scatter jointly in every count replicate",
                 intrinsic_scatter_model="one perpendicular Gaussian variance per task, independent between latent comparison residuals")
    return point, draws, covariance
