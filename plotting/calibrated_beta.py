"""Order-parameter fit with a task-specific starting-checkpoint scale offset."""
import numpy as np
from scipy.optimize import least_squares


def fit_beta_scale(beta_nominal, c, beta_i):
    """Return the fit scale; infinite nominal beta denotes the base checkpoint."""
    beta_nominal = np.asarray(beta_nominal, dtype=float)
    inverse_nominal = np.divide(1.0, beta_nominal,
                                out=np.zeros_like(beta_nominal),
                                where=np.isfinite(beta_nominal))
    return 1.0 / (inverse_nominal / float(c) + 1.0 / float(beta_i))


def remainder_weights(beta):
    beta = np.asarray(beta, dtype=float)
    scale = (beta / beta.max()) ** 3
    weights = 1.0 / (scale**2 + 0.30**2)
    return weights / weights.mean()


def fit_order_parameter(r_values, beta_nominal, c, starts=None):
    """Fit theta and beta_i with remainder weights on the adjusted fit scale."""
    r_values = np.asarray(r_values, dtype=float)
    beta_nominal = np.asarray(beta_nominal, dtype=float)
    if (r_values.shape != beta_nominal.shape or np.any(~np.isfinite(r_values))
            or np.any(np.isnan(beta_nominal)) or np.any(beta_nominal <= 0)):
        raise ValueError("r values must be finite and aligned with positive nominal betas (infinity at n=0)")
    starts = starts or [(65., .47), (350., .05), (100., .2), (1000., .02), (1000., 1.)]

    def residual(log_parameters):
        theta, beta_i = np.exp(log_parameters)
        beta = fit_beta_scale(beta_nominal, c, beta_i)
        weights = remainder_weights(beta)
        prediction = 1.0 / np.sqrt(1.0 + theta * beta**2)
        return np.sqrt(weights) * (prediction - r_values)

    trials = [least_squares(residual, np.log(start),
                            bounds=(np.log([1e-8, 1e-8]), np.log([1e8, 1e3])),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10000)
              for start in starts]
    best = min(trials, key=lambda result: float(np.dot(result.fun, result.fun)))
    theta, beta_i = np.exp(best.x)
    beta = fit_beta_scale(beta_nominal, c, beta_i)
    prediction = 1.0 / np.sqrt(1.0 + theta * beta**2)
    return dict(theta=float(theta), beta_i=float(beta_i),
                rmse=float(np.sqrt(np.mean((prediction-r_values)**2))),
                weighted_sse=float(np.dot(best.fun, best.fun)),
                weights=remainder_weights(beta).tolist(),
                beta_fit=beta.tolist(), fitted_r=prediction.tolist(),
                fit_success=bool(best.success),
                model="order_parameter_fit_with_initial_scale_offset", kappa=0., c=float(c),
                beta_scale=float(c), theta_effective=float(theta),
                beta_i_fit_scale=float(beta_i), observations=len(r_values),
                equation="r=(1+theta*beta_fit^2)^(-1/2)",
                fit_scale_equation="1/beta_fit=n/(c*beta0)+1/beta_i",
                weight_rule="1/((beta_fit/max(beta_fit))^6+0.30^2), normalised to mean one; updated with beta_i",
                interpretation="Task-specific Figure 2 fit: beta_i is the starting-checkpoint fit scale relative to a hypothetical less task-biased reference kernel; it does not alter the plotted beta_effective coordinate or any other analysis.")
