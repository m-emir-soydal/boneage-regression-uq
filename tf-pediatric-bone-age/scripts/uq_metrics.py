"""Uncertainty-quantification metrics for Gaussian-style regression UQ.

Given per-sample predictive mean and standard deviation (e.g. from MC dropout)
this module computes:

- ``picp``  : Prediction Interval Coverage Probability at confidence level alpha.
- ``mpiw``  : Mean Prediction Interval Width at confidence level alpha.
- ``regression_ece`` : two flavors of Expected Calibration Error for regression.
    - ``central`` ECE: average |alpha - PICP(alpha)| over alpha levels.
    - ``quantile`` ECE: average |q - empirical_cdf_at_q| over q levels.
- ``summarize`` : convenience that returns all of the above in a single dict
  alongside MAE, RMSE, and mean predictive std, ready to be written to CSV.

All functions are pure NumPy and assume a Gaussian predictive distribution.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.stats import norm

DEFAULT_PICP_LEVELS = (0.50, 0.80, 0.90, 0.95)
DEFAULT_ECE_LEVELS = tuple(np.round(np.linspace(0.05, 0.95, 10), 4).tolist())


def _z_for_alpha(alpha: float) -> float:
    """Two-sided z-score for a central confidence level alpha in (0, 1)."""

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(norm.ppf(0.5 + alpha / 2.0))


def picp(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, alpha: float = 0.95) -> float:
    """Prediction Interval Coverage Probability at confidence level alpha."""

    z = _z_for_alpha(alpha)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mpiw(y_std: np.ndarray, alpha: float = 0.95) -> float:
    """Mean Prediction Interval Width at confidence level alpha."""

    z = _z_for_alpha(alpha)
    return float(np.mean(2.0 * z * y_std))


def picp_interval(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction interval coverage for precomputed lower/upper bounds."""

    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mpiw_interval(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction interval width for precomputed lower/upper bounds."""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean(upper - lower))


def conformal_residual_quantile(residuals: np.ndarray, alpha: float = 0.95) -> float:
    """Split-conformal residual quantile at confidence level alpha.

    Uses the finite-sample conformal quantile:
      k = ceil((n + 1) * alpha), q_hat = k-th order statistic of residuals.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1:
        residuals = residuals.reshape(-1)
    n = int(residuals.size)
    if n == 0:
        raise ValueError("residuals must be non-empty")

    k = int(np.ceil((n + 1) * alpha))
    k = min(k, n)
    sorted_res = np.sort(residuals)
    return float(sorted_res[k - 1])


def conformal_interval(y_pred: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric split-conformal intervals around point predictions."""

    y_pred = np.asarray(y_pred, dtype=float)
    q_hat = float(q_hat)
    lower = y_pred - q_hat
    upper = y_pred + q_hat
    return lower, upper


def regression_ece_central(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    levels: Iterable[float] = DEFAULT_ECE_LEVELS,
) -> float:
    """Central-interval ECE: mean over alpha levels of |alpha - PICP(alpha)|."""

    errs = []
    for alpha in levels:
        errs.append(abs(alpha - picp(y_true, y_mean, y_std, alpha)))
    return float(np.mean(errs))


def regression_ece_quantile(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    levels: Iterable[float] = DEFAULT_ECE_LEVELS,
) -> float:
    """Quantile ECE: mean over q levels of |q - empirical_fraction(y <= mu + Phi^{-1}(q) sigma)|."""

    errs = []
    for q in levels:
        z = float(norm.ppf(q))
        threshold = y_mean + z * y_std
        empirical = float(np.mean(y_true <= threshold))
        errs.append(abs(q - empirical))
    return float(np.mean(errs))


def regression_metrics(y_true: np.ndarray, y_mean: np.ndarray) -> Dict[str, float]:
    err = y_mean - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
    }


def summarize(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    n_samples: int,
    split: str,
    picp_levels: Iterable[float] = DEFAULT_PICP_LEVELS,
    ece_levels: Iterable[float] = DEFAULT_ECE_LEVELS,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Compute a flat metrics dict ready for CSV/JSON logging.

    Returned keys:
        split, n, n_samples, mae, rmse, mean_std,
        picp@p, mpiw@p (one per p in ``picp_levels``),
        ece_central, ece_quantile,
        plus any keys from ``extra``.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    out: Dict[str, object] = {"split": split, "n": int(len(y_true)), "n_samples": int(n_samples)}
    out.update(regression_metrics(y_true, y_mean))
    out["mean_std"] = float(np.mean(y_std))
    for p in picp_levels:
        out[f"picp@{p:g}"] = picp(y_true, y_mean, y_std, p)
        out[f"mpiw@{p:g}"] = mpiw(y_std, p)
    out["ece_central"] = regression_ece_central(y_true, y_mean, y_std, ece_levels)
    out["ece_quantile"] = regression_ece_quantile(y_true, y_mean, y_std, ece_levels)
    if extra:
        out.update(extra)
    return out


def format_summary(row: Dict[str, object]) -> str:
    """Pretty single-line formatter for stdout / log files."""

    parts = [f"split={row['split']:<5} n={row['n']:>5} T={row['n_samples']:>3}"]
    parts.append(f"MAE={row['mae']:.3f}")
    parts.append(f"RMSE={row['rmse']:.3f}")
    parts.append(f"mean_std={row['mean_std']:.3f}")
    for k in sorted(k for k in row.keys() if k.startswith("picp@")):
        parts.append(f"{k}={row[k]:.3f}")
    for k in sorted(k for k in row.keys() if k.startswith("mpiw@")):
        parts.append(f"{k}={row[k]:.2f}")
    parts.append(f"ECE_central={row['ece_central']:.3f}")
    parts.append(f"ECE_quantile={row['ece_quantile']:.3f}")
    return " | ".join(parts)
