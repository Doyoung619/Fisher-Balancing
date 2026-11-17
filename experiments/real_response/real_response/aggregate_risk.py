"""Uncentered finite-seed error second moments and descriptive uncertainty."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    # NA observations may be absent; true overflow must remain visible.
    values = values[~np.isnan(values)]
    n = len(values)
    if not n:
        return dict(mean=None, sd=None, sem=None, ci_low=None, ci_high=None)
    mean = float(values.mean())
    if not np.all(np.isfinite(values)):
        return dict(mean=mean, sd=None, sem=None, ci_low=None, ci_high=None)
    if n == 1:
        return dict(mean=mean, sd=None, sem=None, ci_low=None, ci_high=None)
    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(n)
    half = float(student_t.ppf(0.975, n - 1)) * sem
    return dict(mean=mean, sd=sd, sem=sem, ci_low=mean-half, ci_high=mean+half)


def risk_statistics(errors):
    """Q=E.T@E/S with no centering, shrinkage, or diagonal loading."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or not len(errors) or not np.all(np.isfinite(errors)):
        raise ValueError("A finite nonempty [seed,dimension] error matrix is required")
    S, d = errors.shape
    Q = errors.T @ errors / S
    Q = (Q + Q.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    bias = errors.mean(axis=0)
    centered = errors - bias
    covariance = centered.T @ centered / S
    spectrum_cov = np.linalg.eigvalsh(covariance)
    tol = max(1.0, float(eigenvalues[-1])) * 1e-10
    top = eigenvectors[:, eigenvalues >= eigenvalues[-1] - tol]
    r2 = float(np.trace(Q))
    rdir = float(eigenvalues[-1])
    if rdir < -tol or r2 < -tol:
        raise ArithmeticError("Second moment unexpectedly non-PSD")
    # max(...,0) below applies solely to roundoff in a PSD quadratic form.
    rdir, r2 = max(0.0, rdir), max(0.0, r2)
    norms = np.linalg.norm(errors, axis=1)
    loo = np.array([np.linalg.eigvalsh(np.delete(errors, j, axis=0).T @
                   np.delete(errors, j, axis=0) / (S - 1))[-1] for j in range(S)]) if S > 1 else np.array([])
    metrics = {
        "R2": float(np.sqrt(r2)), "R2_squared": r2,
        "Rdir": float(np.sqrt(rdir)), "Rdir_squared": rdir,
        "mean_L2": float(norms.mean()), "Q_rank": int(np.sum(eigenvalues > tol)),
        "Q_top_eigengap": float(eigenvalues[-1] - eigenvalues[-2]) if d > 1 else float(eigenvalues[-1]),
        "Q_trace": r2, "bias_norm_squared": float(bias @ bias),
        "centered_covariance_trace": float(np.trace(covariance)),
        "centered_covariance_lambda_max": float(spectrum_cov[-1]),
        "top_eigenspace_dimension": int(top.shape[1]),
        "top_eigenvalue_degenerate": bool(top.shape[1] > 1),
        "Rdir_squared_loo_min": float(loo.min()) if len(loo) else None,
        "Rdir_squared_loo_max": float(loo.max()) if len(loo) else None,
    }
    arrays = dict(Q=Q, Q_eigenvalues=eigenvalues, top_eigenvector=eigenvectors[:, -1],
                  top_eigenspace_projector=top @ top.T, bias=bias, centered_covariance=covariance,
                  Rdir_squared_leave_one_out=loo)
    return metrics, arrays


def bootstrap_risks(errors, indices):
    """Whole-seed resampling; caller shares the indices across paired methods."""
    errors, indices = np.asarray(errors), np.asarray(indices)
    sampled = errors[indices]
    Q = np.einsum("bsd,bse->bde", sampled, sampled) / errors.shape[0]
    r2_squared = np.trace(Q, axis1=1, axis2=2)
    rdir_squared = np.maximum(np.linalg.eigvalsh(Q)[:, -1], 0)
    return dict(R2_squared=r2_squared, R2=np.sqrt(np.maximum(r2_squared, 0)),
                Rdir_squared=rdir_squared, Rdir=np.sqrt(rdir_squared),
                mean_L2=np.linalg.norm(sampled, axis=-1).mean(axis=1))
